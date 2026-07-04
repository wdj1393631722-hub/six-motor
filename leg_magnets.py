#!/usr/bin/env python3
"""每条腿可控磁力（足底电磁吸附）。

实体机器人每条腿足底带磁铁，吸附时单腿约 50kg 保持力，抬脚（不接触）时无磁力。
本模块在 MuJoCo 里用「外力」复现这一行为，并提供每条腿磁力的使能开关：

  - 每条腿一个使能开关 enabled[leg]（True=通电吸附，False=断电无磁）。
  - 只有当「该腿通电」且「足底正贴在地面/坡面上」时，才对该足施加吸附力；
    抬脚悬空（无接触）时即使通电也没有磁力——磁铁离开铁面自然吸不住。
  - 吸附力沿接触面法向压向地面，大小 = force_kg·g（默认 50kg ≈ 490N）。
    这会把该足牢牢按在面上，增大摩擦/抓地、防滑、并让机器人能贴在斜面上。

用法（在仿真主循环里）：
    from leg_magnets import LegMagnets
    magnets = LegMagnets(model, data, force_kg=50.0, start_enabled=True)
    ...
    magnets.apply()              # 每次 mj_step 之前调用一次
    mujoco.mj_step(model, data)

    magnets.toggle(3)            # 切换第 3 条腿磁力使能
    magnets.enable_all() / magnets.disable_all() / magnets.toggle_all()
    print(magnets.status_str())  # 打印各腿使能/吸附状态

足底几何体名约定为 legN_foot_pad（见 SIX-MOTOR_sim.xml / SIX-MOTOR_slope.xml）。
"""
from __future__ import annotations

import numpy as np
import mujoco

G = 9.80665  # m/s^2

# 足底吸附状态对应的着色（仅视觉提示，不影响物理）
RGBA_HOLD = (0.15, 0.85, 0.25, 0.95)   # 通电且已吸附：绿色
RGBA_LIVE = (1.00, 0.70, 0.10, 0.90)   # 通电但悬空未吸附：橙色
# 断电时恢复各足原始颜色（构造时记录）。


class LegMagnets:
    """六足足底可控磁力控制器。"""

    def __init__(
        self,
        model,
        data,
        *,
        force_kg: float = 50.0,
        legs=(1, 2, 3, 4, 5, 6),
        start_enabled: bool = True,
        colorize: bool = True,
    ):
        self.model = model
        self.data = data
        self.legs = tuple(legs)
        self.force_N = float(force_kg) * G
        self._force_kg = float(force_kg)
        self.colorize = colorize

        # 解析各腿足底 geom → 所属 body，并记录原始颜色
        self.foot_gid: dict[int, int] = {}   # leg -> foot geom id
        self.leg_of_gid: dict[int, int] = {}  # foot geom id -> leg
        self.body_id: dict[int, int] = {}     # leg -> tibia body id
        self._orig_rgba: dict[int, np.ndarray] = {}
        for leg in self.legs:
            gid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"leg{leg}_foot_pad"
            )
            if gid < 0:
                raise ValueError(f"未找到足底几何体 leg{leg}_foot_pad")
            self.foot_gid[leg] = gid
            self.leg_of_gid[gid] = leg
            self.body_id[leg] = int(model.geom_bodyid[gid])
            self._orig_rgba[leg] = model.geom_rgba[gid].copy()

        self.enabled: dict[int, bool] = {leg: bool(start_enabled) for leg in self.legs}
        self._attached: dict[int, bool] = {leg: False for leg in self.legs}

    # ---- 使能控制 ----------------------------------------------------------
    def set(self, leg: int, on: bool) -> None:
        if leg in self.enabled:
            self.enabled[leg] = bool(on)

    def enable(self, leg: int) -> None:
        self.set(leg, True)

    def disable(self, leg: int) -> None:
        self.set(leg, False)

    def toggle(self, leg: int) -> bool:
        if leg in self.enabled:
            self.enabled[leg] = not self.enabled[leg]
            return self.enabled[leg]
        return False

    def enable_all(self) -> None:
        for leg in self.legs:
            self.enabled[leg] = True

    def disable_all(self) -> None:
        for leg in self.legs:
            self.enabled[leg] = False

    def toggle_all(self) -> bool:
        """任一腿通电则全部断电，否则全部通电。返回切换后的整体状态。"""
        on = not any(self.enabled.values())
        for leg in self.legs:
            self.enabled[leg] = on
        return on

    def set_force_kg(self, kg: float) -> None:
        self._force_kg = float(kg)
        self.force_N = float(kg) * G

    @property
    def force_kg(self) -> float:
        return self._force_kg

    def attached_legs(self) -> list[int]:
        """当前真正吸住地面的腿（通电且接触面）。"""
        return [leg for leg in self.legs if self._attached[leg]]

    def follow_gait(self, gait, release=(0.15, 0.80), min_hold=3) -> None:
        """磁力‑步态联动：支撑腿吸附，摆动腿仅在摆动"核心段"释放磁力。

        这是让机器人能**爬墙**的关键。每帧（gait.step 之后、mj_step 之前）调用一次。

        对每条腿取其摆动进度 u∈[0,1]（gait.swing_state）：
          - 支撑相 → 通电吸住表面（每时刻 3 条支撑腿托住机身对抗重力）；
          - 摆动相且 u∈release（默认 0.15~0.80，即抬脚正中段）→ 断电释放，让足端
            顺利离面前摆；
          - 摆动相但 u<release[0]（刚要抬脚）或 u>release[1]（即将落地）→ 仍通电。
        "两端仍通电"很关键：将落地的腿提前恢复磁力（悬空零力、一触面即抓），
        与将抬脚的腿的释放**时序重叠**，避免三角步态换步瞬间同时脱附而整体掉落。

        min_hold: **接触门控**——只有在"释放这些腿之后仍有 ≥min_hold 条腿正吸住
        表面"时才真正释放；否则本帧不释放任何腿（全部通电吸牢），等抓地恢复再释放。
        这保证任何时刻贴墙的吸附腿数不低于 min_hold，转向/换步不会瞬间全脱附掉下来
        （抓地不足时退化为"全吸牢"，自愈、绝不掉墙）。设 0 关闭门控。

        步态停止不动时所有腿都判为支撑 → 全部通电吸牢。
        release: (start,end) 摆动进度释放窗口；窗口越窄越"黏"（更稳但更难迈步）。
        """
        r0, r1 = release
        want_off = [
            leg for leg in self.legs
            if (lambda sw_u: sw_u[0] and r0 <= sw_u[1] <= r1)(gait.swing_state(leg))
        ]
        if min_hold and want_off:
            # 释放 want_off 后仍在吸附的腿（上一步 apply 的接触状态）
            holding = [
                leg for leg in self.legs
                if self._attached[leg] and leg not in want_off
            ]
            if len(holding) < min_hold:
                want_off = []   # 抓地不足 → 本帧全部吸牢，等落地腿抓稳再释放
        off = set(want_off)
        for leg in self.legs:
            self.set(leg, leg not in off)

    # ---- 每步施加磁力 ------------------------------------------------------
    def apply(self) -> None:
        """在每次 mj_step 之前调用：按当前接触与使能状态施加足底吸附外力。

        对本控制器管理的每个 tibia body，先清零其外力再重新计算，避免残留。
        """
        model, data = self.model, self.data

        # 清零本模块所拥有的足腿外力（force+torque，世界系，作用于 body 质心）
        for bid in self.body_id.values():
            data.xfrc_applied[bid, :] = 0.0
        for leg in self.legs:
            self._attached[leg] = False

        if any(self.enabled.values()):
            # 逐接触聚合：每条腿把它与地面/坡面的所有接触点合成一个等效接触
            agg = {
                leg: {"n": np.zeros(3), "p": np.zeros(3), "c": 0} for leg in self.legs
            }
            for k in range(data.ncon):
                con = data.contact[k]
                g1, g2 = con.geom1, con.geom2
                for g, other in ((g1, g2), (g2, g1)):
                    leg = self.leg_of_gid.get(g)
                    if leg is None or not self.enabled[leg]:
                        continue
                    # 只吸附「世界系」几何体（floor 平面、ship_slope 高程面），
                    # 排除腿与腿之间的自碰撞。
                    if model.geom_bodyid[other] != 0:
                        continue
                    a = agg[leg]
                    a["n"] += con.frame[0:3]
                    a["p"] += con.pos
                    a["c"] += 1

            for leg in self.legs:
                a = agg[leg]
                if a["c"] == 0:
                    continue  # 通电但悬空/未触面 → 无磁力
                bid = self.body_id[leg]
                p = a["p"] / a["c"]           # 等效接触点（世界系）
                n = a["n"] / a["c"]           # 平均接触法向
                nn = np.linalg.norm(n)
                n = n / nn if nn > 1e-9 else np.array([0.0, 0.0, 1.0])
                # 取指向足端一侧的外法向，再反向 → 压向接触面（吸附方向）
                if np.dot(n, data.xipos[bid] - p) < 0.0:
                    n = -n
                force = -self.force_N * n
                # 力作用在接触点，需相对 body 质心补一个力矩 r×F
                r = p - data.xipos[bid]
                data.xfrc_applied[bid, 0:3] = force
                data.xfrc_applied[bid, 3:6] = np.cross(r, force)
                self._attached[leg] = True

        if self.colorize:
            self._update_colors()

    def _update_colors(self) -> None:
        for leg in self.legs:
            gid = self.foot_gid[leg]
            if not self.enabled[leg]:
                self.model.geom_rgba[gid] = self._orig_rgba[leg]
            elif self._attached[leg]:
                self.model.geom_rgba[gid] = RGBA_HOLD
            else:
                self.model.geom_rgba[gid] = RGBA_LIVE

    # ---- 状态展示 ----------------------------------------------------------
    def status_str(self) -> str:
        cells = []
        for leg in self.legs:
            if not self.enabled[leg]:
                mark = "○"          # 断电
            elif self._attached[leg]:
                mark = "●"          # 通电已吸附
            else:
                mark = "◍"          # 通电悬空
            cells.append(f"{leg}{mark}")
        n_on = sum(self.enabled.values())
        n_hold = len(self.attached_legs())
        return (
            f"磁力[{' '.join(cells)}] 通电{n_on}/6 吸附{n_hold} "
            f"单腿{self._force_kg:.0f}kg  (●吸附 ◍悬空 ○断电)"
        )
