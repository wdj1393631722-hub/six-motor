#!/usr/bin/env python3
"""
独立自包含的关节空间三角步态（足底始终与地面平行接触）。

设计要点
========
1. 三角组交替：A=(1,3,5) 与 B=(2,4,6) 半周期错相，始终 3 腿支撑。
2. 关节空间控制：直接生成 18 个关节角目标，不依赖 IK / physics_gait /
   stance_lock / body_stabilizer 等耦合模块。
3. 足底平面平行约束（关键）：每条腿的 femur 与 tibia 关节共用同一条水平
   旋转轴，脚掌相对地面的俯仰角只取决于 (theta_femur + theta_tibia)。
   因此只要全程保持 (theta_femur + theta_tibia) = 站立标定值，脚掌就始终
   与地面平行 —— 抬脚时 femur 偏置多少，tibia 就反向补偿多少。
4. 真正"行走"而非摩擦拖行：摆动腿明显抬离地面再向前落下；支撑腿足掌平贴
   地面，仅靠 coxa 绕竖直轴转动推进机身（平面接触，绕竖直轴微转不打滑）。
5. 前进 / 后退 / 左转 / 右转：把机身期望速度投影到每条腿 coxa 可达的切向
   方向，自动得到每条腿的 coxa 摆动幅度与符号。

腿号：1 左前 | 2 左中 | 3 左后 | 4 右后 | 5 右中 | 6 右前
前进轴：base_link +Y
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import mujoco
import numpy as np

TRIPOD_A: Tuple[int, ...] = (1, 3, 5)
TRIPOD_B: Tuple[int, ...] = (2, 4, 6)
LEG_JOINTS = ("coxa", "femur", "tibia")


@dataclass
class _LegCal:
    """单腿标定结果。"""

    hip_xy: np.ndarray              # 髋座在 base 系的水平位置 (2,)
    stand_foot_xy: np.ndarray       # 站立足端在 base 系的水平位置 (2,)
    stand_foot_z: float             # 站立足端在 base 系的高度
    coxa_tangent: np.ndarray        # coxa 正向单位增量 → 足端水平位移方向 (2,)
    lift_femur_dir: float           # 抬脚对应的 femur 偏置符号 (+1/-1)
    femur_per_meter: float          # 抬高 1m 所需的 femur 偏置幅度 (rad/m)
    sum_ft_stand: float             # 站立 (femur+tibia)，脚掌平行的守恒量
    jinv: np.ndarray                # 足端位移(base xyz) → 关节增量(coxa,femur,tibia) 的伪逆 (3,3)


@dataclass
class PlanarTripodGait:
    """关节空间三角步态规划器（足底平行）。"""

    model: mujoco.MjModel
    stand_pose: Dict[str, float]
    body_height: float

    cycle_time: float = 0.45        # 一个完整步态周期 (s)；linear 模式下速度≈vy·duty 与之无关，
                                    #   故加大周期=步幅变大+抬脚同步加高，步态更自然且不降速
    duty: float = 0.5               # 支撑相占比（0.5 = 标准三角步态）
    lift_height: float = 0.090      # 摆动相抬脚高度 (m)，抬高更自然、过障更稳、蹭地更少
    height_comp_m: float = 0.055    # 前馈机身抬升补偿 (m)，抬高机身+抵消行走下沉（净高约117mm）
    stride_gain: float = 13.0       # 机身速度 → coxa 摆幅增益（步幅）；偏大使各腿摆幅都吃满上限→左右对称
    max_coxa_swing: float = 0.80    # 单腿 coxa 峰峰摆幅上限 (rad)，站立±0.329 时峰值 0.729 < 限位 0.8
    swing_smooth_tau: float = 0.04  # 摆幅指令一阶平滑时间常数 (s)
    # --- 直线后蹬模式 ('linear') 参数 ---
    gait_mode: str = "linear"       # 'linear' 足端机身系直线后蹬(快、无scrub) | 'arc' 旧版纯coxa画弧
    max_stride: float = 0.28        # 单腿足端单周期最大水平位移 (m)，限幅防越限/打滑
    verbose: bool = True

    # --- 运行时状态（非入参）---
    _phase: float = field(default=0.0, init=False)
    _amp: Dict[int, float] = field(default_factory=dict, init=False)
    _svec: Dict[int, np.ndarray] = field(default_factory=dict, init=False)
    _cal: Dict[int, _LegCal] = field(default_factory=dict, init=False)
    _qadr: Dict[Tuple[int, str], int] = field(default_factory=dict, init=False)
    _jrange: Dict[Tuple[int, str], Tuple[float, float]] = field(
        default_factory=dict, init=False
    )
    _foot_gid: Dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._data = mujoco.MjData(self.model)
        for leg in range(1, 7):
            for j in LEG_JOINTS:
                jn = f"leg{leg}_{j}_joint"
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                self._qadr[(leg, j)] = int(self.model.jnt_qposadr[jid])
                lo, hi = self.model.jnt_range[jid]
                self._jrange[(leg, j)] = (float(lo), float(hi))
            gid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"leg{leg}_foot_pad"
            )
            self._foot_gid[leg] = int(gid)
            self._amp[leg] = 0.0
            self._svec[leg] = np.zeros(2)
        self._calibrate()

    # ----------------------------------------------------------------- 标定
    def _set_base(self) -> None:
        self._data.qpos[:] = 0.0
        self._data.qvel[:] = 0.0
        self._data.qpos[2] = self.body_height
        self._data.qpos[3] = 1.0  # 单位四元数，机身水平、无偏航

    def _apply_stand(self) -> None:
        for leg in range(1, 7):
            for j in LEG_JOINTS:
                self._data.qpos[self._qadr[(leg, j)]] = self.stand_pose[
                    f"leg{leg}_{j}_joint"
                ]

    def _foot_xy(self, leg: int) -> np.ndarray:
        return self._data.geom_xpos[self._foot_gid[leg]][:2].copy()

    def _foot_z(self, leg: int) -> float:
        return float(self._data.geom_xpos[self._foot_gid[leg]][2])

    def _forward_with(self, leg: int, dc: float, df: float, dt_: float) -> None:
        """在站立位基础上对单腿施加关节偏置并正解。"""
        self._set_base()
        self._apply_stand()
        self._data.qpos[self._qadr[(leg, "coxa")]] += dc
        self._data.qpos[self._qadr[(leg, "femur")]] += df
        self._data.qpos[self._qadr[(leg, "tibia")]] += dt_
        mujoco.mj_forward(self.model, self._data)

    def _calibrate(self) -> None:
        """数值标定每条腿的 coxa 切向、抬脚方向与灵敏度。"""
        if self.verbose:
            print("══ 平面三角步态标定 ══", flush=True)
        eps_c = 0.05   # coxa 探测增量 (rad)
        eps_f = 0.08   # femur 探测增量 (rad)
        for leg in range(1, 7):
            f0 = self.stand_pose[f"leg{leg}_femur_joint"]
            t0 = self.stand_pose[f"leg{leg}_tibia_joint"]
            sum_ft = f0 + t0

            # 站立基准
            self._forward_with(leg, 0.0, 0.0, 0.0)
            foot0 = self._foot_xy(leg)
            foot0_z = self._foot_z(leg)
            hip = np.array(_HIP_XY[leg], dtype=float)

            # 足端 3D 雅可比 J = d(foot_xyz)/d(coxa,femur,tibia)，有限差分
            eps_j = 0.03
            cols = []
            for k, (dc, df, dt_) in enumerate(
                ((eps_j, 0, 0), (0, eps_j, 0), (0, 0, eps_j))
            ):
                self._forward_with(leg, dc, df, dt_)
                p = np.array(
                    [*self._foot_xy(leg), self._foot_z(leg)], dtype=float
                )
                self._forward_with(leg, -dc, -df, -dt_)
                pm = np.array(
                    [*self._foot_xy(leg), self._foot_z(leg)], dtype=float
                )
                cols.append((p - pm) / (2.0 * eps_j))
            jac = np.column_stack(cols)  # (3,3): 列=各关节贡献的足端位移
            jinv = np.linalg.pinv(jac)

            # coxa 切向：coxa +eps_c 时足端水平位移方向
            self._forward_with(leg, eps_c, 0.0, 0.0)
            foot_c = self._foot_xy(leg)
            tang = foot_c - foot0
            n = float(np.linalg.norm(tang))
            tang = tang / n if n > 1e-9 else np.array([0.0, 0.0])

            # 抬脚方向：femur +eps_f / tibia -eps_f （保持 femur+tibia 守恒 → 脚掌平行）
            self._forward_with(leg, 0.0, eps_f, -eps_f)
            z_plus = self._foot_z(leg)
            self._forward_with(leg, 0.0, -eps_f, eps_f)
            z_minus = self._foot_z(leg)
            self._forward_with(leg, 0.0, 0.0, 0.0)
            z0 = self._foot_z(leg)

            # 选脚抬高的 femur 方向
            if z_plus >= z_minus:
                lift_dir = 1.0
                dz = z_plus - z0
            else:
                lift_dir = -1.0
                dz = z_minus - z0
            dz_per_rad = abs(dz) / eps_f if eps_f > 1e-9 else 1e-6
            femur_per_m = 1.0 / max(dz_per_rad, 1e-6)

            self._cal[leg] = _LegCal(
                hip_xy=hip,
                stand_foot_xy=foot0,
                stand_foot_z=foot0_z,
                coxa_tangent=tang,
                lift_femur_dir=lift_dir,
                femur_per_meter=femur_per_m,
                sum_ft_stand=sum_ft,
                jinv=jinv,
            )
            if self.verbose:
                print(
                    f"  leg{leg}: coxa切向=({tang[0]:+.2f},{tang[1]:+.2f}) "
                    f"抬脚femur方向={lift_dir:+.0f} "
                    f"灵敏度={dz_per_rad*100:.1f} cm/rad "
                    f"(femur+tibia)={sum_ft:+.3f}",
                    flush=True,
                )

    # ------------------------------------------------------- 抬脚偏置（带限位）
    def _lift_femur_delta(self, leg: int, lift_m: float) -> float:
        """脚相对机身净上移量(m，可正可负) → femur 偏置（夹紧限位，保持脚掌平行）。"""
        cal = self._cal[leg]
        delta = cal.lift_femur_dir * lift_m * cal.femur_per_meter
        f0 = self.stand_pose[f"leg{leg}_femur_joint"]
        t0 = self.stand_pose[f"leg{leg}_tibia_joint"]
        f_lo, f_hi = self._jrange[(leg, "femur")]
        t_lo, t_hi = self._jrange[(leg, "tibia")]
        # femur += delta, tibia -= delta（守恒），两端都不得越限
        hi = min(f_hi - f0, t0 - t_lo)
        lo = max(f_lo - f0, t0 - t_hi)
        return float(np.clip(delta, lo, hi))

    # ------------------------------------------------------------------ 主循环
    def _desired_foot_dir(
        self, leg: int, vx: float, vy: float, omega: float
    ) -> Tuple[float, float]:
        """该腿足端相对机身的期望位移方向·幅度（投影到 coxa 切向，带符号）。"""
        cal = self._cal[leg]
        r = cal.hip_xy
        # 机身期望线速度（世界 xy）：前进沿 +Y
        v_body = np.array([vx, vy], dtype=float)
        # 偏航引起的髋座切向速度： omega * (ẑ × r) = omega*(-ry, rx)
        v_turn = omega * np.array([-r[1], r[0]])
        # 足端相对机身速度 = -(机身平移 + 偏航切向)
        v_foot = -(v_body + v_turn)
        proj = float(np.dot(v_foot, cal.coxa_tangent))
        return proj, 0.0

    def _coxa_amp_target(self, leg: int, vx: float, vy: float, omega: float) -> float:
        proj, _ = self._desired_foot_dir(leg, vx, vy, omega)
        amp = self.stride_gain * proj
        return float(np.clip(amp, -self.max_coxa_swing, self.max_coxa_swing))

    def reset(self) -> None:
        self._phase = 0.0
        for leg in range(1, 7):
            self._amp[leg] = 0.0
            self._svec[leg] = np.zeros(2)

    def _foot_stride_target(self, leg: int, vx: float, vy: float, omega: float) -> np.ndarray:
        """该腿单周期足端水平位移向量(base xy)：机身系直线后蹬方向·幅度。"""
        cal = self._cal[leg]
        r = cal.hip_xy
        v_body = np.array([vx, vy], dtype=float)
        v_turn = omega * np.array([-r[1], r[0]])
        v_foot = -(v_body + v_turn)                 # 足端相对机身速度（机身系）
        stride = v_foot * (self.duty * self.cycle_time)
        n = float(np.linalg.norm(stride))
        if n > self.max_stride:
            stride = stride * (self.max_stride / n)
        return stride

    @staticmethod
    def _smoothstep(u: float) -> float:
        u = max(0.0, min(1.0, u))
        return u * u * (3.0 - 2.0 * u)

    def _leg_swinging(self, leg: int, phase: float) -> Tuple[bool, float]:
        """返回 (是否摆动相, 该相内归一化进度 u∈[0,1])。"""
        in_a = leg in TRIPOD_A
        # A 组：[0,duty) 支撑, [duty,1) 摆动；B 组相反（错相半周期）
        local = phase if in_a else (phase + 0.5) % 1.0
        if local < self.duty:
            return False, local / max(self.duty, 1e-6)          # 支撑
        return True, (local - self.duty) / max(1.0 - self.duty, 1e-6)  # 摆动

    def step(
        self,
        dt: float,
        vx: float = 0.0,
        vy: float = 0.0,
        omega: float = 0.0,
    ) -> Dict[str, float]:
        """推进一帧，返回 18 个关节目标角 {joint_name: rad}。"""
        moving = abs(vx) > 1e-6 or abs(vy) > 1e-6 or abs(omega) > 1e-6

        # 摆幅/步幅指令平滑（避免命令突变导致蹬地）
        alpha = 1.0 - math.exp(-dt / max(self.swing_smooth_tau, 1e-4))
        for leg in range(1, 7):
            tgt = self._coxa_amp_target(leg, vx, vy, omega) if moving else 0.0
            self._amp[leg] += alpha * (tgt - self._amp[leg])
            svec_tgt = (
                self._foot_stride_target(leg, vx, vy, omega)
                if moving
                else np.zeros(2)
            )
            self._svec[leg] += alpha * (svec_tgt - self._svec[leg])

        if moving:
            self._phase = (self._phase + dt / max(self.cycle_time, 1e-6)) % 1.0

        if self.gait_mode == "linear":
            return self._emit_linear(moving)

        comp = self.height_comp_m  # 前馈抬升：脚在地固定时把机身顶高
        out: Dict[str, float] = {}
        for leg in range(1, 7):
            f0 = self.stand_pose[f"leg{leg}_femur_joint"]
            t0 = self.stand_pose[f"leg{leg}_tibia_joint"]
            c0 = self.stand_pose[f"leg{leg}_coxa_joint"]
            amp = self._amp[leg]

            # lift_m = 脚相对机身的净上移量（脚坐标）。
            #   支撑腿伸长(-comp) → 脚相对机身下移 → 机身被顶高 comp。
            #   摆动腿离地 = lift·sin(πu)，叠加 -comp 基准 → 净离地 (lift-comp)。
            if not moving:
                # 静止：全腿落地站立（coxa 回中），支撑腿伸长抬机身
                coxa = c0
                lift_m = -comp
            else:
                swinging, u = self._leg_swinging(leg, self._phase)
                if swinging:
                    # 摆动相：脚抬空中，coxa 由前位(+amp/2) 复位到后位(-amp/2)
                    coxa = c0 + (amp / 2.0 - amp * self._smoothstep(u))
                    lift_m = self.lift_height * math.sin(
                        math.pi * max(0.0, min(1.0, u))
                    ) - comp
                else:
                    # 支撑相：脚平贴地，coxa 由后位(-amp/2) 扫到前位(+amp/2) 推进机身
                    coxa = c0 + (-amp / 2.0 + amp * self._smoothstep(u))
                    lift_m = -comp

            fd = self._lift_femur_delta(leg, lift_m)
            femur = f0 + fd
            tibia = t0 - fd              # 守恒 femur+tibia → 脚掌平行

            out[f"leg{leg}_coxa_joint"] = self._clamp(leg, "coxa", coxa)
            out[f"leg{leg}_femur_joint"] = self._clamp(leg, "femur", femur)
            out[f"leg{leg}_tibia_joint"] = self._clamp(leg, "tibia", tibia)
        return out

    def _emit_linear(self, moving: bool) -> Dict[str, float]:
        """直线后蹬模式：足端在机身系走直线，雅可比解三关节增量。

        支撑相：脚平贴地面，从前位(-stride/2)直线扫到后位(+stride/2)推进机身。
        摆动相：脚抬离地面，从后位直线收回前位准备下一蹬。
        机身抬升 comp：支撑脚目标下压 comp → 机身被顶高。
        """
        comp = self.height_comp_m
        out: Dict[str, float] = {}
        for leg in range(1, 7):
            cal = self._cal[leg]
            f0 = self.stand_pose[f"leg{leg}_femur_joint"]
            t0 = self.stand_pose[f"leg{leg}_tibia_joint"]
            c0 = self.stand_pose[f"leg{leg}_coxa_joint"]
            stride = self._svec[leg]  # base xy 位移向量（指向后蹬方向）

            if not moving:
                d_xy = np.zeros(2)
                d_z = -comp
            else:
                swinging, u = self._leg_swinging(leg, self._phase)
                s = self._smoothstep(u)
                if swinging:
                    # 摆动：后位(+1/2)→前位(-1/2)，抬脚 lift·sin
                    d_xy = stride * (0.5 - s)
                    d_z = self.lift_height * math.sin(
                        math.pi * max(0.0, min(1.0, u))
                    ) - comp
                else:
                    # 支撑：前位(-1/2)→后位(+1/2)，贴地
                    d_xy = stride * (s - 0.5)
                    d_z = -comp

            # 足端目标位移 (base xyz) → 关节增量
            dxyz = np.array([d_xy[0], d_xy[1], d_z], dtype=float)
            dq = cal.jinv @ dxyz
            coxa = c0 + dq[0]
            femur = f0 + dq[1]
            tibia = t0 + dq[2]
            out[f"leg{leg}_coxa_joint"] = self._clamp(leg, "coxa", coxa)
            out[f"leg{leg}_femur_joint"] = self._clamp(leg, "femur", femur)
            out[f"leg{leg}_tibia_joint"] = self._clamp(leg, "tibia", tibia)
        return out

    def _clamp(self, leg: int, j: str, v: float) -> float:
        lo, hi = self._jrange[(leg, j)]
        return float(np.clip(v, lo, hi))

    @property
    def phase(self) -> float:
        return self._phase

    def phase_label(self) -> str:
        a_sw, _ = self._leg_swinging(1, self._phase)
        return "A摆动/B支撑" if a_sw else "A支撑/B摆动"

    # ---- 摆动/支撑查询（供磁力‑步态联动等外部使用）----------------------
    def is_swing(self, leg: int, lead: float = 0.0) -> bool:
        """第 leg 条腿当前是否处于摆动相（足端抬离/前摆）。

        lead: 相位提前量 ∈[0,1)。>0 时相当于"提前 lead 个相位"判断，
        可用于抬脚前提前释放磁力（避免足端被磁钉住却要抬脚而较劲）。
        """
        sw, _ = self._leg_swinging(leg, (self._phase + lead) % 1.0)
        return sw

    def swing_state(self, leg: int) -> Tuple[bool, float]:
        """返回第 leg 条腿当前 (是否摆动相, 相内进度 u∈[0,1])。

        u 是摆动/支撑子相内的归一化进度：摆动时 u 从 0（刚抬脚）到 1（将落地），
        足端抬升高度 ∝ sin(π·u)（u=0.5 最高）。磁力‑步态联动用它判断"何时释放"。
        """
        return self._leg_swinging(leg, self._phase)

    def swing_legs(self, lead: float = 0.0) -> list:
        """当前（提前 lead 相位后）处于摆动相的腿号列表。"""
        return [leg for leg in range(1, 7) if self.is_swing(leg, lead)]

    def stance_legs(self, lead: float = 0.0) -> list:
        """当前（提前 lead 相位后）处于支撑相的腿号列表。"""
        return [leg for leg in range(1, 7) if not self.is_swing(leg, lead)]


# 髋座在 base 系的水平位置（与 robot_params_for_gait.json 一致）
_HIP_XY: Dict[int, Tuple[float, float]] = {
    1: (0.1, 0.16209),
    6: (-0.1, 0.16209),
    2: (0.16, 0.002094),
    5: (-0.16, 0.002094),
    3: (0.1, -0.15791),
    4: (-0.1, -0.15791),
}


__all__ = ["PlanarTripodGait", "TRIPOD_A", "TRIPOD_B"]
