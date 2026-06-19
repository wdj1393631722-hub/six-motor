#!/usr/bin/env python3
"""机身 IMU：MuJoCo site 传感器 + 姿态/角速度/加速度读取。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import mujoco
import numpy as np

IMU_SITE_NAME = "imu_site"
IMU_GYRO_NAME = "imu_gyro"
IMU_ACC_NAME = "imu_acc"

# 机体坐标系 IMU 安装点（base_link 系，约上板中心）
IMU_SITE_POS = (0.0, 0.0, 0.042)


@dataclass
class ImuState:
    roll: float
    pitch: float
    gyro: np.ndarray  # body frame rad/s (wx, wy, wz)
    acc: np.ndarray  # body frame m/s² (含重力投影)
    yaw: float = 0.0


class ImuBinding:
    """绑定 MuJoCo IMU 传感器地址；无传感器时回退到 root 状态。"""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        prefix: str = "",
        root_qposadr: int = 0,
        root_dofadr: int = 0,
        base_body_id: Optional[int] = None,
    ):
        self.prefix = prefix
        self.root_qposadr = int(root_qposadr)
        self.root_dofadr = int(root_dofadr)
        self.base_body_id = base_body_id
        self._gyro_adr: Optional[int] = None
        self._acc_adr: Optional[int] = None
        gname = f"{prefix}{IMU_GYRO_NAME}"
        aname = f"{prefix}{IMU_ACC_NAME}"
        try:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, gname)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, aname)
            self._gyro_adr = int(model.sensor_adr[gid])
            self._acc_adr = int(model.sensor_adr[aid])
        except Exception:
            pass

    @property
    def has_sensors(self) -> bool:
        return self._gyro_adr is not None and self._acc_adr is not None

    def read(self, data: mujoco.MjData) -> ImuState:
        adr = self.root_qposadr
        dof = self.root_dofadr
        q = data.qpos[adr + 3 : adr + 7]
        roll, pitch, yaw = _quat_rpy(q)
        if self.base_body_id is not None:
            R = data.xmat[self.base_body_id].reshape(3, 3)
            v_body = R.T @ data.qvel[dof : dof + 3]
            w_body = R.T @ data.qvel[dof + 3 : dof + 6]
        else:
            R = _quat_to_mat(q)
            v_body = R.T @ data.qvel[dof : dof + 3]
            w_body = R.T @ data.qvel[dof + 3 : dof + 6]

        if self.has_sensors:
            gyro = np.array(
                data.sensordata[self._gyro_adr : self._gyro_adr + 3], dtype=np.float64
            )
            acc = np.array(
                data.sensordata[self._acc_adr : self._acc_adr + 3], dtype=np.float64
            )
        else:
            gyro = w_body.astype(np.float64)
            # 近似：线加速度 = R^T * (a_world - g)，用 qacc 近似
            acc = np.array([0.0, 0.0, 9.81], dtype=np.float64)

        return ImuState(
            roll=float(roll),
            pitch=float(pitch),
            gyro=gyro,
            acc=acc,
            yaw=float(yaw),
        )


def _quat_to_mat(q) -> np.ndarray:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_rpy(q) -> Tuple[float, float, float]:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def imu_obs_vector(imu: ImuState) -> np.ndarray:
    """RL 观测：roll, pitch, gyro(3), acc(3)。"""
    return np.concatenate(
        [
            np.array([imu.roll, imu.pitch], dtype=np.float32),
            imu.gyro.astype(np.float32),
            (imu.acc * 0.1).astype(np.float32),
        ]
    ).astype(np.float32)


def imu_level_cost(imu: ImuState) -> float:
    """越小越水平。"""
    return float(imu.roll * imu.roll + imu.pitch * imu.pitch)


def patch_imu_xml(xml: str) -> str:
    """在 base_link 上添加 IMU site 与传感器（幂等）。"""
    import re

    if 'name="imu_site"' in xml:
        return xml

    site = (
        f'      <site name="{IMU_SITE_NAME}" pos="'
        f'{IMU_SITE_POS[0]:.4f} {IMU_SITE_POS[1]:.4f} {IMU_SITE_POS[2]:.4f}" '
        f'size="0.010" rgba="0.95 0.25 0.15 0.85"/>\n'
    )
    xml = re.sub(
        r'(<body name="base_link"[^>]*>\s*)',
        r"\1" + site,
        xml,
        count=1,
    )

    sensor_block = f"""  <sensor>
    <gyro name="{IMU_GYRO_NAME}" site="{IMU_SITE_NAME}"/>
    <accelerometer name="{IMU_ACC_NAME}" site="{IMU_SITE_NAME}"/>
  </sensor>
"""
    if "<sensor>" in xml:
        if IMU_GYRO_NAME not in xml:
            xml = xml.replace(
                "  </sensor>",
                f'    <gyro name="{IMU_GYRO_NAME}" site="{IMU_SITE_NAME}"/>\n'
                f'    <accelerometer name="{IMU_ACC_NAME}" site="{IMU_SITE_NAME}"/>\n'
                "  </sensor>",
                1,
            )
    else:
        xml = xml.replace("</mujoco>", sensor_block + "</mujoco>", 1)
    return xml
