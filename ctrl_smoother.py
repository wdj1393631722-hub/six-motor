#!/usr/bin/env python3
"""关节控制目标一阶平滑，抑制步态/防滑修正带来的高频抖动。"""
from __future__ import annotations

import math
from typing import Dict


class CtrlSmoother:
    def __init__(self, tau: float = 0.14):
        self.tau = max(1e-3, float(tau))
        self._prev: Dict[str, float] = {}

    def reset(self, pose: Dict[str, float] | None = None) -> None:
        self._prev = dict(pose) if pose else {}

    def filter(self, targets: Dict[str, float], dt: float) -> Dict[str, float]:
        if dt <= 0:
            return dict(targets)
        alpha = 1.0 - math.exp(-dt / self.tau)
        out: Dict[str, float] = {}
        for jn, val in targets.items():
            prev = self._prev.get(jn, val)
            merged = prev + alpha * (float(val) - prev)
            out[jn] = merged
            self._prev[jn] = merged
        return out
