#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题三：关键参数 (h, hm, D) ±20% 灵敏度（经典 Fick 口径）。

输出论文"参数灵敏度表"的数值。基准与网格/时间步收敛表一致：
N=100、Δt=5 s、Fick 口径。

运行:
    PY=/Users/sqz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
    $PY 代码/sens.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solve_p234 as base  # noqa: E402


def run(hscale=1.0, hmscale=1.0, dscale=1.0, n=100, dt=5.0):
    """按扰动后的物性重解问题三，返回烘干时长（秒）。"""
    H0, HM0 = base.H_HEAT, base.H_MASS
    orig = base.PROPERTY_SETS["p23"]

    def props(conc, temp):
        rho, drho, cp, k, d = orig(conc, temp)
        return rho, drho, cp, k, d * dscale

    base.H_HEAT = H0 * hscale
    base.H_MASS = HM0 * hmscale
    base.PROPERTY_SETS["p23"] = props
    try:
        sol = base.solve(prop="p23", shrink=False, n=n, dt=dt, tend=259200.0)
        return sol["dry_time"]
    finally:
        base.H_HEAT, base.H_MASS = H0, HM0
        base.PROPERTY_SETS["p23"] = orig


def main():
    cases = [
        ("基准值", 1.0, 1.0, 1.0),
        ("h -20%", 0.8, 1.0, 1.0),
        ("h +20%", 1.2, 1.0, 1.0),
        ("hm -20%", 1.0, 0.8, 1.0),
        ("hm +20%", 1.0, 1.2, 1.0),
        ("D -20%", 1.0, 1.0, 0.8),
        ("D +20%", 1.0, 1.0, 1.2),
    ]
    base_t = None
    for label, hs, hms, ds in cases:
        t = run(hs, hms, ds)
        if base_t is None:
            base_t = t
        rel = 100.0 * (t - base_t) / base_t
        print("%-9s %8.0f s %8.3f h %+7.2f%%" % (label, t, t / 3600.0, rel))


if __name__ == "__main__":
    main()
