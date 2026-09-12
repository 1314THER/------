#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题三的第三种水分方程口径：干基密度守恒。

论文表 11 列了三种口径，但 `p23_variants.py` 只实现了前两种（经典 Fick 与
体积密度守恒），第三行 "变密度守恒口径（等效 D→(1+C)D）→ 51.39 h" 一直
没有对应的实现，属于"文中有数、代码无源"。本脚本把它补上。

三种口径（几何同为固定半径 R0，温度方程完全相同）：

  1. fick  ∂C/∂t = (1/r)∂_r(r D ∂_rC)
     accum = 1,              face = D
  2. cons  ∂(ρC)/∂t = (1/r)∂_r(r ρ D ∂_rC)          ρ = a+bC
     accum = ρ + Cρ',        face = ρD
  3. rhod  ∂(ρ_dC)/∂t = (1/r)∂_r(r ρ_d D ∂_rC)      ρ_d = ρ/(1+C)
     accum = ρ_d + Cρ_d',    face = ρ_d D
     其中 ρ_d' = dρ_d/dC = (b-a)/(1+C)²

表面第三类边界沿用与 cons 口径相同的写法（alpha = dt·accum_N·R·h_m），
以保证三种口径之间只有"累积项与通量项用什么密度"这一个差别。

自检：本脚本给出的 fick / cons 必须与 输出/问题三-网格与口径对照.csv
逐位一致，否则说明驱动方式与主对照求解器不同源。

跑法：
    PY=/Users/sqz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
    $PY 代码/closure_rhod.py --n 100 --dt 5 --form rhod
    $PY 代码/closure_rhod.py --all            # 三种口径 × 两套网格
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT = PROJECT / "输出"
sys.path.insert(0, str(HERE))

import solve_p234 as base                                # noqa: E402

A3, B3 = 650.0, 128.0                                    # 附录 3 的 ρ=a+bC


def solve(form="rhod", n=100, dt=5.0, tend=259200.0, picard=8, tol=1e-11,
          bc="accum"):
    """按指定口径求解问题三（固定半径），返回中心首次达标时刻。"""
    h = 1.0 / n
    xi = np.arange(n + 1) * h
    xif = (np.arange(n) + 0.5) * h
    weight = xi * h
    weight[0] = h * h / 8.0
    weight[n] = h * (1.0 - h / 4.0) / 2.0
    radius = base.R0

    temp = np.full(n + 1, base.T_INIT)
    conc = np.full(n + 1, base.C_INIT)
    n_steps = int(round(tend / dt))

    for step in range(1, n_steps + 1):
        t_now = step * dt
        t_inf, c_inf = base.air_conditions(t_now)
        conc_old, temp_old = conc.copy(), temp.copy()

        for _ in range(picard):
            rho, drho, cp, k_cond, diff = base.props_p23(conc, temp)
            # ---- 水分方程：三种口径只差 accum 与 face ----
            if form == "fick":
                accum = np.ones_like(conc)
                mass_face = 0.5 * (diff[:-1] + diff[1:])
            else:
                if form == "cons":
                    rho_r, drho_r = rho, drho
                elif form == "rhod":
                    c = np.maximum(conc, 1e-12)
                    rho_r = rho / (1.0 + c)
                    drho_r = drho / (1.0 + c) - rho / (1.0 + c) ** 2
                else:
                    raise ValueError(form)
                accum = rho_r + conc * drho_r
                mass_face = (0.5 * (rho_r[:-1] + rho_r[1:])) * (
                    0.5 * (diff[:-1] + diff[1:]))
            if form == "fick":
                alpha = dt * radius * base.H_MASS
            elif bc == "accum":
                alpha = dt * accum[n] * radius * base.H_MASS
            else:                       # 用表面处的密度本身作 Robin 系数
                alpha = dt * rho_r[n] * radius * base.H_MASS

            lam = accum * weight * radius ** 2
            fc = dt * mass_face * xif / h
            lo, di, up, rhs = base._assemble(
                lam, fc, conc_old, alpha, c_inf, n)
            conc_new = base.thomas(lo, di, up, rhs)

            # ---- 温度方程（三种口径完全相同）----
            rho_cp = rho * cp
            heat_face = 0.5 * (k_cond[:-1] + k_cond[1:])
            lam_t = rho_cp * weight * radius ** 2
            fc_t = dt * heat_face * xif / h
            alpha_t = dt * radius * base.H_HEAT
            lo_t, di_t, up_t, rhs_t = base._assemble(
                lam_t, fc_t, temp_old, alpha_t, t_inf, n)
            temp_new = base.thomas(lo_t, di_t, up_t, rhs_t)

            change = max(float(np.max(np.abs(conc_new - conc))),
                         float(np.max(np.abs(temp_new - temp))))
            temp, conc = temp_new, conc_new
            if change < tol:
                break

        if conc[0] < base.C_DRY:
            return t_now
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", default="rhod",
                    choices=("fick", "cons", "rhod"))
    ap.add_argument("--bc", default="accum", choices=("accum", "rho"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--dt", type=float, default=5.0)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if not args.all:
        t = solve(form=args.form, n=args.n, dt=args.dt, bc=args.bc)
        print("%s 口径 N=%d dt=%g : t_dry = %.0f s = %.2f h"
              % (args.form, args.n, args.dt, t, t / 3600))
        return

    print("三种口径对照（固定半径，附录 3 物性）")
    print("%-6s %-5s %10s %8s %9s" % ("口径", "N", "t_dry/s", "t_dry/h", "相对 Fick"))
    ref = {}
    for n in (100, 200):
        for form in ("fick", "cons", "rhod"):
            t = solve(form=form, n=n, dt=5.0)
            if form == "fick":
                ref[n] = t
            print("%-6s %-5d %10.0f %8.2f %+8.2f%%"
                  % (form, n, t, t / 3600, 100 * (t / ref[n] - 1)))
        print()

    # 与既有对照 CSV 交叉验证（只在前两种口径上）
    csvp = OUT / "问题三-网格与口径对照.csv"
    if csvp.exists():
        tab = {(r["form"], int(r["N"]), float(r["dt_s"])): float(r["t_center_s"])
               for r in csv.DictReader(csvp.open(encoding="utf-8"))}
        print("与 输出/问题三-网格与口径对照.csv 的交叉验证：")
        for form in ("fick", "cons"):
            for n in (100, 200):
                mine = solve(form=form, n=n, dt=5.0)
                theirs = tab[(form, n, 5.0)]
                flag = "一致" if abs(mine - theirs) < 1e-6 else "★不一致"
                print("   %-5s N=%-4d 本脚本 %8.0f s ；CSV %8.0f s  %s"
                      % (form, n, mine, theirs, flag))


if __name__ == "__main__":
    main()
