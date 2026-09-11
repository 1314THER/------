#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题三：径向网格数 N、时间步长 Δt 与水分方程口径的对照求解器。

用途：复现论文第 10、11 节的两张对照表（表 10 与表 11）。

与主求解器 solve_p234.py 的关系
------------------------------
本脚本**不重复实现**离散格式，而是直接复用 solve_p234 中的
物性经验式（props_p23）、空气条件插值（air_conditions）、
三对角组装（_assemble）与追赶法（thomas）。这样做的目的是保证
对照实验与论文主结果**逐位同源**：唯一被改变的量就是被测的那个因素
（N、Δt 或水分方程口径），而不是"另写一套代码"。

两种水分方程口径
----------------
1. form="cons"（本文口径，守恒形式）
       d(rho*C)/dt = (1/r) * d/dr ( rho*D*r*dC/dr )
   其中 rho = 650 + 128*C 取自附录 3。累计项系数为
       accum = d(rho*C)/dC = rho + C*drho/dC，
   界面通量系数为 rho*D。等价扩散系数为 rho*D/(rho + C*rho')。

2. form="fick"（经典 Fick 口径）
       dC/dt = (1/r) * d/dr ( D*r*dC/dr )
   即不含 rho。两式的差别只在水分方程内部，温度方程完全相同，
   因此可以直接比较"烘干时长"这一单一输出。

两者的物理差别与选取理由见论文 §5.9（口径的闭合）与 §8.6.5（口径敏感性）。

跑法
----
    PY=/Users/sqz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3

    # 单组求解：N=200、dt=5 s、本文口径
    $PY 代码/p23_variants.py --n 200 --dt 5 --form cons

    # 复现论文表 10 与表 11（并行，约 5 分钟）
    $PY 代码/p23_variants.py --table

    # 只跑一小部分用于自检
    $PY 代码/p23_variants.py --table --n-list 20,50 --dt-list 5

输出：控制台打印烘干时长（中心含水率首次低于 0.15 kg/kg 的时刻）。
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solve_p234 as base  # noqa: E402


def solve_variant(form="cons", thermo="coupled", n=200, dt=5.0,
                  tend=259200.0, picard=8, tol=1e-11):
    """按指定口径求解问题三，返回中心达标时刻。

    参数
    ----
    form   : "cons"（本文守恒口径）或 "fick"（经典 Fick 口径）
    thermo : "coupled" 解耦合温度方程；"iso_air" 令温度场等于烘房温度
             （用于检验"热质解耦"结论）；"iso50" 全程 50 °C
    n      : 径向控制体个数（求解网格，不是输出网格）
    dt     : 时间步长 [s]
    tend   : 最长时间 [s]

    返回
    ----
    dict: 中心达标时刻 t_center、表面达标时刻 t_surface、以及若干时刻快照
    """
    props = base.props_p23

    # ---- 贴体网格与有限体积权重（与 solve_p234 完全一致）----
    h = 1.0 / n
    xi = np.arange(n + 1) * h
    xif = (np.arange(n) + 0.5) * h
    weight = xi * h
    weight[0] = h * h / 8.0          # 中心控制体
    weight[n] = h * (1.0 - h / 4.0) / 2.0   # 表面控制体
    radius = base.R0                 # 问题三半径固定

    temp = np.full(n + 1, base.T_INIT)
    conc = np.full(n + 1, base.C_INIT)
    n_steps = int(round(tend / dt))
    t_center = t_surface = None
    snaps = {}

    for step in range(1, n_steps + 1):
        t_now = step * dt
        t_inf, c_inf = base.air_conditions(t_now)
        conc_old = conc.copy()
        temp_old = temp.copy()

        # ---- Picard 迭代：滞后物性，把非线性方程化为线性三对角 ----
        for _ in range(picard):
            if thermo == "coupled":
                rho, drho, cp, k_cond, diff = props(conc, temp)
            else:
                t_set = t_inf if thermo == "iso_air" else 50.0
                temp = np.full(n + 1, t_set)
                rho, drho, cp, k_cond, diff = props(conc, temp)

            # ---------------- 水分方程 ----------------
            if form == "fick":
                # dC/dt = (1/r) d/dr ( D r dC/dr )
                accum = np.ones_like(conc)
                mass_face = 0.5 * (diff[:-1] + diff[1:])
                alpha = dt * radius * base.H_MASS
            else:
                # d(rho C)/dt = (1/r) d/dr ( rho D r dC/dr )
                accum = rho + conc * drho
                mass_face = (0.5 * (rho[:-1] + rho[1:])) * (
                    0.5 * (diff[:-1] + diff[1:])
                )
                alpha = dt * accum[n] * radius * base.H_MASS
            lam = accum * weight * radius ** 2
            fc = dt * mass_face * xif / h
            adv = np.zeros(n + 1)            # 问题三不收缩，无对流项
            lo, di, up, rhs = base._assemble(
                lam, fc, adv, conc_old, alpha, c_inf, n
            )
            conc_new = base.thomas(lo, di, up, rhs)

            # ---------------- 温度方程 ----------------
            if thermo == "coupled":
                rho_cp = rho * cp
                heat_face = 0.5 * (k_cond[:-1] + k_cond[1:])
                lam_t = rho_cp * weight * radius ** 2
                fc_t = dt * heat_face * xif / h
                alpha_t = dt * radius * base.H_HEAT
                lo_t, di_t, up_t, rhs_t = base._assemble(
                    lam_t, fc_t, adv, temp_old, alpha_t, t_inf, n
                )
                temp_new = base.thomas(lo_t, di_t, up_t, rhs_t)
            else:
                temp_new = temp

            change = max(
                float(np.max(np.abs(conc_new - conc))),
                float(np.max(np.abs(temp_new - temp))),
            )
            temp, conc = temp_new, conc_new
            if change < tol:
                break

        # ---- 记录达标时刻与快照 ----
        if t_center is None and conc[0] < base.C_DRY:
            t_center = t_now
        if t_surface is None and conc[-1] < base.C_DRY:
            t_surface = t_now
        for hh in (6.0, 12.0, 24.0, 48.0):
            if hh not in snaps and t_now >= hh * 3600:
                snaps[hh] = (t_now, conc.copy(), temp.copy())
        if t_center is not None:
            break

    return dict(form=form, thermo=thermo, n=n, dt=dt, t_center=t_center,
                t_surface=t_surface, snaps=snaps)


def _job(args):
    """给 multiprocessing.Pool 用的单组任务包装。"""
    form, n, dt = args
    r = solve_variant(form=form, n=n, dt=dt)
    return form, n, dt, r["t_center"], r["t_surface"]


def build_tasks(n_list, dt_list, n_ref=100, dt_ref=5.0):
    """构造两张表所需的 (口径, N, Δt) 任务集合。

    表 10 固定 Δt=dt_ref、让 N 变；表 11 固定 N=n_ref、让 Δt 变。
    两者的公共点 (n_ref, dt_ref) 只跑一次。
    """
    pairs = {(n, dt_ref) for n in n_list} | {(n_ref, dt) for dt in dt_list}
    return [(f, n, dt) for f in ("cons", "fick") for n, dt in sorted(pairs)]


def run_table(n_list, dt_list, jobs, n_ref=100, dt_ref=5.0):
    """并行跑完全部对照任务，返回结果列表。"""
    tasks = build_tasks(n_list, dt_list, n_ref, dt_ref)
    with Pool(processes=min(jobs, len(tasks))) as pool:
        out = pool.map(_job, tasks)
    return out


def _fmt(rows, n_list, dt_list, n_ref=100, dt_ref=5.0):
    """打印两张表：表 10（N 的影响）与表 11（Δt 的影响）。"""
    index = {(f, n, dt): v for f, n, dt, v, _ in rows}
    head = ("%5s %8s %10s %9s %10s %9s %9s %9s"
            % ("N", "dt/s", "本文/s", "本文/h", "Fick/s", "Fick/h",
               "差值/s", "相对"))
    lines = []
    for title, cases in (
        ("表 10　径向网格数的影响（Δt=%g s）" % dt_ref,
         [(n, dt_ref) for n in n_list]),
        ("表 11　时间步长的影响（N=%d）" % n_ref,
         [(n_ref, dt) for dt in dt_list]),
    ):
        lines += ["", title, head]
        for n, dt in cases:
            cons = index.get(("cons", n, dt))
            fick = index.get(("fick", n, dt))
            d = fick - cons
            lines.append("%5d %8g %10.1f %9.3f %10.1f %9.3f %9.1f %8.2f%%"
                         % (n, dt, cons, cons / 3600.0, fick, fick / 3600.0,
                            d, 100.0 * d / cons))
    return lines


def main():
    ap = argparse.ArgumentParser(description="问题三 网格/时间步/口径 对照")
    ap.add_argument("--form", default="cons", choices=("cons", "fick"))
    ap.add_argument("--thermo", default="coupled",
                    choices=("coupled", "iso_air", "iso50"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dt", type=float, default=5.0)
    ap.add_argument("--tend", type=float, default=259200.0)
    ap.add_argument("--table", action="store_true", help="跑完整对照表")
    ap.add_argument("--n-list", default="20,50,100,200,400")
    ap.add_argument("--dt-list", default="2,5,10,20,60,120")
    ap.add_argument("--n-ref", type=int, default=100)
    ap.add_argument("--dt-ref", type=float, default=5.0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    if not a.table:
        r = solve_variant(form=a.form, thermo=a.thermo, n=a.n, dt=a.dt,
                          tend=a.tend)
        print("form=%s thermo=%s N=%d dt=%g" % (a.form, a.thermo, a.n, a.dt))
        print("  中心达标 %8.1f s = %6.3f h" % (r["t_center"],
                                                r["t_center"] / 3600.0))
        print("  表面达标 %8.1f s = %6.3f h" % (r["t_surface"],
                                                r["t_surface"] / 3600.0))
        return

    n_list = [int(v) for v in a.n_list.split(",")]
    dt_list = [float(v) for v in a.dt_list.split(",")]
    rows = run_table(n_list, dt_list, a.jobs, a.n_ref, a.dt_ref)
    for line in _fmt(rows, n_list, dt_list, a.n_ref, a.dt_ref):
        print(line)
    if a.csv:
        with open(a.csv, "w") as fh:
            fh.write("form,N,dt_s,t_center_s,t_surface_s\n")
            for form, n, dt, tc, ts in rows:
                fh.write("%s,%d,%g,%.1f,%.1f\n" % (form, n, dt, tc, ts))
        print("已保存: %s" % a.csv)


if __name__ == "__main__":
    main()
