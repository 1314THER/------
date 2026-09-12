#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§11「收缩的建模判据与代价」的全部计算。

三块内容
--------
1. **判据（解析）**：收缩几何只由斜率截距比 r = b/a 决定。给定半径容差 ε 与
   工作窗口 [C_min, C0]，忽略收缩安全的充要条件是 r ≥ r*(ε, C_min)，等价地
   C_min ≥ C*(ε, r)。两式都有闭式解，按各向异性指数 n 分两支
   （n=2 径向等长、n=3 各向同性）。

2. **对照（数值）**：固定附录 3 的物性，只切换"几何"这一个开关，看烘干时长
   变多少：
       fixed    —— 固定半径 R0（本文问题二/三的做法）
       measured —— 附件 2 的实测 R(t)（问题四的做法）
       implied  —— 由 ρ=a+bC 反演的 R(C̄)，自洽闭合（本文构造的第三条路）
   implied 用不动点迭代：先按固定半径解出 C̄(t)，由 r 反演 R(t)，再解一次，
   迭代若干轮至 R 稳定。

3. **阈值（数值）**：固定 a=650 与附录 3 的 D、c_p、k，只改斜率 b（因而改 r），
   对每个 r 求 t_fixed 与 t_implied，得到 Δ(r) = t_fixed/t_implied − 1，
   再取 Δ 达到 5%/10%/20% 的 r 值，即**时长口径**的阈值 r*_time。

跑法：
    PY=/Users/sqz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
    $PY 代码/shrink_criterion.py --analytic          # 只算闭式判据（秒级）
    $PY 代码/shrink_criterion.py --matrix --jobs 4   # 加几何对照（约 3 分钟）
    $PY 代码/shrink_criterion.py --sweep --jobs 4    # 加 b 扫描（约 15 分钟）
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT = PROJECT / "输出"
sys.path.insert(0, str(HERE))

import solve_p234 as base                                # noqa: E402

C0 = base.C_INIT
R0 = base.R0
A3, B3 = 650.0, 128.0


# ====================== 解析判据 ======================
def v_ratio(C, r, c0=C0):
    """V(C)/V(C0)，只含 r。"""
    C = np.asarray(C, dtype=float)
    return (1.0 + C) * (1.0 + r * c0) / ((1.0 + c0) * (1.0 + r * C))


def line_shrink(C, r, n, c0=C0):
    """线收缩 1 − R/R0；n=2 径向等长，n=3 各向同性。"""
    return 1.0 - v_ratio(C, r, c0) ** (1.0 / n)


def r_star(eps, Cmin, n, c0=C0):
    """忽略收缩安全的阈值：r ≥ r* 时窗口内线收缩不超过 ε。"""
    g = (1 - eps) ** n * (1 + c0) / (1 + Cmin)
    return (g - 1.0) / (c0 - g * Cmin)


def C_star(eps, r, n, c0=C0):
    """临界含水率：C 降到 C* 以下，忽略收缩的线收缩超过 ε。"""
    G = (1 - eps) ** n * (1 + c0) / (1 + r * c0)
    return (G - 1.0) / (1.0 - G * r)


def beta_of(r, a):
    """与线性收缩模型对应的端点收缩系数。"""
    return (1000.0 / a) * (1 - r) / (1 + r * C0)


# ====================== 求解封装 ======================
def _fv_weights(xi):
    h = xi[1] - xi[0]
    w = xi * h
    w[0] = h * h / 8.0
    w[-1] = h * (1.0 - h / 4.0) / 2.0
    return w


def solve_with_radius(t_tab, R_tab, prop="p23", n=100, dt=5.0, tend=259200.0):
    """把自定义的 R(t) 注入求解器（贴体材料坐标，无对流项）。"""
    old_t, old_v = base._RAD_T, base._RAD_V
    t_ext = np.r_[t_tab, tend]
    R_ext = np.r_[R_tab, R_tab[-1]]
    base._RAD_T, base._RAD_V = np.asarray(t_ext, float), np.asarray(R_ext, float)
    try:
        return base.solve(prop=prop, shrink=True, n=n, dt=dt, tend=tend)
    finally:
        base._RAD_T, base._RAD_V = old_t, old_v


def mean_moisture(sol):
    """体积加权平均含水率 C̄(t)。"""
    w = _fv_weights(sol["xi"])
    return (sol["conc"] * w).sum(axis=1) / w.sum()


def implied_radius(sol, a, b, n):
    """由 ρ=a+bC 反演 R(C̄)，并强制单调不增（收缩不可逆）。"""
    Cbar = mean_moisture(sol)
    r = b / a
    R = R0 * v_ratio(Cbar, r) ** (1.0 / n)
    return Cbar, np.minimum.accumulate(R)


def closure_solve(a, b, prop="p23", n=100, dt=5.0, iters=3, geom_n=2):
    """自洽闭合：固定半径起步，反复用 C̄ 反演 R(t) 直到稳定。"""
    sol = base.solve(prop=prop, shrink=False, n=n, dt=dt)
    hist = []
    for _ in range(iters):
        Cbar, R = implied_radius(sol, a, b, geom_n)
        old = sol["dry_time"]
        sol = solve_with_radius(sol["times"], R * 1.0, prop=prop, n=n, dt=dt)
        hist.append(sol["dry_time"])
        if old is not None and sol["dry_time"] is not None and \
                abs(sol["dry_time"] - old) / old < 1e-3:
            break
    return sol, hist


def parse_criterion():
    """复算结果文件里的判据表，供正文引用。"""
    spec = {
        "预热窗口 C 2.55→2.0": 2.0,
        "问题二窗口 2.55→1.364": 1.364,
        "全程 2.55→0.15": 0.15,
    }
    out = {}
    for name, Cmin in spec.items():
        out[name] = {}
        for e in (0.03, 0.05, 0.10):
            out[name]["eps=%.0f%%" % (100 * e)] = {
                "r*_radial": float(r_star(e, Cmin, 2)),
                "r*_iso": float(r_star(e, Cmin, 3)),
            }
    return out


def make_props(key, a, b):
    """只替换密度式 ρ=a+bC，其余物性（c_p、k、D）一律沿用附录 3。

    这一步是扫描的关键：b 同时影响两处——热方程里的体积热容 ρc_p，
    以及（经由 r=b/a）收缩律。两者都要真的随 b 变，否则"固定几何"
    那一列会退化成同一个算例。
    """
    def f(conc, temp):
        _rho, _drho, cp, k_cond, diff = base.props_p23(conc, temp)
        c = np.maximum(conc, 1e-12)
        return a + b * c, np.full_like(c, b), cp, k_cond, diff
    base.PROPERTY_SETS[key] = f
    return key


def job(kind, a, b, prop, n, dt):
    """单个算例（模块级，供 multiprocessing 取用）。"""
    if kind == "fixed":
        return base.solve(prop=prop, shrink=False, n=n, dt=dt)["dry_time"]
    if kind == "measured":
        return base.solve(prop=prop, shrink=True, n=n, dt=dt)["dry_time"]
    sol, _ = closure_solve(a, b, prop=prop, n=n, dt=dt)
    return sol["dry_time"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analytic", action="store_true")
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    if not (args.analytic or args.matrix or args.sweep):
        args.analytic = True

    # 已在磁盘上的结果先读回来，避免只跑一部分时把其它结果覆盖掉
    jf = OUT / "收缩判据与代价.json"
    rep = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else {}
    rep["常数"] = {"C0": C0, "R0_cm": R0 * 100, "a": A3, "b": B3,
                   "n_grid": args.n}
    if args.analytic or args.matrix or args.sweep:
        rep["判据"] = parse_criterion()
        rep["题面两式的r"] = {"附录3": B3 / A3, "附录4": 90.0 / 760.0}
        rep["临界含水率C*"] = {
            "附录3": {("eps=%.0f%%" % (100 * e)):
                      {"iso": float(C_star(e, B3 / A3, 3)),
                       "radial": float(C_star(e, B3 / A3, 2))}
                      for e in (0.03, 0.05, 0.10)},
            "附录4": {("eps=%.0f%%" % (100 * e)):
                      {"iso": float(C_star(e, 90.0 / 760.0, 3)),
                       "radial": float(C_star(e, 90.0 / 760.0, 2))}
                      for e in (0.03, 0.05, 0.10)},
        }
        rep["收缩系数"] = {"beta_附录3": float(beta_of(B3 / A3, A3)),
                          "beta_附录4": float(beta_of(90.0 / 760.0, 760.0))}

    from multiprocessing import get_context
    ctx = get_context("fork")

    if args.matrix:
        tasks = [("fixed", A3, B3, "p23"), ("measured", A3, B3, "p23"),
                 ("implied", A3, B3, "p23")]
        with ctx.Pool(args.jobs) as pool:
            res = pool.starmap(job, [(k, a, b, p, args.n, 5.0)
                                     for k, a, b, p in tasks])
        rep["几何对照"] = {k: (None if v is None else float(v))
                          for (k, _, _, _), v in zip(tasks, res)}
        base_t = rep["几何对照"]["fixed"]
        rep["几何对照相对固定半径"] = {
            k: (None if v is None or base_t is None
                else float(100 * (v / base_t - 1)))
            for k, v in rep["几何对照"].items()}
        print("几何对照：", {k: (round(v / 3600, 2) if v else None)
                            for k, v in rep["几何对照"].items()})

    if args.sweep:
        # 固定 a=650，只改 b（等价于改 r）；ρ 随 b 变，c_p/k/D 仍用附录 3
        # 阈值 r* 落在 r 较大的一侧，故在 b=320~650 之间加密
        b_list = [0.0, 64.0, 96.0, 128.0, 192.0, 320.0,
                  420.0, 500.0, 560.0, 610.0, 650.0]
        keys = {b: make_props("sweep_%.0f" % b, A3, b) for b in b_list}
        with ctx.Pool(args.jobs) as pool:
            tf = pool.starmap(job, [("fixed", A3, b, keys[b], args.n, 5.0)
                                    for b in b_list])
            ti = pool.starmap(job, [("implied", A3, b, keys[b], args.n, 5.0)
                                    for b in b_list])
        rows = []
        for b, a_, c_ in zip(b_list, tf, ti):
            rows.append({
                "b": b, "r": b / A3,
                "t_fixed_h": None if a_ is None else a_ / 3600.0,
                "t_implied_h": None if c_ is None else c_ / 3600.0,
                "delta_pct": (None if not a_ or not c_ else 100 * (a_ / c_ - 1)),
            })
        rep["b扫描"] = rows
        print("b 扫描完成")

    (OUT / "收缩判据与代价.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已写出 -> 输出/收缩判据与代价.json")


if __name__ == "__main__":
    main()
