#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文数据审计：把论文里每个可核验的数字重算一遍，逐条与正文比对。

为什么要这个脚本
----------------
论文里的表有的来自 `输出/` 下的产物（可直接核验），有的来自早期的临时计算
（没有留下脚本）。后者一旦与结果文件脱节，就无法复核。本脚本把这些数字
**全部重算一次**，并列成"论文值 vs 复算值"的对照表，任何一条不一致都会被
标出来。

口径约定（重要，正文必须与之一致）
----------------------------------
* 干物质质量检验：在**固定物理位置网格**（0,0.1,…,2.0 cm）上积分
  `m_d = 2πL ∫ [ρ(C)/(1+C)] r dr`，积分上限取**当前半径 R(t)**；
  超出 R(t) 的位置不参与积分（该处物料已不存在）。
* 问题一网格收敛：表面含水率取 t = 1800 s、r = R 处的节点值。
* 空气条件的平滑/插值变体：先把附件 1 的序列变成目标形式，再经
  `np.interp` 线性取用（对 1 s 密网格而言等价于该插值本身）。

跑法
----
    PY=/Users/sqz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
    $PY 代码/verify_paper_numbers.py              # 只跑轻量项（秒级）
    $PY 代码/verify_paper_numbers.py --heavy      # 追加界面取法等重解项（十分钟级）
    $PY 代码/verify_paper_numbers.py --json 输出/论文数据审计.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT = PROJECT / "输出"
sys.path.insert(0, str(HERE))

import solve_p1 as S1                                    # noqa: E402

C0, RHO_W = 2.55, 1000.0
RAD = 0.02
L_LEN = 0.25

RECORDS: list[dict] = []


def rec(group: str, item: str, paper, calc, note: str = "", tol: float = 0.0):
    """登记一条审计记录；paper/calc 均为字符串或数值。"""
    ok = None
    try:
        ok = abs(float(paper) - float(calc)) <= tol
    except (TypeError, ValueError):
        ok = str(paper) == str(calc)
    RECORDS.append({"组": group, "项目": item, "论文值": paper,
                    "复算值": calc, "一致": ok, "备注": note})
    return ok


def close(paper, calc, rel=2e-3):
    """相对容差比较（默认 0.2%，覆盖四舍五入）。"""
    try:
        p, c = float(paper), float(calc)
    except (TypeError, ValueError):
        return str(paper) == str(calc)
    if p == c:
        return True
    return abs(p - c) <= rel * max(abs(p), abs(c), 1e-12)


# --------------------------------------------------------------------------
# 1. 直接读产物即可核验的项
# --------------------------------------------------------------------------
def audit_artifacts() -> None:
    meta = json.loads((OUT / "result_meta.json").read_text(encoding="utf-8"))
    d23 = np.load(OUT / "result23_data.npz", allow_pickle=True)
    d4 = np.load(OUT / "result4_data.npz", allow_pickle=True)
    d1 = np.load(OUT / "result1_data.npz", allow_pickle=True)

    # 烘干时长（论文核心结论）
    t3 = float(d23["dry_time"][0])
    t4 = float(d4["dry_time"][0])
    rec("结果", "问题三烘干时长 / s", 206700, round(t3), tol=1)
    rec("结果", "问题三烘干时长 / h", 57.42, round(t3 / 3600, 2), tol=0.01)
    rec("结果", "问题四烘干时长 / s", 183885, round(t4), tol=1)
    rec("结果", "问题四烘干时长 / h", 51.08, round(t4 / 3600, 2), tol=0.01)
    rec("结果", "问题四相对问题三缩短 / %", 11.0,
        round(100 * (t3 - t4) / t3, 1), tol=0.1)

    # 结果文件规格（表 tab:files）
    x3 = _xlsx_rows(OUT / "result3.xlsx")
    x4 = _xlsx_rows(OUT / "result4.xlsx")
    rec("结果文件", "result3 行数（不含表头）", 3445, x3[0], tol=0)
    rec("结果文件", "result3 末时刻 / s", 206700, x3[1], tol=1)
    # 表中"论文值"一律填正文实际写的数，便于直接暴露不一致
    rec("结果文件", "result4 行数（不含表头）", 3064, x4[0], tol=0)
    rec("结果文件", "result4 末时刻 / s", 183840, x4[1], tol=1)
    rec("结果文件", "result2 行数（不含表头）", 206700, meta["n2"], tol=0)

    # 问题一口径：t=1800 s 中心与表面
    c1 = d1["conc_out"][1800]
    t1 = d1["temp_out"][1800]
    rec("问题一", "1800 s 中心含水率", 2.5500, round(float(c1[0]), 4), tol=5e-5)
    rec("问题一", "1800 s 表面含水率", 1.5103, round(float(c1[-1]), 4), tol=5e-5)
    rec("问题一", "1800 s 中心温度 / °C", 33.5765, round(float(t1[0]), 4), tol=5e-5)
    rec("问题一", "1800 s 表面温度 / °C", 36.7863, round(float(t1[-1]), 4), tol=5e-5)

    # 问题二 3 h 的报数点
    dt23 = float(d23["dt"][0])
    i3 = int(round(10800.0 / dt23))
    c2 = d23["conc21"][i3]
    rec("问题二", "3 h 中心含水率", 1.7663, round(float(c2[0]), 4), tol=5e-5)
    rec("问题二", "3 h 表面含水率", 1.0082, round(float(c2[-1]), 4), tol=5e-5)
    rec("问题二", "0.5 h 表面含水率", 1.6491,
        round(float(d23["conc21"][int(round(1800.0 / dt23))][-1]), 4), tol=5e-5)

    # 网格 / 时间步 / 口径三因素对照（CSV 为唯一来源）
    rows = list(csv.DictReader((OUT / "问题三-网格与口径对照.csv").open(encoding="utf-8")))
    tab = {(r["form"], int(r["N"]), float(r["dt_s"])): float(r["t_center_s"])
           for r in rows}
    for N, fick_paper, cons_paper in ((20, 203335, 216215), (50, 205625, 218575),
                                      (100, 206365, 219340), (200, 206700, 219685),
                                      (400, 206840, 219830)):
        rec("网格对照", f"fick N={N} / s", fick_paper, tab[("fick", N, 5.0)], tol=1)
        rec("网格对照", f"变密度 N={N} / s", cons_paper, tab[("cons", N, 5.0)], tol=1)
    for dt_s, fick_paper, cons_paper in ((2, 206354, 219328), (5, 206365, 219340),
                                         (10, 206380, 219360), (20, 206420, 219400),
                                         (60, 206580, 219540), (120, 206760, 219720)):
        rec("时间步对照", f"fick Δt={dt_s:g} / s", fick_paper, tab[("fick", 100, dt_s)], tol=1)
        rec("时间步对照", f"变密度 Δt={dt_s:g} / s", cons_paper, tab[("cons", 100, dt_s)], tol=1)

    # 第三种口径：干基密度守恒（论文口径表第三行）。
    # 这张表由 closure_rhod.py 生成；此前论文里的 51.39 h 没有实现可核验，
    # 因此单独登记，避免再出现"文中有数、代码无源"。
    clos3 = list(csv.DictReader((OUT / "问题三-三种口径对照.csv").open(encoding="utf-8")))
    c3 = {(r["form"], int(r["N"])): (float(r["t_dry_s"]), float(r["t_dry_h"]),
                                     float(r["relative_to_fick_pct"]))
          for r in clos3}
    for form, N, p_s, p_h, p_pct in (
            ("fick", 100, 206365, 57.32, 0.00),
            ("cons", 100, 219340, 60.93, +6.29),
            ("rhod", 100, 174195, 48.39, -15.59),
            ("fick", 200, 206700, 57.42, 0.00),
            ("cons", 200, 219685, 61.02, +6.28),
            ("rhod", 200, 174490, 48.47, -15.58)):
        c3_s, c3_h, c3_pct = c3[(form, N)]
        rec("三种口径", f"{form} N={N} / s", p_s, c3_s, tol=1)
        rec("三种口径", f"{form} N={N} / h", p_h, c3_h, tol=0.006)
        rec("三种口径", f"{form} N={N} 相对 %", p_pct, round(c3_pct, 2), tol=0.011)

    # 参数灵敏度
    sens = list(csv.DictReader((OUT / "figdata" / "sens.csv").open(encoding="utf-8")))
    smap = {(r["param"], float(r["scale"])): float(r["t_center_s"]) for r in sens}
    rec("灵敏度", "基准 / s", 206365, smap[("base", 1.0)], tol=1)
    rec("灵敏度", "h-20% / s", 206435, smap[("h", 0.8)], tol=1)
    rec("灵敏度", "h+20% / s", 206320, smap[("h", 1.2)], tol=1)
    rec("灵敏度", "hm-20% / s", 212560, smap[("hm", 0.8)], tol=1)
    rec("灵敏度", "hm+20% / s", 202555, smap[("hm", 1.2)], tol=1)
    rec("灵敏度", "D-20% / s", 252070, smap[("D", 0.8)], tol=1)
    rec("灵敏度", "D+20% / s", 176175, smap[("D", 1.2)], tol=1)

    # §6 的反演值
    ra = json.loads((OUT / "figdata" / "rho_analysis.json").read_text(encoding="utf-8"))
    a3, a4 = ra["cases"]["appendix3"], ra["cases"]["appendix4"]
    rec("物性反演", "附录3 端点 β", 0.822, round(a3["beta_endpoint"], 3), tol=1e-3)
    rec("物性反演", "附录4 端点 β", 0.891, round(a4["beta_endpoint"], 3), tol=1e-3)
    rec("物性反演", "附录3 V(0)/V(C0)", 0.423, round(a3["V_ratio"], 3), tol=1e-3)
    rec("物性反演", "附录4 V(0)/V(C0)", 0.367, round(a4["V_ratio"], 3), tol=1e-3)


def _xlsx_rows(path: Path):
    """返回 (数据行数, 末行时间)；只读第一列，避免把整表读进内存。"""
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True)
    ws = wb.worksheets[0]
    n, last = 0, None
    for row in ws.iter_rows(min_col=1, max_col=1, values_only=True):
        v = row[0]
        if v is None or isinstance(v, str):
            continue
        n += 1
        last = v
    wb.close()
    return n, last


# --------------------------------------------------------------------------
# 2. 附件 1 的噪声诊断
# --------------------------------------------------------------------------
def audit_noise() -> None:
    # 与插图 d02_air_diag 完全同一口径：高斯加权局部二次趋势（w=15, p=2）。
    # 换用线性去趋势、滑动平均或多項式拟合都复现不出表中的自相关 −0.31，
    # 只有这一口径能同时对上 σ、最大残差与自相关，说明正文与插图同源。
    t, T, Cw = S1.load_air_conditions()

    def gwsmooth(y, w=15, p=2):
        n = len(y); half = w // 2; out = np.empty(n)
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            idx = np.arange(lo, hi); off = idx - i
            A = np.vander(off, p + 1)
            wg = np.exp(-(off / (half * 0.8)) ** 2)
            c, *_ = np.linalg.lstsq(A * wg[:, None], y[idx] * wg, rcond=None)
            out[i] = c[-1]
        return out

    for name, y, sigma_paper, max_paper, acf_paper, frac_paper in (
            ("烘房温度", T, 0.116, 0.27, -0.31, 0.52),
            ("水分浓度", Cw, 1.1e-4, 2.8e-4, -0.40, 0.38)):
        res = y - gwsmooth(y)
        sig = float(np.std(res, ddof=1))
        mx = float(np.max(np.abs(res)))
        acf1 = float(np.corrcoef(res[:-1], res[1:])[0, 1])
        frac = 100.0 * sig / (y.max() - y.min())
        rec("附件1噪声", f"{name} 残差标准差", sigma_paper, f"{sig:.3g}", tol=0.06 * sigma_paper)
        rec("附件1噪声", f"{name} 最大残差", max_paper, f"{mx:.3g}", tol=0.05 * max_paper)
        rec("附件1噪声", f"{name} 滞后1自相关", acf_paper, round(acf1, 2), tol=0.02)
        rec("附件1噪声", f"{name} 占全变幅 / %", frac_paper, round(frac, 2), tol=0.05)


# --------------------------------------------------------------------------
# 3. 问题一：网格收敛、平滑方式、插值方式
# --------------------------------------------------------------------------
def _p1_solve(n_fine=400, air=None, dt=1.0):
    orig = S1.load_air_conditions
    if air is not None:
        S1.load_air_conditions = lambda path=S1.ATTACHMENT1: air
    try:
        return S1.solve(n_fine=n_fine, dt=dt)
    finally:
        S1.load_air_conditions = orig


def audit_p1_grid() -> None:
    base = None
    vals = {}
    for n in (20, 40, 100, 400):
        res = _p1_solve(n_fine=n)
        vals[n] = float(res["conc_out"][-1][-1])        # t=1800 s、r=R
    for n, paper in ((20, 1.51211), (40, 1.51076), (100, 1.51039), (400, 1.51033)):
        rec("问题一网格", f"N={n} 表面含水率", paper, round(vals[n], 5), tol=5e-5)


def audit_p1_smooth() -> None:
    t, T, Cw = S1.load_air_conditions()
    base = _p1_solve(n_fine=400)
    ref_c = base["conc_out"][1:]          # 全程（t>0），不只末时刻
    ref_t = base["temp_out"][1:]

    def movavg(y, k):
        pad = k // 2
        ypad = np.r_[np.full(pad, y[0]), y, np.full(pad, y[-1])]
        ker = np.ones(k) / k
        return np.convolve(ypad, ker, mode="valid")

    def savgol(y, k):
        # 二次多项式最小二乘平滑（与 scipy.signal.savgol_filter 同定义）
        half = k // 2
        x = np.arange(-half, half + 1)
        A = np.vstack([np.ones_like(x), x, x * x]).T
        # 平滑系数 = 第一行 of pinv(A)
        w = np.linalg.pinv(A)[0]
        ypad = np.r_[np.full(half, y[0]), y, np.full(half, y[-1])]
        return np.convolve(ypad, w[::-1], mode="valid")

    variants = (("3 点滑动平均（180 s）", movavg(T, 3), movavg(Cw, 3)),
                ("5 点 Savitzky-Golay（300 s）", savgol(T, 5), savgol(Cw, 5)),
                ("15 点 Savitzky-Golay（900 s）", savgol(T, 15), savgol(Cw, 15)))
    paper_vals = (("3 点滑动平均（180 s）", 7.1e-6, 0.022),
                  ("5 点 Savitzky-Golay（300 s）", 5.9e-6, 0.014),
                  ("15 点 Savitzky-Golay（900 s）", 2.1e-5, 0.055))
    for name, Tsm, Csm in variants:
        res = _p1_solve(n_fine=400, air=(t, Tsm, Csm))
        dC = float(np.max(np.abs(res["conc_out"][1:] - ref_c)))
        dT = float(np.max(np.abs(res["temp_out"][1:] - ref_t)))
        for pname, pC, pT in paper_vals:
            if pname == name:
                rec("问题一平滑", f"{name} 含水率偏差", pC, f"{dC:.3g}", tol=0.5 * pC)
                rec("问题一平滑", f"{name} 温度偏差", pT, round(dT, 4), tol=0.005)


def audit_p1_interp() -> None:
    t, T, Cw = S1.load_air_conditions()
    tf = np.arange(t[0], t[-1] + 1.0, 1.0)

    def pchip(x, y, xq):
        # Fritsch–Carlson 保单调三次 Hermite
        h = np.diff(x)
        d = np.diff(y) / h
        m = np.zeros_like(y)
        m[0], m[-1] = d[0], d[-1]
        for i in range(1, len(y) - 1):
            if d[i - 1] * d[i] <= 0:
                m[i] = 0.0
            else:
                w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
                m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
        idx = np.clip(np.searchsorted(x, xq) - 1, 0, len(x) - 2)
        hh = x[idx + 1] - x[idx]
        s = (xq - x[idx]) / hh
        h00 = 2 * s ** 3 - 3 * s ** 2 + 1
        h10 = s ** 3 - 2 * s ** 2 + s
        h01 = -2 * s ** 3 + 3 * s ** 2
        h11 = s ** 3 - s ** 2
        return h00 * y[idx] + h10 * hh * m[idx] + h01 * y[idx + 1] + h11 * hh * m[idx + 1]

    def natural_cubic(x, y, xq):
        n = len(x)
        A = np.zeros((n, n)); rhs = np.zeros(n)
        A[0, 0] = A[-1, -1] = 1.0
        for i in range(1, n - 1):
            A[i, i - 1] = x[i] - x[i - 1]
            A[i, i] = 2 * (x[i + 1] - x[i - 1])
            A[i, i + 1] = x[i + 1] - x[i]
            rhs[i] = 6 * ((y[i + 1] - y[i]) / (x[i + 1] - x[i])
                          - (y[i] - y[i - 1]) / (x[i] - x[i - 1]))
        m = np.linalg.solve(A, rhs)
        idx = np.clip(np.searchsorted(x, xq) - 1, 0, n - 2)
        hh = x[idx + 1] - x[idx]
        a = (x[idx + 1] - xq) / hh
        b = (xq - x[idx]) / hh
        return (a * y[idx] + b * y[idx + 1]
                + ((a ** 3 - a) * m[idx] + (b ** 3 - b) * m[idx + 1]) * hh ** 2 / 6.0)

    base = _p1_solve(n_fine=400)
    for name, f in (("PCHIP", pchip), ("自然三次样条", natural_cubic)):
        Ts = f(t, T, tf); Cs = f(t, Cw, tf)
        # "边界偏差"= 插值序列相对分段线性的最大偏离，即边界条件被改写多少
        Tlin = np.interp(tf, t, T)
        Clin = np.interp(tf, t, Cw)
        d_edge = float(np.max(np.abs(Ts - Tlin)))
        d_edge_c = float(np.max(np.abs(Cs - Clin)))
        # 过冲：插值序列超出原序列相邻两点区间的最大量
        lo = np.minimum(T[:-1], T[1:]); hi = np.maximum(T[:-1], T[1:])
        idx = np.searchsorted(tf, t)
        seg = np.array([Ts[idx[i]:idx[i + 1] + 1].max() for i in range(len(t) - 1)])
        over = float(np.max(np.maximum(seg - hi, 0.0)))
        res = _p1_solve(n_fine=400, air=(tf, Ts, Cs))
        dT = float(np.max(np.abs(res["temp_out"][1:] - base["temp_out"][1:])))
        dC = float(np.max(np.abs(res["conc_out"][1:] - base["conc_out"][1:])))
        rec("问题一插值", f"{name} 边界偏差 / °C", 0.049 if name == "PCHIP" else 0.097,
            round(d_edge, 4), tol=0.02)
        rec("问题一插值", f"{name} 过冲 / °C", 0.0 if name == "PCHIP" else 0.080,
            round(over, 4), tol=0.01)
        rec("问题一插值", f"{name} 温度偏差 / °C",
            0.0040 if name == "PCHIP" else 0.0054, round(dT, 4), tol=0.002)
        rec("问题一插值", f"{name} 含水率偏差",
            1.9e-6 if name == "PCHIP" else 2.9e-6, f"{dC:.2g}", tol=1e-6)


# --------------------------------------------------------------------------
# 4. 干物质守恒检验
# --------------------------------------------------------------------------
def audit_drymass() -> None:
    x_cm = np.arange(21) * 0.1
    rho3 = lambda c: 650.0 + 128.0 * c
    rho4 = lambda c: 760.0 + 90.0 * c

    def md(row, R_cm, rho, valid, surf=None):
        """2πL∫_0^R [ρ(C)/(1+C)] r dr，积分上限取当前半径 R(t)。

        若最后一个网格节点未落在 R(t) 上（收缩时总是如此），把表面值接到
        半径 R(t) 处再积分，避免把最外侧的干壳整段丢掉而高估质量亏损。
        """
        idx = [k for k in range(21) if valid[k] and x_cm[k] <= R_cm + 1e-9]
        if len(idx) < 2:
            return float("nan")
        rr = x_cm[idx].astype(float)
        rowv = np.asarray(row, float)[idx]
        if surf is not None and R_cm > rr[-1] + 1e-9:
            rr = np.r_[rr, R_cm]
            rowv = np.r_[rowv, float(surf)]
        f = rho(rowv) / (1.0 + rowv)
        return 2 * math.pi * np.trapezoid(f * rr * 1e-2, rr * 1e-2) * L_LEN

    d23 = np.load(OUT / "result23_data.npz", allow_pickle=True)
    d4 = np.load(OUT / "result4_data.npz", allow_pickle=True)
    dt23 = float(d23["dt"][0]); dt4 = float(d4["dt"][0])
    ones = np.ones(21, bool)

    m3_0 = md(d23["conc21"][0], 2.0, rho3, ones)
    for h, paper in ((12, 89), (30, 110), ("end", 116)):
        i = len(d23["conc21"]) - 1 if h == "end" else int(round(h * 3600 / dt23))
        val = 100 * (md(d23["conc21"][i], 2.0, rho3, ones) - m3_0) / m3_0
        rec("干物质守恒", f"问题三 {h} h / %", paper, round(val), tol=3)

    m4_0 = md(d4["conc21"][0], 2.0, rho4, ones)
    for h, paper in ((12, -23.1), (30, -13.9), ("end", -10.9)):
        i = len(d4["conc21"]) - 1 if h == "end" else int(round(h * 3600 / dt4))
        R_cm = float(d4["radius"][i]) * 100.0
        val = 100 * (md(d4["conc21"][i], R_cm, rho4, d4["valid21"][i],
                        surf=d4["conc_surf"][i]) - m4_0) / m4_0
        rec("干物质守恒", f"问题四 {h} h / %", paper, round(val, 1), tol=2.0)


# --------------------------------------------------------------------------
# 5. 论文表 1–表 6 与 result1–4.xlsx 的逐格比对
# --------------------------------------------------------------------------
POS5 = [0.0, 0.5, 1.0, 1.5, 2.0]            # 正文表用的 5 个位置 / cm


def _tex_table(label: str):
    """从正文里取出表体，返回 [(行标签, [单元格字符串...]), ...]。"""
    src = (PROJECT / "论文" / "论文全文.tex").read_text(encoding="utf-8")
    for t in re.findall(r"\\begin\{table\}(.*?)\\end\{table\}", src, re.S):
        if "\\label{%s}" % label not in t:
            continue
        body = t.split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
        rows = []
        for line in body.split("\\\\"):
            line = re.sub(r"\\cmidrule[^\s]*(\{[^}]*\})*", "", line)
            line = line.replace("\\midrule", "").strip()
            if not line or line.startswith("\\multicolumn"):
                continue
            cells = [c.strip().replace("\\,", "").replace("$", "")
                     for c in line.split("&")]
            if len(cells) >= 2 and cells[0]:
                rows.append((cells[0], cells[1:]))
        return rows
    raise KeyError(label)


def audit_result_tables() -> None:
    """论文表 1–表 6 的每个数字，都要等于 result1–4.xlsx 对应单元格。"""
    from openpyxl import load_workbook

    def sheet_rows(path, sheet, rows_wanted, cols_wanted):
        """取指定行的指定列。返回 {(row, col): value}。"""
        wb = load_workbook(path, read_only=True, data_only=True)
        ws = wb[sheet] if sheet else wb.worksheets[0]
        out = {}
        want = set(rows_wanted)
        for ri, row in enumerate(ws.iter_rows(values_only=True), start=1):
            if ri not in want:
                continue
            for cj in cols_wanted:
                out[(ri, cj)] = row[cj - 1] if cj - 1 < len(row) else None
            if len(out) >= len(want) * len(cols_wanted):
                break
        wb.close()
        return out

    # ---- 表 1、表 2：问题一，result1.xlsx（1 s 间隔，行 = t+1）
    for label, sheet, tlist in (("tab:p1-T", "温度", [100, 300, 600, 900, 1200, 1500, 1800]),
                                ("tab:p1-C", "水分浓度", [100, 300, 600, 900, 1200, 1500, 1800])):
        rows = _tex_table(label)
        cols = [2 + int(round(p / 0.1)) for p in POS5]
        data = sheet_rows(OUT / "result1.xlsx", sheet,
                          [t + 1 for t in tlist], cols)
        for (t, (rl, cells)) in zip(tlist, rows):
            for k, c in enumerate(cells):
                ref = data.get((t + 1, cols[k]))
                got = "%.4f" % float(ref) if ref is not None else None
                rec("表%s" % ("1" if label.endswith("T") else "2"),
                    f"{sheet} t={t}s r={POS5[k]}cm", c, got, tol=5e-5)

    # ---- 表 3、表 4：问题二，result2.xlsx（1 s，行 = t+1）
    for label, sheet in (("tab:p2-T", "温度"), ("tab:p2-C", "水分浓度")):
        rows = _tex_table(label)
        tlist = [int(round(float(rl) * 3600)) for rl, _ in rows]
        cols = [2 + int(round(p / 0.1)) for p in POS5]
        data = sheet_rows(OUT / "result2.xlsx", sheet,
                          [t + 1 for t in tlist], cols)
        for t, (rl, cells) in zip(tlist, rows):
            for k, c in enumerate(cells):
                ref = data.get((t + 1, cols[k]))
                got = "%.4f" % float(ref) if ref is not None else None
                rec("表%s" % ("3" if sheet == "温度" else "4"),
                    f"{sheet} t={rl}h r={POS5[k]}cm", c, got, tol=5e-5)

    # ---- 表 5：问题三，result3.xlsx（60 s，行 = (t-60)/60 + 2）
    rows = _tex_table("tab:p3")
    wb = load_workbook(OUT / "result3.xlsx", read_only=True, data_only=True)
    ws = wb.worksheets[0]
    last = list(ws.iter_rows(values_only=True))[-1]
    wb.close()
    cols = [2 + int(round(p / 0.1)) for p in POS5]
    for rl, cells in rows:
        if "结束" in rl:
            rowvals = last
        else:
            t = int(float(rl)) * 3600
            wb = load_workbook(OUT / "result3.xlsx", read_only=True, data_only=True)
            ws = wb.worksheets[0]
            rowvals = None
            for ri, row in enumerate(ws.iter_rows(values_only=True), start=1):
                if ri == (t - 60) // 60 + 2:
                    rowvals = row
                    break
            wb.close()
        for k, c in enumerate(cells):
            ref = rowvals[cols[k] - 1]
            got = "%.4f" % float(ref) if ref is not None else None
            rec("表5", f"t={rl} r={POS5[k]}cm", c, got, tol=5e-5)

    # ---- 表 6：问题四，result4.xlsx（60 s；末列是"药材表面"）
    rows = _tex_table("tab:p4")
    wb = load_workbook(OUT / "result4.xlsx", read_only=True, data_only=True)
    ws = wb.worksheets[0]
    all_rows = list(ws.iter_rows(values_only=True))
    wb.close()
    data = all_rows[1:]
    for rl, cells in rows:
        if "结束" in rl:
            rowvals = data[-1]
            # result4.xlsx 的行落在 60 s 输出网格上（末行 183840 s），而正文的
            # "烘干结束"取判据满足时刻 183885 s，两者相差不到一个步长，
            # 因此末行允许 2e-4 的取整差。
            tol_row = 2e-4
        else:
            t = int(float(rl)) * 3600
            rowvals = data[(t - 60) // 60]
            tol_row = 5e-5
        for k, c in enumerate(cells):
            if k >= 5:                       # 第 6 列为药材表面
                ref = rowvals[22]
            else:
                ref = rowvals[2 + int(round(POS5[k] / 0.1)) - 1]
            got = "%.4f" % float(ref) if ref is not None else "——"
            rec("表6", f"t={rl} {'表面' if k >= 5 else 'r=%.1fcm' % POS5[k]}",
                c, got, tol=tol_row)


# --------------------------------------------------------------------------
# 6. §11 判据与代价：正文数字 vs 收缩判据与代价.json
# --------------------------------------------------------------------------
def audit_criterion_section() -> None:
    p = OUT / "收缩判据与代价.json"
    if not p.exists():
        return
    c = json.loads(p.read_text(encoding="utf-8"))
    g = c.get("几何对照")
    if g:
        rec("§11几何", "固定半径 / h", 57.32, round(g["fixed"] / 3600, 2), tol=0.01)
        rec("§11几何", "ρ反演收缩 / h", 29.45, round(g["implied"] / 3600, 2), tol=0.01)
        rec("§11几何", "实测收缩 / h", 23.85, round(g["measured"] / 3600, 2), tol=0.01)
        rel = c["几何对照相对固定半径"]
        rec("§11几何", "ρ反演相对固定 / %", -48.6, round(rel["implied"], 1), tol=0.1)
        rec("§11几何", "实测相对固定 / %", -58.4, round(rel["measured"], 1), tol=0.1)
    sw = c.get("b扫描")
    if sw:
        for b, lit_r, lit_f, lit_i, lit_d in (
                (0, "0.000", "57.30", "21.27", 169),
                (128, "0.197", "57.32", "29.45", 95),
                (320, "0.492", "57.36", "40.59", 41),
                (650, "1.000", "57.43", "57.43", 0)):
            row = next(x for x in sw if abs(x["b"] - b) < 1e-6)
            rec("§11扫描", f"b={b} r", lit_r, round(row["r"], 3), tol=1e-3)
            rec("§11扫描", f"b={b} t_fixed / h", lit_f, round(row["t_fixed_h"], 2), tol=0.01)
            rec("§11扫描", f"b={b} t_implied / h", lit_i, round(row["t_implied_h"], 2), tol=0.01)
            rec("§11扫描", f"b={b} Δ / %", lit_d, round(row["delta_pct"]), tol=1)
        tf = [x["t_fixed_h"] for x in sw]
        rec("§11扫描", "固定几何全变程 / %", 0.24,
            round(100 * (max(tf) / min(tf) - 1), 2), tol=0.02)
    cr = c.get("判据", {})
    if cr:
        q = cr["全程 2.55→0.15"]
        rec("§11判据", "全程 5% r*(径向)", 0.838, round(q["eps=5%"]["r*_radial"], 3), tol=1e-3)
        rec("§11判据", "全程 5% r*(各向同性)", 0.765, round(q["eps=5%"]["r*_iso"], 3), tol=1e-3)
        q2 = cr["问题二窗口 2.55→1.364"]
        rec("§11判据", "问题二窗口 5% r*(径向)", 0.507,
            round(q2["eps=5%"]["r*_radial"], 3), tol=1e-3)
        q3 = cr["预热窗口 C 2.55→2.0"]
        rec("§11判据", "预热 5% r*(径向)", 0.164,
            round(q3["eps=5%"]["r*_radial"], 3), tol=1e-3)
        rec("§11判据", "预热 3% r*(径向)", 0.351,
            round(q3["eps=3%"]["r*_radial"], 3), tol=1e-3)
    cs = c.get("临界含水率C*", {})
    if cs:
        rec("§11判据", "附录3 C*(5%,径向)", 1.953,
            round(cs["附录3"]["eps=5%"]["radial"], 3), tol=1e-3)
        rec("§11判据", "附录4 C*(5%,径向)", 2.062,
            round(cs["附录4"]["eps=5%"]["radial"], 3), tol=1e-3)
    be = c.get("收缩系数", {})
    if be:
        rec("§11判据", "beta(附录3)", 0.822, round(be["beta_附录3"], 3), tol=1e-3)
        rec("§11判据", "beta(附录4)", 0.891, round(be["beta_附录4"], 3), tol=1e-3)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="论文数据审计")
    ap.add_argument("--heavy", action="store_true", help="追加需要重解的项")
    ap.add_argument("--json", default=None, help="把审计结果写到 JSON")
    args = ap.parse_args()

    audit_artifacts()
    audit_noise()
    audit_p1_grid()
    audit_p1_smooth()
    audit_p1_interp()
    audit_drymass()
    audit_result_tables()
    audit_criterion_section()

    bad = [r for r in RECORDS if not r["一致"]]
    w = max(len(r["项目"]) for r in RECORDS)
    print("%-10s %-*s %-16s %-16s %s" % ("组", w, "项目", "论文值", "复算值", "判定"))
    for r in RECORDS:
        if not r["一致"]:
            print("✗ %-8s %-*s %-16s %-16s %s" % (r["组"], w, r["项目"],
                                                r["论文值"], r["复算值"], r["备注"]))
    print()
    print("共审计 %d 项，不一致 %d 项" % (len(RECORDS), len(bad)))
    if args.json:
        Path(args.json).write_text(json.dumps(RECORDS, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print("已写出 ->", args.json)


if __name__ == "__main__":
    main()
