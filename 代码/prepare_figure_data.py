#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为论文插图准备全部派生数据，缓存到 输出/figdata/。

为什么单独一步
插图脚本要读的东西有一部分需要重新求解（问题一的全细网格解、问题四的
"不收缩对照解"、参数灵敏度扫描）。这些算一次要几分钟，不适合放在出图
循环里反复跑；更重要的是，把它们固化成可复现的文件后，任何一张图都能
脱离求解器单独重绘。

产物（均在 输出/figdata/ 下）
    air.csv             附件 1 的 (t, T_inf, C_inf)
    radius.csv          附件 2 的 (t, R)
    p1_full.npz         问题一细网格解 + 守恒检验序列 + 指定时刻剖面
    p4_noshrink.npz     问题四物性但半径固定 2 cm 的对照解
    sens.csv            问题三 h / h_m / D ±20% 灵敏度
    rho_analysis.json   物性关联式自洽性检验的全部标量结果
    props_curves.npz    附录 3 / 附录 4 的 rho, cp, k, D 随 C（与 T）的曲线

运行：
    PY=/Users/sqz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
    $PY 代码/prepare_figure_data.py
    $PY 代码/prepare_figure_data.py --only air,radius,rho
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import openpyxl

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT = PROJECT / "输出"
FIGDATA = OUT / "figdata"
sys.path.insert(0, str(HERE))

RHO_W = 1000.0      # 纯水真密度 kg/m^3
RHO_S = 1400.0      # 干物质真密度 kg/m^3（PDF 取值，见 §附注）
C0 = 2.55           # 初始干基含水率


def _read_attachment(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    rows = [r for r in wb.active.iter_rows(min_row=2, values_only=True)
            if r[0] is not None]
    return np.array(rows, dtype=float)


# ============================================================ 附件数据
def build_air():
    d = _read_attachment(PROJECT / "附件" / "附件1-烘房温度与水分浓度.xlsx")
    np.savetxt(FIGDATA / "air.csv", d[:, :3], delimiter=",",
               header="t_s,T_inf_C,C_inf_kgkg", comments="", fmt="%.6f")
    print("air.csv      %d 行" % len(d))
    return d


def build_radius():
    d = _read_attachment(PROJECT / "附件" / "附件2-药材半径.xlsx")
    np.savetxt(FIGDATA / "radius.csv", d[:, :2], delimiter=",",
               header="t_s,R_cm", comments="", fmt="%.6f")
    # 再导出一份"插值＋解析导数"的密网格版本。用求解器同一份 PCHIP 实现，
    # 保证图上画出来的 R(t)、dR/dt 与问题四方程里用的完全同源。
    import solve_p234

    pch = solve_p234.Pchip(d[:, 0], d[:, 1] / 100.0)
    tt = np.linspace(d[0, 0], d[-1, 0], 2000)
    rr = pch(tt)
    dr = pch.deriv(tt)
    np.savetxt(FIGDATA / "radius_pchip.csv",
               np.column_stack([tt, rr * 100.0, dr * 100.0]), delimiter=",",
               header="t_s,R_cm,dRdt_cm_per_s", comments="", fmt="%.8f")
    print("radius.csv   %d 行, R: %.4f -> %.4f cm"
          % (len(d), d[0, 1], d[-1, 1]))
    return d


# ============================================================ 问题一
def build_p1():
    """重解问题一（N=400, dt=1 s）并把守恒检验序列一并缓存。

    结果与 输出/result1_data.npz 逐位一致（已验证），此处额外保存
    Q(t)=∫C dV、表面累计通量与细网格末态剖面，供守恒检验图使用。
    """
    import solve_p1

    res = solve_p1.solve()
    rep = solve_p1.conservation_report(res)
    snap_idx = [int(round(t / res["dt"])) for t in solve_p1.REPORT_TIMES]
    np.savez_compressed(
        FIGDATA / "p1_full.npz",
        times=res["times"],
        r_out_cm=res["r_out_cm"],
        temp_out=res["temp_out"],
        conc_out=res["conc_out"],
        q_volume=res["c_volume"],
        phys_volume=res["phys_volume"],
        surf_flux=res["surf_flux"],
        snap_idx=np.array(snap_idx),
        r_fine_cm=np.arange(len(res["final_temp"])) * res["hx"] * 100.0,
        temp_fine=res["final_temp"],
        conc_fine=res["final_conc"],
        hx=np.array([res["hx"]]),
        dt=np.array([res["dt"]]),
        residual=np.array([rep["relative_residual"]]),
        # 两个特征量随解一起存下来，绘图脚本就不必再 import 求解器
        # （求解器依赖 openpyxl，而绘图虚拟环境里没有）。
        alpha=np.array([solve_p1.ALPHA]),
        d_c0=np.array([float(solve_p1.diffusion_coefficient(solve_p1.C_INIT))]),
        bi=np.array([solve_p1.H_HEAT * solve_p1.RADIUS / solve_p1.K_COND]),
        bi_m=np.array([solve_p1.H_MASS * solve_p1.RADIUS
                       / float(solve_p1.diffusion_coefficient(solve_p1.C_INIT))]),
    )
    print("p1_full.npz  守恒残差 %.3e" % rep["relative_residual"])


# ============================================================ 问题四对照
def build_p4_noshrink():
    """问题四物性 + 固定半径 2 cm 的对照解（论文中 5.41 天的来源）。

    时间步取 dt=5 s，与论文正文其余计算口径完全一致（正文给出的
    467260 s 就是这一组设定下的结果），避免图文出现两套数字。
    """
    import solve_p234

    dt = 5.0
    tend = 900000.0
    sol = solve_p234.solve(prop="p4", shrink=False, n=200, dt=dt, tend=tend)
    pos = np.array([0.0, 0.5, 1.0, 1.5, 2.0]) / 100.0
    conc = np.empty((len(sol["times"]), len(pos)))
    temp = np.empty_like(conc)
    for k in range(len(sol["times"])):
        conc[k] = np.interp(pos / solve_p234.R0, sol["xi"], sol["conc"][k])
        temp[k] = np.interp(pos / solve_p234.R0, sol["xi"], sol["temp"][k])
    dry = sol["dry_time"]
    np.savez_compressed(
        FIGDATA / "p4_noshrink.npz",
        times=sol["times"], conc=conc, temp=temp, xi=sol["xi"],
        conc_all=sol["conc"],
        dry_time=np.array([np.nan if dry is None else dry]),
        dt=np.array([dt]),
    )
    print("p4_noshrink.npz  烘干时间 %s s"
          % ("未达标" if dry is None else "%.0f" % dry))


# ============================================================ 灵敏度
def build_sens():
    import sens

    cases = [("h", -0.2), ("h", +0.2), ("hm", -0.2), ("hm", +0.2),
             ("D", -0.2), ("D", +0.2)]
    rows = [["param", "scale", "t_center_s", "t_center_h"]]
    base = sens.run()
    rows.append(["base", "1.00", "%.1f" % base, "%.4f" % (base / 3600.0)])
    for name, dv in cases:
        kw = {"h": "hscale", "hm": "hmscale", "D": "dscale"}[name]
        t = sens.run(**{kw: 1.0 + dv})
        rows.append([name, "%.2f" % (1.0 + dv), "%.1f" % t,
                     "%.4f" % (t / 3600.0)])
        print("  sens %-3s %+5.0f%% : %9.1f s = %7.4f h (%+.2f%%)"
              % (name, dv * 100, t, t / 3600.0, 100 * (t - base) / base))
    with open(FIGDATA / "sens.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print("sens.csv  基准 %.1f s" % base)


# ============================================================ 物性曲线
def build_props():
    import solve_p234

    c = np.linspace(0.0, 2.55, 400)
    t_k = np.full_like(c, 323.15)
    rho3, _, cp3, k3, d3 = solve_p234.props_p23(c, t_k - 273.15)
    rho4, _, cp4, k4, d4 = solve_p234.props_p4(c, t_k - 273.15)
    temps = np.linspace(28.0, 50.0, 120)
    cs = np.array([2.55, 1.50, 0.60, 0.15])
    dd3 = np.array([2.4e-3 * np.exp(-0.45 / cc)
                    * np.exp(-3850.0 / (temps + 273.15)) for cc in cs])
    dd4 = np.array([4.2e-4 * np.exp(-0.30 / cc)
                    * np.exp(-3850.0 / (temps + 273.15)) for cc in cs])
    np.savez_compressed(
        FIGDATA / "props_curves.npz",
        c=c, rho3=rho3, cp3=cp3, k3=k3, d3=d3,
        rho4=rho4, cp4=cp4, k4=k4, d4=d4,
        temps=temps, cs=cs, dd3=dd3, dd4=dd4,
    )
    print("props_curves.npz  rho3: %.1f->%.1f, rho4: %.1f->%.1f"
          % (rho3[0], rho3[-1], rho4[0], rho4[-1]))


# ============================================================ rho 辨析
def build_rho():
    """复现《干基含水率-体密度经验关系式的自洽性检验》一文的全部数值。

    三类结果：
      (1) 收缩系数 beta 的端点反演与最小二乘反演；
      (2) 三相体积分数（水/固/气）的非负性检验；
      (3) 由经验式推算的几何收缩（体积比、线收缩比、半径）。
    全部由 a+bC、C0、rho_w、rho_s 直接算出，可完全复现。
    """
    out = {"C0": C0, "rho_w": RHO_W, "rho_s": RHO_S, "cases": {}}
    cg = np.linspace(0.0, C0, 2001)

    for name, (a, b, cp_form, k_form) in {
        "appendix3": (650.0, 128.0, (1450.0, 2736.0), (0.21, 0.38)),
        "appendix4": (760.0, 90.0, (1850.0, 2150.0), (0.12, 0.20)),
    }.items():
        rho_end = a + b * C0
        beta_ep = (RHO_W / C0) * ((1 + C0) / rho_end - 1.0 / a)
        v_ratio = 1.0 / (1.0 + beta_ep * a * C0 / RHO_W)
        l_ratio = v_ratio ** (1.0 / 3.0)
        best, best_res = None, np.inf
        for beta in np.linspace(0.5, 1.4, 90001):
            model = a * (1 + cg) / (1 + beta * a * cg / RHO_W)
            r_ = float(np.sqrt(np.mean((model - (a + b * cg)) ** 2)))
            if r_ < best_res:
                best, best_res = beta, r_
        model = a * (1 + cg) / (1 + best * a * cg / RHO_W)
        lin = a + b * cg
        rel_dev = float(np.max(np.abs(model - lin) / lin))
        rho_lo = a * (1 + C0)
        rho_hi = a * (1 + C0) / (1 + a * C0 / RHO_W)
        b_lo = (rho_lo - a) / C0
        b_hi = (rho_hi - a) / C0
        phases = {}
        for tag, cc in (("C0", C0), ("dry", 0.0)):
            rr = a + b * cc
            w = cc / (1 + cc)
            e_w = rr * w / RHO_W
            e_s = rr * (1 - w) / RHO_S
            phases[tag] = {"rho": rr, "eps_w": e_w, "eps_s": e_s,
                           "eps_a": 1 - e_w - e_s}
        # 几何收缩：单位干物质体积 V(C) = (1+C)/rho(C)，以初始态 V(C0) 为基准。
        # 注意这里**不再**出现 beta——它是经验式本身的直接推论，正是"rho 的
        # 线性拟合式已内含收缩"这句话的定量表达。
        v_at_C0 = (1 + C0) / (a + b * C0)
        geo = {}
        for cc in (2.00, 1.00, 0.50, 0.15):
            ratio = ((1 + cc) / (a + b * cc)) / v_at_C0
            geo["%.2f" % cc] = {"V_ratio": ratio,
                                "L_ratio": ratio ** (1.0 / 3.0),
                                "R_cm": 2.0 * ratio ** (1.0 / 3.0)}
        out["cases"][name] = {
            "a": a, "b": b,
            "rho_end": rho_end,
            "V_ratio": v_ratio, "shrink_vol_pct": 100 * (1 - v_ratio),
            "L_ratio": l_ratio,
            "beta_endpoint": beta_ep, "beta_lsq": best,
            "rmse": best_res, "max_rel_dev": rel_dev,
            "b_allowed": [b_lo, b_hi],
            "phases": phases,
            "geo": geo,
            "cp_w": cp_form[0] + cp_form[1],
            "k_w": k_form[0] + k_form[1],
            "rho_d_C0": rho_end / (1 + C0),
            "rho_d_0": a,
        }
    out["naive_rho_C0"] = 650.0 * (1 + C0)
    (FIGDATA / "rho_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    c3 = out["cases"]["appendix3"]
    print("rho_analysis.json  beta: 附录3 %.3f/%.3f, 附录4 %.3f/%.3f"
          % (c3["beta_endpoint"], c3["beta_lsq"],
             out["cases"]["appendix4"]["beta_endpoint"],
             out["cases"]["appendix4"]["beta_lsq"]))


# ============================================================ 判据阈值扫描
def build_threshold():
    """把中心含水率一直算到 0.10，用于"判据阈值 → 烘干时长"的敏感性图。

    主结果只算到中心低于 0.15 就停（题面判据），因此拿不到 0.15 以下的曲线。
    这里把判据临时放宽到 0.10 多算一段。因为中心含水率沿时间单调下降，
    "阈值为 C* 的烘干时长"就等于中心曲线首次穿过 C* 的时刻，一次求解即可
    给出任意阈值的结果，无需对每个阈值重解。
    """
    import solve_p234

    original = solve_p234.C_DRY
    solve_p234.C_DRY = 0.10
    try:
        sol = solve_p234.solve(prop="p23", shrink=False, n=100, dt=5.0,
                               tend=330000.0)
    finally:
        solve_p234.C_DRY = original
    np.savetxt(FIGDATA / "threshold.csv",
               np.column_stack([sol["times"], sol["conc"][:, 0]]), delimiter=",",
               header="t_s,C_center", comments="", fmt="%.6f")
    print("threshold.csv  算到 %.0f s，中心含水率末值 %.4f"
          % (sol["times"][-1], sol["conc"][-1, 0]))


BUILDERS = {
    "air": build_air,
    "radius": build_radius,
    "p1": build_p1,
    "p4ns": build_p4_noshrink,
    "sens": build_sens,
    "props": build_props,
    "rho": build_rho,
    "thr": build_threshold,
}


def main():
    ap = argparse.ArgumentParser(description="论文插图派生数据准备")
    ap.add_argument("--only", default=None,
                    help="逗号分隔的子集: " + ",".join(BUILDERS))
    args = ap.parse_args()
    FIGDATA.mkdir(parents=True, exist_ok=True)
    keys = list(BUILDERS) if not args.only else [
        k.strip() for k in args.only.split(",") if k.strip()]
    for k in keys:
        if k not in BUILDERS:
            raise SystemExit("未知步骤: %s" % k)
        print("== %s" % k)
        BUILDERS[k]()
    print("\n完成，输出目录: %s" % FIGDATA)


if __name__ == "__main__":
    main()
