#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文插图生成脚本（依据 scipilot-figure-skill 工作流）.

图型选择依据（references/chart_selection.md）：
  fig1  时间序列看趋势 + 残差        -> 折线（时间 vs 连续）
  fig2  二维矩阵看模式               -> 热力图（感知均匀色图 + colorbar）
  fig3  剖面：连续 vs 连续（参数化） -> 折线族（冗余编码：线型 + marker）
  fig4  时间序列看趋势 + 阈值        -> 折线 + 水平判据线 + 阶段标注
  fig5  双变量同 x                   -> 上下子图共享 x（禁止双 Y 轴）
  fig6  参数扰动看影响               -> 横向条形（按值排序）

出图规范：figsize 直接取论文最终尺寸（A4 正文宽 15.6 cm = 6.14 in），
导出 PDF（矢量）+ PNG（预览），并生成灰度版做色盲检查。

运行: MPLCONFIGDIR=/tmp/mplcache /tmp/figvenv/bin/python 代码/make_figures.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

SKILL_SCRIPTS = os.path.expanduser(
    "~/.codex/skills/scipilot-figure-skill/scripts"
)
sys.path.insert(0, SKILL_SCRIPTS)

from setup_style import setup_style          # noqa: E402
from export_figure import export_figure      # noqa: E402
import visual_qa                             # noqa: E402

import matplotlib                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.patches import (             # noqa: E402
    Arc, Circle, Ellipse, Rectangle, Wedge,
)


PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(PROJECT, "论文", "figs")
os.makedirs(FIGDIR, exist_ok=True)

W = 6.14          # 正文宽度 15.6 cm

# 色盲安全配色（Okabe-Ito 子集）
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def style():
    setup_style(journal="general", lang="zh", serif_for_zh=True)
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": False,
    })


def load_air():
    d = np.loadtxt("/tmp/air.csv", delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1], d[:, 2]


# ------------------------------------------------- 示意图（几何与坐标）
def fig_geom():
    """圆柱几何与一维径向简化示意（示意图不在 scipilot skill 范围内，
    此处直接用 matplotlib 绘制，以与正文排版保持一致）。"""
    fig, axes = plt.subplots(
        1, 2, figsize=(W, 2.95), layout="constrained",
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    # ---------------- (a) 圆柱几何与柱坐标
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0.30, 0.26), 0.40, 0.48,
                           fc="#eaf2fa", ec="#333333", lw=1.0, zorder=1))
    ax.add_patch(Ellipse((0.50, 0.74), 0.40, 0.13,
                         fc="#dbe8f5", ec="#333333", lw=1.0, zorder=2))
    ax.add_patch(Arc((0.50, 0.26), 0.40, 0.13, theta1=180, theta2=360,
                     ec="#333333", lw=1.0, zorder=2))
    # 对称轴
    ax.plot([0.50, 0.50], [0.12, 0.90], "-.", color="#B00020", lw=1.0, zorder=3)
    ax.text(0.515, 0.845, "对称轴 $r=0$", color="#B00020", fontsize=7.5)
    # 半径
    ax.annotate("", xy=(0.70, 0.58), xytext=(0.50, 0.58),
                arrowprops=dict(arrowstyle="<->", color=CB[0], lw=1.0))
    ax.text(0.545, 0.605, "$R=2$ cm", color=CB[0], fontsize=7.5)
    # 长度
    ax.annotate("", xy=(0.22, 0.26), xytext=(0.22, 0.74),
                arrowprops=dict(arrowstyle="<->", color=CB[2], lw=1.0))
    ax.text(0.155, 0.50, "$L=25$ cm", color=CB[2], fontsize=7.5,
            rotation=90, va="center")
    # 坐标轴（r 沿半径向外、z 沿轴向，构成坐标架，避免与边界标注重叠）
    ax.annotate("", xy=(0.90, 0.945), xytext=(0.50, 0.945),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9))
    ax.text(0.905, 0.945, "$r$", fontsize=8, va="center")
    ax.annotate("", xy=(0.44, 0.30), xytext=(0.44, 0.74),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9))
    ax.text(0.415, 0.295, "$z$", fontsize=8, ha="center", va="top")
    # 表面交换
    for yy, txt, col in ((0.66, "对流换热 $h$", CB[1]),
                         (0.44, "对流传质 $h_m$", CB[3])):
        ax.annotate("", xy=(0.715, yy), xytext=(0.93, yy),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
        ax.text(0.94, yy, txt, color=col, fontsize=7.5, va="center")
    ax.text(0.02, 0.06,
            "端面 $2\\pi R^2$ 仅占侧面 $2\\pi RL$\n的 $R/L=8\\%$，轴向输运可忽略",
            fontsize=7.5, va="bottom")
    ax.set_title("(a) 圆柱几何与柱坐标", loc="left")

    # ---------------- (b) 一维径向简化
    ax = axes[1]
    ax.set_xlim(-1.35, 1.45)
    ax.set_ylim(-1.30, 1.30)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.0, fc="#eaf2fa", ec="#333333", lw=1.2,
                        zorder=1))
    for a in range(0, 360, 30):
        th = np.deg2rad(a)
        ax.plot([0, np.cos(th)], [0, np.sin(th)], color="#c9d6e2",
                lw=0.6, zorder=2)
    ax.add_patch(Wedge((0, 0), 1.0, 0, 360, width=0.20,
                       fc="#cfe0f0", ec="#333333", lw=0.7, zorder=3))
    ax.add_patch(Circle((0, 0), 0.80, fill=False, ec="#8a99a8", lw=0.6,
                        ls=":", zorder=3))
    for a in (35, 90, 145, 215, 325):
        th = np.deg2rad(a)
        ax.annotate("", xy=(0.93 * np.cos(th), 0.93 * np.sin(th)),
                    xytext=(1.28 * np.cos(th), 1.28 * np.sin(th)),
                    arrowprops=dict(arrowstyle="->", color=CB[1], lw=0.9),
                    zorder=5)
    ax.plot([0], [0], "o", ms=3.2, color="#B00020", zorder=6)
    ax.annotate("", xy=(0.78, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=CB[0], lw=1.0),
                zorder=6)
    ax.text(0.34, 0.07, "$r$", color=CB[0], fontsize=8)
    ax.text(-0.80, 0.33, "中心 $r=0$：\n$\\partial_r T=\\partial_r C=0$",
            fontsize=7.5, color="#B00020", ha="left", va="center",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8))
    ax.text(-0.10, -1.24, "表面 $r=R$：第三类边界（换热 $h$ ＋ 传质 $h_m$）",
            fontsize=7.5, color=CB[1], va="bottom", ha="center")
    ax.set_title("(b) 一维径向简化（横截面）", loc="left")
    return fig, axes


def trend(y, w=15, p=2):
    """高斯加权局部多项式趋势（与正文噪声诊断所用口径一致，无相位滞后）."""
    n = len(y)
    half = w // 2
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        idx = np.arange(lo, hi)
        off = idx - i
        A = np.vander(off, p + 1)
        wg = np.exp(-(off / (half * 0.8)) ** 2)
        c, *_ = np.linalg.lstsq(A * wg[:, None], y[idx] * wg, rcond=None)
        out[i] = c[-1]
    return out


# ---------------------------------------------------------------- fig 1
def fig_air():
    t, T, C = load_air()
    th = t / 3600.0
    Tr = trend(T)
    Cr = trend(C)
    res = T - Tr
    s = res.std()

    fig, axes = plt.subplots(
        3, 1, figsize=(W, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.25, 1.0], "hspace": 0.16},
        layout="constrained",
    )
    ax = axes[0]
    ax.plot(th, T, "-", color=CB[0], lw=1.1, label="实测序列（60 s 采样）")
    ax.set_ylabel("烘房温度 / °C")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("(a) 附件 1 烘房温度", loc="left")

    ax = axes[1]
    ax.plot(th, C, "-", color=CB[1], lw=1.1, label="实测序列（60 s 采样）")
    ax.set_ylabel("水分浓度 / (kg/kg)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("(b) 附件 1 烘房水分浓度", loc="left")

    ax = axes[2]
    ax.axhspan(-2 * s, 2 * s, color="#bbbbbb", alpha=0.45)
    ax.plot(th, res, "-", color=CB[2], lw=0.8)
    ax.axhline(0, color="#666666", lw=0.7)
    ax.text(0.985, 0.88, "$\\pm2\\sigma$（$\\sigma$=%.3f °C）" % s,
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    ax.set_ylabel("残差 / °C")
    ax.set_xlabel("时间 / h")
    ax.set_xlim(0, th[-1])
    ax.set_title("(c) 温度残差：高频交替型噪声，无异常点", loc="left")
    return fig, axes


# ---------------------------------------------------------------- fig 2
def fig_p1():
    d = np.load(os.path.join(PROJECT, "输出", "result1_data.npz"))
    tt = np.concatenate([[0.0], d["times"]]) / 60.0   # 含 t=0
    r = d["r_out_cm"]
    T = d["temp_out"].T             # (r, t)
    C = d["conc_out"].T

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.75), layout="constrained")
    for ax, Z, cmap, lab, title in (
        (axes[0], T, "viridis", "温度 / °C", "(a) 温度场 $T(r,t)$"),
        (axes[1], C, "magma", "含水率 / (kg/kg)", "(b) 水分浓度场 $C(r,t)$"),
    ):
        im = ax.pcolormesh(tt, r, Z, cmap=cmap, shading="auto")
        cb = fig.colorbar(im, ax=ax, pad=0.03)
        cb.set_label(lab)
        cb.ax.tick_params(labelsize=7)
        ax.set_xlabel("时间 / min")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("到中心距离 / cm")
    return fig, axes


# ---------------------------------------------------------------- fig 3
def fig_p2():
    d = np.load(os.path.join(PROJECT, "输出", "result23_data.npz"))
    tt = d["times"]
    r = np.arange(21) * 0.1
    T = d["temp21"]
    C = d["conc21"]
    hours = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ls = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
    mk = ["o", "s", "^", "v", "D", "P"]

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.75))
    for ax, Z, lab, title in (
        (axes[0], T, "温度 / °C", "(a) 温度剖面"),
        (axes[1], C, "含水率 / (kg/kg)", "(b) 水分浓度剖面"),
    ):
        for k, h in enumerate(hours):
            i = int(round(h * 3600 / 5.0))
            ax.plot(r, Z[i], linestyle=ls[k], marker=mk[k], ms=3.0, lw=1.1,
                    color=CB[k % len(CB)], label="%.1f h" % h)
        ax.set_xlabel("到中心距离 / cm")
        ax.set_ylabel(lab)
        ax.set_title(title, loc="left")
    axes[1].legend(frameon=False, ncol=2, loc="lower left")
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------- fig 4
def fig_p3():
    d = np.load(os.path.join(PROJECT, "输出", "result23_data.npz"))
    tt = d["times"] / 3600.0
    cen = d["conc21"][:, 0]
    sur = d["conc_surf"]
    dry = float(d["dry_time"][0]) / 3600.0

    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.axvspan(0, 12, color="#eef3f8", zorder=0)
    ax.axvspan(12, 36, color="#e3ecf5", zorder=0)
    ax.axvspan(36, dry, color="#d6e4f0", zorder=0)
    ax.plot(tt, cen, "-", color=CB[0], lw=1.5, label="中心 $C(0,t)$")
    ax.plot(tt, sur, "--", color=CB[1], lw=1.5, label="表面 $C(R,t)$")
    ax.axhline(0.15, color="#B00020", lw=1.2, ls="-.",
               label="烘干判据 0.15 kg/kg")
    ax.plot([dry], [cen[-1]], "o", color="#B00020", ms=4)
    ax.annotate("中心达标 $t$=%.2f h" % dry,
                xy=(dry, cen[-1]), xytext=(dry - 24, 0.62),
                fontsize=8, color="#B00020", ha="left",
                arrowprops=dict(arrowstyle="->", color="#B00020", lw=0.8))
    for x, txt in ((6, "快速干燥段"), (24, "降速干燥段"), (48, "极慢段")):
        ax.text(x, 2.35, txt, fontsize=7.5, color="#555555", ha="center")
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("水分浓度 / (kg/kg)")
    ax.set_xlim(0, dry + 2)
    ax.set_ylim(0, 2.7)
    ax.legend(frameon=False, loc="lower left", ncol=3, mode="expand",
              bbox_to_anchor=(0.0, 1.005, 1.0, 0.12))
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------- fig 5
def fig_p4():
    d = np.load(os.path.join(PROJECT, "输出", "result4_data.npz"))
    t = d["times"] / 3600.0
    R = d["radius"] * 100.0
    cen = d["conc21"][:, 0]
    dry = float(d["dry_time"][0]) / 3600.0
    ns = np.load("/tmp/p4_noshrink.npz")
    tn = ns["times"] / 3600.0
    cn = ns["conc"]
    dryn = float(ns["dry"]) / 3600.0

    fig, axes = plt.subplots(
        2, 1, figsize=(W, 4.1), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.5], "hspace": 0.12},
        layout="constrained",
    )
    ax = axes[0]
    ax.plot(t, R, "-", color=CB[0], lw=1.6, label="PCHIP 插值 $R(t)$")
    rad = np.loadtxt("/tmp/radius.csv", delimiter=",", skiprows=1)
    ax.plot(rad[:, 0] / 3600.0, rad[:, 1], "o", color="#111111", ms=2.4,
            label="附件 2 实测点（n=145）")
    ax.set_ylabel("半径 / cm")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("(a) 药材半径收缩历程", loc="left")

    ax = axes[1]
    ax.plot(t, cen, "-", color=CB[0], lw=1.6, label="含收缩（本文，%.1f h）" % dry)
    ax.plot(tn, cn, "--", color=CB[1], lw=1.6,
            label="不含收缩（对照，%.1f h）" % dryn)
    ax.axhline(0.15, color="#B00020", lw=1.1, ls="-.")
    ax.text(0.5, 0.22, "判据 0.15 kg/kg", color="#B00020", fontsize=7.5)
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("中心水分浓度 / (kg/kg)")
    ax.set_xlim(0, dryn + 4)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("(b) 中心含水率：收缩把烘干时长压缩约 63%", loc="left")
    return fig, axes


# ---------------------------------------------------------------- fig 6
def fig_sens():
    base = 60.93
    labels = ["$D$ ±20%", "$h_m$ ±20%", "$h$ ±20%"]
    lo = np.array([51.93, 59.88, 60.91])
    hi = np.array([74.53, 62.62, 60.95])
    dl = 100 * (lo - base) / base
    dh = 100 * (hi - base) / base

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7))
    ax = axes[0]
    y = np.arange(len(labels))
    ax.barh(y, dh, height=0.5, color=CB[1], alpha=0.9, label="参数 +20%")
    ax.barh(y, dl, height=0.5, color=CB[0], alpha=0.9, label="参数 −20%")
    ax.axvline(0, color="#444444", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("烘干时长相对变化 / %")
    ax.set_title("(a) 关键参数的灵敏度（基准 %.2f h）" % base, loc="left")
    ax.legend(frameon=False, loc="upper right", ncol=2,
              bbox_to_anchor=(1.0, 1.0))
    ax.set_xlim(-22, 38)
    ax.set_ylim(-0.85, 2.95)
    for i in range(len(labels)):
        if abs(dh[i]) < 0.3 and abs(dl[i]) < 0.3:
            ax.text(-1.4, y[i], "≈0%", va="center", ha="right", fontsize=7.5,
                    color="#333333")

    ax = axes[1]
    names = ["本文口径\n$\\partial(\\rho C)/\\partial t$",
             "经典 Fick\n$\\partial C/\\partial t$",
             "变密度守恒\n$D\\to(1+C)D$"]
    vals = np.array([60.93, 57.32, 51.39])
    ax.barh(np.arange(3), vals, height=0.5,
            color=[CB[0], CB[2], CB[3]])
    for i, v in enumerate(vals):
        ax.text(v + 0.6, i, "%.2f h" % v, va="center", fontsize=7.5)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("烘干时长 / h")
    ax.set_xlim(45, 68)
    ax.set_title("(b) 水分方程口径的影响", loc="left")
    fig.tight_layout()
    return fig, axes


def main():
    style()
    jobs = [
        ("fig0_geom", fig_geom),
        ("fig1_air", fig_air),
        ("fig2_p1", fig_p1),
        ("fig3_p2", fig_p2),
        ("fig4_p3", fig_p3),
        ("fig5_p4", fig_p4),
        ("fig6_sens", fig_sens),
    ]
    audit: list[tuple[str, str]] = []
    for name, fn in jobs:
        fig, _ = fn()
        png = os.path.join(FIGDIR, "_preview_%s.png" % name)
        visual_qa.render_preview(fig, png)
        issues = visual_qa.audit_layout(fig)
        audit += [(sev, "[%s] %s" % (name, msg)) for sev, msg in issues]
        saved = export_figure(
            fig, basename=os.path.join(FIGDIR, name),
            formats=["pdf", "png"], dpi=300,
            size_inches=fig.get_size_inches(), grayscale_preview=True,
            tight=False,
        )
        plt.close(fig)
        print("已导出 %s：%d 个文件" % (name, len(saved)))
    print("\n=== 版面自检 ===")
    visual_qa.print_report(audit)


if __name__ == "__main__":
    main()
