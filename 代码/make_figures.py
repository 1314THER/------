#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2026 A 题论文插图生成器（全量重绘版）。

工作流
------
本脚本同时遵守两套外部规范，取长补短：

1. **scipilot-figure-skill**（本地已装）——负责"出图自检闭环"：
   每张图渲染中分辨率 PNG → `visual_qa.audit_layout()` 程序自检
   （缺字乱码 / 文字越界 / 刻度重叠）→ 记录问题 → 导出矢量 PDF +
   300 dpi PNG + 灰度预览（色盲可辨性）。
2. **mathmodel-kit / mathmodel-figure**（GitHub 开源 skill，Apache-2.0，
   见 代码/figlib/figstyle.py 顶部来源声明）——负责"视觉语言"：
   身份色 / 方向色 / 层级色三职能配色、细轴线无上右边框、小字、
   仅单方向浅灰虚线网格。改用这套外观的一个直接效果是，论文里的图
   不会与该 skill 默认输出的成图撞样。
   其配套的 mathmodel-diagram 规范（示意图栅格先行、中文手动断行、
   连接器端点不搭线）用于本文件的全部原理示意图。

取材原则
--------
所有数据图的数值**一律来自 输出/ 下的结果文件**，不在绘图脚本里硬编码
任何结论数字；派生数据由 代码/prepare_figure_data.py 统一生成到
输出/figdata/。这样"图"与"表"必然同源，不会出现图文打架。

运行
----
    MPLCONFIGDIR=/tmp/mplcache /tmp/figvenv/bin/python 代码/make_figures.py
    ... --only s01,d05        # 只画指定图
    ... --list                # 列出全部图 id
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIGDIR = PROJECT / "论文" / "figs"
FIGDATA = PROJECT / "输出" / "figdata"
OUT = PROJECT / "输出"

sys.path.insert(0, str(HERE / "figlib"))
sys.path.insert(0, os.path.expanduser("~/.codex/skills/scipilot-figure-skill/scripts"))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import (                     # noqa: E402
    FancyArrow, FancyBboxPatch, Polygon, Rectangle, Circle, Wedge, Arc,
    Ellipse, FancyArrowPatch,
)
from matplotlib.lines import Line2D                  # noqa: E402

import figstyle as FS                                # noqa: E402
from export_figure import export_figure              # noqa: E402
import visual_qa                                     # noqa: E402


# ======================================================================
# 版式常量
# ======================================================================
W = 15.6 / 2.54          # A4 正文宽度 15.6 cm = 6.142 in，不做二次缩放

C_MAIN = FS.COLOR_MAIN
C_MAIN_L = FS.COLOR_MAIN_LIGHT
C_MAIN_P = FS.COLOR_MAIN_PALE
C_ORANGE = FS.ACCENT_ORANGE
C_PURPLE = FS.ACCENT_PURPLE
C_TEAL = FS.ACCENT_TEAL
C_CORAL = FS.ACCENT_CORAL
C_LAV = FS.ACCENT_LAVENDER
C_BASE = FS.COLOR_BASELINE
C_INK = FS.COLOR_INK
C_POS = FS.COLOR_POSITIVE
C_NEG = FS.COLOR_NEGATIVE
IDENT = FS.IDENTITY_PALETTE

# 顺序型 / 发散型色图统一用图库自带的、与身份色同族的两条
SEQ = FS.SEQUENTIAL_CMAP
DIV = FS.DIVERGENT_CMAP

AUDIT: list[tuple[str, str]] = []
ONLY: set[str] = set()


# ======================================================================
# 数据读取
# ======================================================================
def load_air():
    d = np.loadtxt(FIGDATA / "air.csv", delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1], d[:, 2]


def load_radius():
    d = np.loadtxt(FIGDATA / "radius.csv", delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1]


def load_npz(name, base=None):
    """在 输出/ 与 输出/figdata/ 两处按序查找（结果文件与派生文件分开放）。"""
    if base is not None:
        return np.load(base / name)
    for d in (OUT, FIGDATA):
        if (d / name).exists():
            return np.load(d / name)
    raise FileNotFoundError(name)


def load_rho():
    return json.loads((FIGDATA / "rho_analysis.json").read_text(encoding="utf-8"))


def load_sens():
    rows = list(csv.DictReader(open(FIGDATA / "sens.csv")))
    return {r["param"] + r["scale"]: float(r["t_center_h"]) for r in rows}, rows


def load_grid_csv():
    rows = list(csv.DictReader(open(OUT / "问题三-网格与口径对照.csv")))
    d = {}
    for r in rows:
        d[(r["form"], int(r["N"]), float(r["dt_s"]))] = float(r["t_center_s"])
    return d


# ======================================================================
# 绘图小工具
# ======================================================================
def trend(y, w=15, p=2):
    """高斯加权局部多项式趋势（无相位滞后），与正文噪声诊断口径一致。"""
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


def panel(ax, label, dx=-0.10, dy=1.02):
    FS.add_panel_label(ax, label, x=dx, y=dy)


def style(ax, grid="y"):
    return FS.style_axes(ax, grid=grid)


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec=C_INK, lw=0.9,
        fs=7.6, tc=C_INK, radius=0.02, weight="normal", ha="center"):
    """示意图里的圆角文本框。坐标用轴归一化坐标，便于栅格对齐。"""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=%.4f" % radius,
        fc=fc, ec=ec, lw=lw, mutation_aspect=1.0, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha=ha, va="center", fontsize=fs,
            color=tc, zorder=3, fontweight=weight, linespacing=1.45)


def arrow(ax, p0, p1, color=C_INK, lw=0.9, style_="-|>", rad=0.0, ls="-",
          ms=6.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style_, mutation_scale=ms, lw=lw, color=color,
        linestyle=ls, shrinkA=0, shrinkB=0, zorder=4,
        connectionstyle="arc3,rad=%.3f" % rad))


def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


# ======================================================================
# 出图 + 自检闭环
# ======================================================================
def emit(name, fig, tight=False):
    """渲染预览 → 程序自检 → 导出 PDF/PNG/灰度。"""
    size = tuple(fig.get_size_inches())
    fake_bold(fig)
    preview = FIGDIR / ("_preview_%s.png" % name)
    visual_qa.render_preview(fig, str(preview), dpi=150)
    issues = visual_qa.audit_layout(fig)
    AUDIT.extend((sev, "[%s] %s" % (name, msg)) for sev, msg in issues)
    export_figure(
        fig, basename=str(FIGDIR / name), formats=["pdf", "png"], dpi=300,
        size_inches=size, grayscale_preview=True,
        tight=tight,
    )
    plt.close(fig)
    bad = [m for s, m in issues if s in ("WARN", "FAIL")]
    flag = "OK" if not bad else "自检 %d 项" % len(bad)
    print("  %-22s %-12s %s" % (name, "%.2fx%.2f in" % size, flag))


def fake_bold(fig, amount=0.36):
    """用"同色描边"模拟粗体。

    本机可用的全覆盖字体（Arial Unicode MS、Heiti SC）都只有 400 一个字重，
    matplotlib 遇到 weight='bold' 会静默退回 400（stderr 打印
    "Failed to find font weight bold"），标题与面板标号因此失去层级。
    这里给所有声明为 bold 的文字加一圈同色细描边，等效于加粗，
    且对中文、拉丁、数学符号一致生效。
    """
    import matplotlib.patheffects as pe

    def visit(t):
        w = t.get_fontweight()
        if w in ("bold", "heavy", 700, 800, 900, "semibold", 600):
            lw = amount * max(t.get_fontsize(), 6.0) / 10.0
            t.set_path_effects([pe.withStroke(
                linewidth=lw, foreground=t.get_color())])

    for ax in fig.axes:
        for t in (ax.title, ax.xaxis.label, ax.yaxis.label):
            visit(t)
        for t in list(ax.texts) + list(ax.get_xticklabels()) \
                + list(ax.get_yticklabels()):
            visit(t)


def run(jobs):
    for name, fn in jobs:
        if ONLY and name not in ONLY:
            continue
        fig = fn()
        emit(name, fig)


# ======================================================================
# 一、原理示意图
# ======================================================================
def s01_geometry():
    """圆柱几何与一维径向简化。"""
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.85), layout="constrained")

    ax = axes[0]
    blank(ax)
    ax.add_patch(Rectangle((0.30, 0.26), 0.36, 0.44, fc="#EAF1F8",
                           ec=C_INK, lw=1.0, zorder=1))
    ax.add_patch(Ellipse((0.48, 0.70), 0.36, 0.12, fc="#DCE8F4",
                         ec=C_INK, lw=1.0, zorder=2))
    ax.add_patch(Arc((0.48, 0.26), 0.36, 0.12, theta1=180, theta2=360,
                     ec=C_INK, lw=1.0, zorder=2))
    ax.plot([0.48, 0.48], [0.13, 0.86], "-.", color=C_CORAL, lw=1.0, zorder=3)
    ax.text(0.492, 0.815, "对称轴", color=C_CORAL, fontsize=7.2)
    ax.annotate("", xy=(0.66, 0.55), xytext=(0.48, 0.55),
                arrowprops=dict(arrowstyle="<->", color=C_MAIN, lw=1.0))
    ax.text(0.505, 0.572, "$R=2$ cm", color=C_MAIN, fontsize=7.2)
    ax.annotate("", xy=(0.235, 0.26), xytext=(0.235, 0.70),
                arrowprops=dict(arrowstyle="<->", color=C_TEAL, lw=1.0))
    ax.text(0.16, 0.48, "$L=25$ cm", color=C_TEAL, fontsize=7.2, rotation=90,
            va="center")
    ax.annotate("", xy=(0.88, 0.90), xytext=(0.48, 0.90),
                arrowprops=dict(arrowstyle="->", color=C_INK, lw=0.8))
    ax.text(0.885, 0.90, "$r$", fontsize=7.6, va="center")
    ax.annotate("", xy=(0.43, 0.30), xytext=(0.43, 0.70),
                arrowprops=dict(arrowstyle="->", color=C_INK, lw=0.8))
    ax.text(0.405, 0.295, "$z$", fontsize=7.6, ha="center", va="top")
    for yy, txt, col in ((0.635, "对流换热 $h$", C_ORANGE),
                         (0.435, "对流传质 $h_m$", C_PURPLE)):
        ax.annotate("", xy=(0.685, yy), xytext=(0.90, yy),
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.9))
        ax.text(0.905, yy, txt, color=col, fontsize=7.2, va="center")
    ax.text(0.02, 0.06, "端面 $2\\pi R^2$ 仅占侧面 $2\\pi RL$ 的 $R/L=8\\%$\n"
                        "轴向输运可忽略 → 一维径向轴对称",
            fontsize=7.0, va="bottom", color=C_INK)
    ax.set_title("(a) 圆柱几何与柱坐标", loc="left")

    ax = axes[1]
    ax.set_xlim(-1.34, 1.42)
    ax.set_ylim(-1.48, 1.30)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.0, fc="#EAF1F8", ec=C_INK, lw=1.1, zorder=1))
    for a in range(0, 360, 30):
        th = np.deg2rad(a)
        ax.plot([0, np.cos(th)], [0, np.sin(th)], color="#C7D4E0", lw=0.5,
                zorder=2)
    ax.add_patch(Wedge((0, 0), 1.0, 0, 360, width=0.18, fc="#D3E1EF",
                       ec=C_INK, lw=0.7, zorder=3))
    ax.add_patch(Circle((0, 0), 0.82, fill=False, ec="#8A99A8", lw=0.5,
                        ls=":", zorder=3))
    for a in (35, 90, 145, 215, 325):
        th = np.deg2rad(a)
        ax.annotate("", xy=(0.92 * np.cos(th), 0.92 * np.sin(th)),
                    xytext=(1.26 * np.cos(th), 1.26 * np.sin(th)),
                    arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=0.8),
                    zorder=5)
    ax.plot([0], [0], "o", ms=3.0, color=C_CORAL, zorder=6)
    ax.annotate("", xy=(0.78, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C_MAIN, lw=1.0),
                zorder=6)
    ax.text(0.34, 0.07, "$r$", color=C_MAIN, fontsize=7.6)
    ax.text(-0.78, 0.36, "中心 $r=0$：\n$\\partial_rT=\\partial_rC=0$",
            fontsize=7.0, color=C_CORAL, ha="left", va="center",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
    ax.text(0.0, -1.40, "表面 $r=R$：第三类边界\n（换热 $h$ ＋ 传质 $h_m$）",
            fontsize=7.0, color=C_ORANGE, va="bottom", ha="center")
    ax.set_title("(b) 一维径向简化（横截面）", loc="left")
    return fig


def s02_physics():
    """热风烘干的两条传输通道：热量向内导热，水分向外扩散＋表面蒸发。"""
    fig, ax = plt.subplots(figsize=(W, 3.15), layout="constrained")
    blank(ax)

    ax.add_patch(Rectangle((0.05, 0.10), 0.30, 0.80, fc="#F3F7FB",
                           ec="none", zorder=0))
    ax.text(0.065, 0.925, "烘房热风（边界激励）", fontsize=7.6, color=C_INK)
    ax.text(0.065, 0.045, "恒温干燥段外推：$T_\\infty=50\\,^\\circ$C，"
                          "$C_\\infty=0.05$ kg/kg", fontsize=7.0, color=C_BASE)

    # 药材横截面（扇形）
    cx, cy, rr = 0.72, 0.50, 0.30
    ax.add_patch(Circle((cx, cy), rr, fc="#FBF6EE", ec=C_INK, lw=1.1, zorder=1))
    ax.add_patch(Wedge((cx, cy), rr, 90, 180, fc="#F6EDDE", ec="none", zorder=2))
    ax.plot([cx, cx], [cy, cy + rr], "-.", color=C_CORAL, lw=0.9, zorder=3)
    ax.plot([cx - rr, cx + rr], [cy, cy], ":", color="#9AA7B4", lw=0.9, zorder=3)
    ax.text(cx - 0.012, cy + 0.035, "中心", fontsize=7.0, color=C_CORAL,
            ha="right")
    ax.annotate("", xy=(cx + rr * 0.70, cy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color=C_MAIN, lw=0.9))
    ax.text(cx + 0.11, cy + 0.03, "$r$", color=C_MAIN, fontsize=7.6)

    # 热量通道：由外向内
    for i, dy in enumerate((0.20, 0.06, -0.08, -0.22)):
        ax.annotate("", xy=(cx + rr * 0.92, cy + dy),
                    xytext=(cx + rr * 0.32, cy + dy * 0.75),
                    arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=1.0),
                    zorder=5)
    ax.text(cx + 0.20, cy + 0.34, "热量内传\n导热 $k\\,\\partial_rT$",
            fontsize=7.2, color=C_ORANGE, ha="center")

    # 水分通道：由内向外
    for dy in (0.17, 0.02, -0.13, -0.28):
        ax.annotate("", xy=(cx + rr * 1.02, cy + dy),
                    xytext=(cx + rr * 0.42, cy + dy * 0.75),
                    arrowprops=dict(arrowstyle="-|>", color=C_TEAL, lw=1.0),
                    zorder=5)
    ax.text(cx + 0.20, cy - 0.40, "水分外移\n扩散 $D\\,\\partial_rC$",
            fontsize=7.2, color=C_TEAL, ha="center")

    # 表面交换
    for yy, txt, col, dy0 in ((cy + 0.435, "对流换热", C_ORANGE, 0.0),
                              (cy - 0.435, "对流传质＋蒸发", C_PURPLE, 0.0)):
        ax.annotate("", xy=(cx + rr + 0.002, cy + (0.14 if dy0 == 0 and yy > cy else -0.14)),
                    xytext=(cx - 0.02, yy),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0,
                                    connectionstyle="arc3,rad=0.25"), zorder=5)
        ax.text(cx - 0.05, yy + 0.055, txt, fontsize=7.2, color=col, ha="center")

    ax.text(0.05, 0.62, "边界条件\n$-k\\,\\partial_rT|_R=h(T_s-T_\\infty)$\n"
                        "$-D\\,\\partial_rC|_R=h_m(C_s-C_\\infty)$",
            fontsize=7.2, color=C_INK, va="center")
    ax.text(0.05, 0.30, "内部\n$\\rho c_p\\,\\partial_tT=\\nabla\\!\\cdot\\!(k\\nabla T)$\n"
                        "$\\partial_tC=\\nabla\\!\\cdot\\!(D\\nabla C)$",
            fontsize=7.2, color=C_INK, va="center")
    ax.text(0.95, 0.955, "两方程在问题一解耦、问题二起强耦合",
            fontsize=7.0, color=C_BASE, ha="right")
    return fig


def s03_discretisation():
    """节点中心有限体积离散：网格、三类控制体权重、三对角结构。"""
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.45), layout="constrained")

    # (a) 网格与控制体
    ax = axes[0]
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.30, 1.05)
    ax.axis("off")
    n = 8
    xs = np.arange(n + 1) / n
    for i in range(n + 1):
        ax.plot([xs[i], xs[i]], [0.05, 0.86], color="#C7D4E0", lw=0.6, zorder=1)
    for i in range(n + 1):
        ax.plot([xs[i]], [0.455], "o", ms=3.4, color=C_MAIN, zorder=4)
    for i in range(n):
        ax.add_patch(Rectangle((xs[i], 0.30), 1.0 / n, 0.31,
                               fc=(C_MAIN_P if i % 2 == 0 else "#FFFFFF"),
                               ec="#9AA7B4", lw=0.5, zorder=2))
    ax.annotate("", xy=(xs[1], 0.20), xytext=(xs[0], 0.20),
                arrowprops=dict(arrowstyle="<->", color=C_INK, lw=0.7))
    ax.text(xs[0] + 0.5 / n, 0.135, "$\\Delta r$", fontsize=7.0, ha="center")
    ax.annotate("", xy=(1.03, 0.455), xytext=(0.0, 0.455),
                arrowprops=dict(arrowstyle="->", color=C_MAIN, lw=0.8))
    ax.text(1.045, 0.455, "$r$", fontsize=7.6, va="center")
    ax.text(0.0, 0.90, "$r_0=0$", fontsize=7.0, ha="left", color=C_CORAL)
    ax.text(1.0, 0.90, "$r_N=R$", fontsize=7.0, ha="right", color=C_CORAL)
    ax.text(0.5, -0.02, "节点中心有限体积：未知量取节点值，\n"
                        "通量取界面值，格式守恒",
            fontsize=7.0, ha="center", va="top")
    ax.set_title("(a) 网格与节点控制体", loc="left")

    # (b) 三类权重
    ax = axes[1]
    blank(ax)
    y0 = 0.66
    for i, (lab, wid, col, note) in enumerate((
            ("$i=0$ 中心", 0.22, C_CORAL, "$w_0=\\Delta r^2/8$"),
            ("$1\\leq i\\leq N-1$ 内部", 0.30, C_MAIN, "$w_i=r_i\\Delta r$"),
            ("$i=N$ 表面", 0.26, C_ORANGE, "$w_N=\\frac{\\Delta r}{2}(R-\\frac{\\Delta r}{4})$"))):
        yy = y0 - i * 0.26
        ax.add_patch(Rectangle((0.10, yy - 0.055), wid, 0.11, fc=col,
                               alpha=0.25, ec=col, lw=0.8))
        ax.text(0.10, yy + 0.10, lab, fontsize=7.2, color=col)
        ax.text(0.56, yy, note, fontsize=7.0, va="center", color=C_INK)
    ax.text(0.05, 0.06, "直接用 $r_i\\Delta r$ 会把中心权重记成 0、\n"
                        "表面权重放大一倍，产生约 1% 的虚假守恒残差",
            fontsize=7.0, va="bottom", color=C_BASE)
    ax.set_title("(b) 控制体权重的三处修正", loc="left")

    # (c) 三对角结构
    ax = axes[2]
    n2 = 7
    ax.set_xlim(-0.6, n2 - 0.4)
    # 注意 y 轴反向：下限是"画布底部"。上下各留一行的空白放说明文字，
    # 否则说明文字会跑到坐标框外压住小标题（P18 类排版坑）。
    ax.set_ylim(n2 + 0.75, -1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    for i in range(n2):
        for j in range(n2):
            filled = abs(i - j) <= 1
            ax.add_patch(Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                   fc=C_MAIN if filled else "#FFFFFF",
                                   alpha=1.0 if filled else 1.0,
                                   ec="#C7D4E0", lw=0.6))
    ax.text(n2 / 2 - 0.5, -1.05, "非零元仅 3 条对角线", fontsize=7.0,
            ha="center", va="center", color=C_INK)
    ax.text(n2 / 2 - 0.5, n2 + 0.42, "后向 Euler＋Picard 迭代后仍为三对角",
            fontsize=6.8, ha="center", va="center", color=C_BASE)
    ax.set_title("(c) 系数矩阵结构", loc="left")
    return fig


def s04_flowchart():
    """数值求解流程框图。"""
    fig, ax = plt.subplots(figsize=(W, 3.5), layout="constrained")
    blank(ax)

    boxes = {
        "start": (0.02, 0.855, 0.20, 0.105, "读取附件 1 / 附件 2\n与附录物性关系式",
                  "#EAF1F8"),
        "grid": (0.02, 0.700, 0.20, 0.095, "建立贴体网格\n$\\xi_i=i/N$，$\\Delta t$",
                 "#FFFFFF"),
        "init": (0.02, 0.555, 0.20, 0.095, "赋初值\n$T=28\\,^\\circ$C，$C=2.55$",
                 "#FFFFFF"),
        "time": (0.30, 0.700, 0.185, 0.115, "时间层推进\n$t^{n+1}=t^n+\\Delta t$",
                 "#EAF1F8"),
        "pic": (0.30, 0.520, 0.185, 0.115, "Picard 迭代\n滞后物性 $\\rho,c_p,k,D$",
                "#FFF4E6"),
        "asm": (0.30, 0.340, 0.185, 0.115, "组装三对角方程组\n（扩散＋对流＋第三类边界）",
                "#FFFFFF"),
        "tho": (0.30, 0.160, 0.185, 0.115, "追赶法求解\n$O(N)$", "#FFFFFF"),
        "conv": (0.555, 0.340, 0.185, 0.115, "两场变化量\n$<\\varepsilon$？",
                 "#FFF0F0"),
        "dry": (0.555, 0.140, 0.185, 0.105, "中心 $C(0,t)<0.15$？",
                "#FFF0F0"),
        "out": (0.795, 0.480, 0.19, 0.175,
                "输出\n$T(r,t)$、$C(r,t)$\n表 1–表 6\nresult1–4.xlsx", "#EAF7F3"),
        "end": (0.795, 0.140, 0.19, 0.135, "烘干时间\n$t_{dry}$\n（问题三、四）",
                "#EAF7F3"),
    }
    for key, (x, y, w, h, txt, fc) in boxes.items():
        box(ax, x, y, w, h, txt, fc=fc, fs=7.0)

    def c(k, side):
        x, y, w, h = boxes[k][:4]
        return {"b": (x + w / 2, y), "t": (x + w / 2, y + h),
                "l": (x, y + h / 2), "r": (x + w, y + h / 2)}[side]

    arrow(ax, c("start", "b"), c("grid", "t"), C_INK, 0.8, ms=5)
    arrow(ax, c("grid", "b"), c("init", "t"), C_INK, 0.8, ms=5)
    arrow(ax, c("init", "r"), (0.30 + 0.0925, 0.555 + 0.0475), C_INK, 0.8, ms=5)
    arrow(ax, c("time", "b"), c("pic", "t"), C_MAIN, 0.9, ms=5)
    arrow(ax, c("pic", "b"), c("asm", "t"), C_MAIN, 0.9, ms=5)
    arrow(ax, c("asm", "b"), c("tho", "t"), C_MAIN, 0.9, ms=5)
    arrow(ax, c("tho", "l"), (0.245, 0.2175), C_MAIN, 0.9, ms=5)
    arrow(ax, (0.245, 0.2175), (0.245, 0.5775), C_MAIN, 0.9, ms=5)
    arrow(ax, (0.245, 0.5775), (0.30, 0.5775), C_MAIN, 0.9, ms=5)
    arrow(ax, c("asm", "r"), (0.555, 0.3975), C_MAIN, 0.9, ms=5)
    arrow(ax, c("conv", "l"), (0.485, 0.3975), C_MAIN, 0.9, ms=5, ls="--")
    ax.text(0.492, 0.418, "否", fontsize=7.0, color=C_MAIN)
    arrow(ax, c("conv", "b"), c("dry", "t"), C_MAIN, 0.9, ms=5)
    ax.text(0.6475, 0.268, "是", fontsize=7.0, color=C_MAIN, ha="center")
    arrow(ax, (0.795, 0.5675), c("out", "l"), C_MAIN, 0.9, ms=5)
    arrow(ax, (0.6475, 0.14), c("end", "l"), C_MAIN, 0.9, ms=5)
    ax.text(0.6475, 0.105, "是", fontsize=7.0, color=C_MAIN, ha="center")
    arrow(ax, c("dry", "r"), (0.875, 0.1925), C_MAIN, 0.9, ms=5)
    arrow(ax, (0.875, 0.1925), (0.875, 0.135), C_MAIN, 0.9, ms=5)
    ax.text(0.885, 0.155, "否 → 下一时间层", fontsize=7.0, color=C_BASE)
    ax.text(0.02, 0.44, "外层：时间推进\n内层：Picard 迭代", fontsize=7.2,
            color=C_INK, va="top")
    ax.text(0.02, 0.30, "收敛判据\n$\\max|\\Delta C|,|\\Delta T|<10^{-11}$",
            fontsize=7.0, color=C_BASE, va="top")
    return fig


def s05_xi_transform():
    """贴体坐标变换：动边界 → 固定区域。"""
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.5), layout="constrained")

    for ax, (rr, title, note) in zip(axes, (
            (1.0, "(a) 物理坐标 $r$：初始态", "$t=0$，$R_0=2$ cm"),
            (0.6, "(b) 物理坐标 $r$：收缩后", "$t=t_{dry}$，$R=1.20$ cm"),
            (1.0, "(c) 贴体坐标 $\\xi=r/R(t)$", "动边界恒为 $\\xi=1$"))):
        ax.set_xlim(-1.30, 1.30)
        ax.set_ylim(-1.30, 1.30)
        ax.set_aspect("equal")
        ax.axis("off")
        if title.startswith("(c)"):
            ax.add_patch(Rectangle((-0.92, -0.92), 1.84, 1.84, fc="#F3F7FB",
                                   ec=C_INK, lw=1.0))
            for xi in np.linspace(0, 1, 6):
                ax.add_patch(Rectangle((-0.92, -0.92 + 1.84 * xi), 1.84,
                                       1.84 * 0.16, fc=C_MAIN_P,
                                       ec="#FFFFFF", lw=0.6))
            ax.annotate("", xy=(0.0, 1.02), xytext=(0.0, -1.02),
                        arrowprops=dict(arrowstyle="<->", color=C_INK, lw=0.8))
            ax.text(0.06, 0.0, "$\\xi$", fontsize=7.6, rotation=90, va="center")
            ax.text(0.0, 1.12, "网格不随时间移动", fontsize=7.0, ha="center",
                    color=C_BASE)
        else:
            ax.add_patch(Circle((0, 0), rr, fc="#FBF6EE", ec=C_INK, lw=1.0))
            ax.add_patch(Wedge((0, 0), rr, 0, 360, width=rr * 0.16,
                               fc="#F0E4D0", ec="none"))
            for a in np.linspace(0, 2 * np.pi, 25)[:-1]:
                ax.plot([0, rr * np.cos(a)], [0, rr * np.sin(a)],
                        color="#DED3C2", lw=0.4)
            ax.annotate("", xy=(0.98, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=C_MAIN, lw=0.9))
            if title.startswith("(b)"):
                ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=C_CORAL,
                                    lw=0.9, ls="--"))
                ax.text(1.0, -1.16, "原半径位置", fontsize=7.0, color=C_CORAL,
                        ha="right")
            ax.text(rr + 0.05, 0.06, "$R(t)$", fontsize=7.2, color=C_MAIN)
        ax.text(0.0, -1.24, note, fontsize=7.0, ha="center", color=C_BASE)
        ax.set_title(title, loc="left")

    for i in range(2):
        arrow(axes[i], (1.02, 0.55), (1.20, 0.55), C_INK, 0.9, ms=6)
    return fig


def s06_stages():
    """干燥三阶段与干燥速率曲线（示意图，非计算结果）。"""
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), layout="constrained")

    t = np.linspace(0, 60, 600)
    u = np.maximum(t - 2.0, 0.0)          # 先截断再取幂，避免负底数的无效开方
    c = np.where(t < 2.0, 2.55,
                 np.where(t < 30, 2.55 - 0.062 * u ** 0.93,
                          0.30 * np.exp(-(t - 30) / 12.0) + 0.051))
    ax = axes[0]
    style(ax)
    ax.plot(t, c, color=C_MAIN, lw=1.6)
    ax.axhline(0.15, color=C_CORAL, lw=1.0, ls="-.")
    ax.text(58, 0.20, "判据 0.15", color=C_CORAL, fontsize=7.0, ha="right")
    ax.axvspan(0, 2, color="#EEF3F9", zorder=0)
    ax.text(1.0, 2.72, "预热", fontsize=7.0, ha="center", color=C_BASE)
    ax.text(16, 2.72, "恒速干燥段", fontsize=7.0, ha="center", color=C_BASE)
    ax.text(44, 2.72, "降速干燥段", fontsize=7.0, ha="center", color=C_BASE)
    ax.annotate("表面传质控制", xy=(14, 1.35), fontsize=7.0, color=C_TEAL)
    ax.annotate("内部扩散控制", xy=(38, 0.85), fontsize=7.0, color=C_PURPLE)
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("含水率 $C$ / (kg/kg)")
    ax.set_title("(a) 含水率历程的三段结构（示意）", loc="left")
    ax.set_ylim(0, 3.0)

    ax = axes[1]
    style(ax)
    rate = np.gradient(-c, t)
    ax.plot(c[10:], rate[10:], color=C_MAIN, lw=1.6)
    ax.set_xlabel("含水率 $C$ / (kg/kg)")
    ax.set_ylabel("干燥速率 $-dC/dt$ / (1/h)")
    ax.invert_xaxis()
    ax.annotate("恒速段：速率近常数", xy=(1.6, rate[np.argmin(abs(c - 1.6))]),
                xytext=(2.35, rate.max() * 0.72), fontsize=7.0, color=C_TEAL,
                arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=0.7))
    ax.annotate("降速段：速率随 $C$ 下降", xy=(0.35, rate[np.argmin(abs(c - 0.35))]),
                xytext=(1.65, rate.max() * 0.28), fontsize=7.0, color=C_PURPLE,
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=0.7))
    ax.set_title("(b) 干燥速率曲线（示意）", loc="left")
    return fig


def s07_roadmap():
    """技术路线图。"""
    fig, ax = plt.subplots(figsize=(W, 3.6), layout="constrained")
    blank(ax)
    stages = [
        ("问题一\n预热平衡\n0–30 min", "#EAF1F8",
         "常数物性＋$D(C)$\n热质解耦\n两方程分别求解", "表 1 / 表 2\nresult1.xlsx"),
        ("问题二\n全流程耦合\n0–3 h 展示", "#FFF4E6",
         "$\\rho,c_p,k,D$ 全部随\n$C$（$D$ 还随 $T$）\nPicard 联立迭代",
         "表 3 / 表 4\nresult2.xlsx"),
        ("问题三\n烘干时长\n判据 $C<0.15$", "#EAF7F3",
         "中心处达标即全局达标\n网格 / 时间步 / 口径\n三因素对照",
         "表 5\nresult3.xlsx\n$t_{dry}=57.42$ h"),
        ("问题四\n尺寸变化\n含收缩", "#F7F0F8",
         "附件 2 实测 $R(t)$\n贴体材料坐标\nPCHIP 保单调插值",
         "表 6\nresult4.xlsx\n$t_{dry}=51.08$ h"),
    ]
    x0, wt, gap = 0.015, 0.225, 0.028
    for i, (title, fc, method, res) in enumerate(stages):
        x = x0 + i * (wt + gap)
        box(ax, x, 0.72, wt, 0.235, title, fc=fc, fs=7.8, weight="bold")
        box(ax, x, 0.375, wt, 0.275, method, fc="#FFFFFF", fs=7.0)
        box(ax, x, 0.075, wt, 0.215, res, fc="#F3F7FB", fs=7.0)
        arrow(ax, (x + wt / 2, 0.72), (x + wt / 2, 0.655), C_BASE, 0.7, ms=5)
        arrow(ax, (x + wt / 2, 0.375), (x + wt / 2, 0.292), C_BASE, 0.7, ms=5)
        if i < 3:
            arrow(ax, (x + wt, 0.8375), (x + wt + gap, 0.8375), C_MAIN,
                  1.0, ms=7)
    ax.text(0.015, 0.985, "模型逐步升级：物性常数 → 物性随 $C,T$ 变化 → 加判据 → 加动边界",
            fontsize=7.2, color=C_INK, va="top")
    ax.text(0.985, 0.985, "共同基础：圆柱一维径向有限体积＋后向 Euler＋追赶法",
            fontsize=7.2, color=C_BASE, va="top", ha="right")
    return fig


def s08_conditions():
    """定解条件与两类边界。"""
    fig, ax = plt.subplots(figsize=(W, 2.7), layout="constrained")
    blank(ax)
    n = 10
    for i in range(n + 1):
        x = 0.10 + 0.72 * i / n
        ax.plot([x, x], [0.30, 0.72], color="#C7D4E0", lw=0.5, zorder=1)
    for i in range(n + 1):
        ax.plot([0.10 + 0.72 * i / n], [0.51], "o", ms=3.0, color=C_MAIN,
                zorder=3)
    ax.add_patch(Rectangle((0.10, 0.42), 0.72, 0.18, fc="#F3F7FB", ec="none",
                           zorder=0))
    ax.plot([0.10, 0.82], [0.30, 0.30], color=C_INK, lw=0.9)
    ax.plot([0.10, 0.82], [0.72, 0.72], color=C_INK, lw=0.9)
    ax.text(0.10, 0.245, "$r=0$", ha="center", fontsize=7.4, color=C_CORAL)
    ax.text(0.82, 0.245, "$r=R$", ha="center", fontsize=7.4, color=C_ORANGE)
    ax.text(0.10, 0.765, "中心：对称", ha="center", fontsize=7.2, color=C_CORAL)
    ax.text(0.82, 0.765, "表面：第三类", ha="center", fontsize=7.2, color=C_ORANGE)
    ax.annotate("", xy=(0.86, 0.51), xytext=(0.82, 0.51),
                arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1.0))
    ax.text(0.87, 0.51, "$h,\\ h_m$", fontsize=7.2, color=C_ORANGE, va="center")
    ax.text(0.02, 0.90, "初始条件", fontsize=7.6, color=C_INK)
    ax.text(0.02, 0.79, "$T(r,0)=28\\,^\\circ$C\n$C(r,0)=2.55$ kg/kg",
            fontsize=7.2, va="top")
    ax.text(0.02, 0.60, "中心对称条件", fontsize=7.6, color=C_CORAL)
    ax.text(0.02, 0.49, "$\\partial_rT|_{r=0}=0$\n$\\partial_rC|_{r=0}=0$",
            fontsize=7.2, va="top")
    ax.text(0.02, 0.30, "表面第三类边界", fontsize=7.6, color=C_ORANGE)
    ax.text(0.02, 0.19, "$-k\\,\\partial_rT|_R=h(T_s-T_\\infty(t))$\n"
                        "$-D\\,\\partial_rC|_R=h_m(C_s-C_\\infty(t))$",
            fontsize=7.2, va="top")
    ax.text(0.10, 0.10, "两式在量纲上自洽；传质式的密度口径需显式约定（见正文口径一节）",
            fontsize=7.0, color=C_BASE, va="top")
    return fig


SCHEMATICS = [
    ("s01_geometry", s01_geometry),
    ("s02_physics", s02_physics),
    ("s03_discretisation", s03_discretisation),
    ("s04_flowchart", s04_flowchart),
    ("s05_xi_transform", s05_xi_transform),
    ("s06_stages", s06_stages),
    ("s07_roadmap", s07_roadmap),
    ("s08_conditions", s08_conditions),
]


# ======================================================================
# 二、数据图：附件数据与预处理
# ======================================================================
def d01_air_series():
    """附件 1：实测序列、趋势与残差（温度与水分浓度各一行）。"""
    t, T, C = load_air()
    th = t / 3600.0
    Tr, Cr = trend(T), trend(C)
    sT, sC = (T - Tr).std(), (C - Cr).std()

    fig, axes = plt.subplots(2, 2, figsize=(W, 4.0), sharex="col",
                             layout="constrained")
    for j, (y, ytr, lab, col, sg, unit, nm) in enumerate((
            (T, Tr, "烘房温度 / $^\\circ$C", C_MAIN, sT, "$^\\circ$C", "温度"),
            (C, Cr, "水分浓度 / (kg/kg)", C_TEAL, sC, "kg/kg", "水分浓度"))):
        ax = axes[0, j]
        style(ax)
        ax.plot(th, y, "-", color=col, lw=1.0, label="实测序列（60 s 采样）")
        ax.plot(th, ytr, "--", color=C_ORANGE, lw=1.0, label="高斯加权趋势")
        ax.set_ylabel(lab)
        ax.legend(loc="lower right", fontsize=7.0)
        ax.set_title("(%s) 附件 1 %s，$n=%d$" % ("ab"[j], nm, len(y)),
                     loc="left")

        ax = axes[1, j]
        style(ax, grid="both")
        ax.axhspan(-2 * sg, 2 * sg, color="#D8D8D8", alpha=0.6, zorder=0)
        ax.plot(th, y - ytr, "-", color=C_PURPLE, lw=0.7)
        ax.axhline(0, color=C_BASE, lw=0.7)
        ax.set_xlabel("时间 / h")
        ax.set_ylabel("残差 / %s" % unit)
        # σ 用 3 位有效数字，避免小量显示成 0.000
        ax.text(0.985, 0.90, "$\\pm2\\sigma$，$\\sigma=%.3g$ %s" % (sg, unit),
                transform=ax.transAxes, ha="right", va="top", fontsize=7.0,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
        ax.set_title("(%s) %s残差的时序" % ("cd"[j], nm), loc="left")
    axes[1, 0].set_xlim(0, th[-1])
    return fig


def d02_air_diag():
    """附件 1 的数据质量诊断：相轨、残差分布、自相关。"""
    t, T, C = load_air()
    resT, resC = T - trend(T), C - trend(C)

    fig, axes = plt.subplots(1, 3, figsize=(W, 2.45), layout="constrained")

    ax = axes[0]
    style(ax, grid="both")
    sc = ax.scatter(C, T, c=t / 3600.0, s=7, cmap=SEQ, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.03)
    cb.set_label("时间 / h", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    r = float(np.corrcoef(C, T)[0, 1])
    ax.text(0.04, 0.94, "$r=%.4f$" % r, transform=ax.transAxes, fontsize=7.2,
            va="top")
    ax.set_xlabel("水分浓度 $C_\\infty$ / (kg/kg)")
    ax.set_ylabel("烘房温度 $T_\\infty$ / $^\\circ$C")
    ax.set_title("(a) 温度–湿度相轨（$n=241$）", loc="left")

    ax = axes[1]
    style(ax)
    for res, col, lab in ((resT, C_MAIN, "温度残差"), (resC, C_TEAL,
                                                       "湿度残差")):
        z = (res - res.mean()) / res.std()
        ax.hist(z, bins=26, histtype="step", lw=1.0, color=col, label=lab)
    xs = np.linspace(-3.4, 3.4, 200)
    ax.plot(xs, len(resT) * 0.30 * np.exp(-xs ** 2 / 2), "--", color=C_BASE,
            lw=0.9, label="标准正态")
    ax.set_xlabel("标准化残差 / $\\sigma$")
    ax.set_ylabel("频数")
    ax.legend(fontsize=7.0, loc="upper right")
    ax.set_title("(b) 残差分布", loc="left")

    ax = axes[2]
    style(ax)
    for res, col, lab in ((resT, C_MAIN, "温度"), (resC, C_TEAL, "湿度")):
        x = res - res.mean()
        ac = [float(np.dot(x[:len(x) - k], x[k:]) / np.dot(x, x))
              for k in range(11)]
        ax.plot(range(11), ac, "-o", ms=3.0, lw=1.0, color=col, label=lab)
    ax.axhline(0, color=C_BASE, lw=0.7)
    ax.axhspan(-1.96 / np.sqrt(len(resT)), 1.96 / np.sqrt(len(resT)),
               color="#D8D8D8", alpha=0.6, zorder=0)
    ax.set_xlabel("滞后阶数 $k$")
    ax.set_ylabel("自相关")
    ax.legend(fontsize=7.0)
    ax.text(0.97, 0.06, "灰带为 95% 白噪声界", transform=ax.transAxes,
            fontsize=6.8, ha="right", color=C_BASE)
    ax.set_title("(c) 残差自相关", loc="left")
    ax.text(0.03, 0.94, "滞后 1 为负 → 高频交替型", transform=ax.transAxes,
            fontsize=6.6, va="top", color=C_BASE)
    return fig


def d03_radius():
    """附件 2：半径历程、收缩速率与收缩完成度。"""
    t, R = load_radius()
    d = np.loadtxt(FIGDATA / "radius_pchip.csv", delimiter=",", skiprows=1)
    th, Rh, dR = d[:, 0] / 3600.0, d[:, 1], d[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(W, 2.5), layout="constrained")

    ax = axes[0]
    style(ax)
    ax.plot(th, Rh, "-", color=C_MAIN, lw=1.4, label="PCHIP 保单调插值")
    ax.plot(t / 3600.0, R, "o", ms=2.2, color=C_INK, label="附件 2 实测点（$n=145$）")
    ax.axhline(1.1980, color=C_CORAL, lw=0.8, ls="-.")
    ax.text(70, 1.22, "终值 1.198 cm", color=C_CORAL, fontsize=6.8, ha="right")
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("半径 $R$ / cm")
    ax.set_xlim(0, 72)
    ax.legend(fontsize=6.8, loc="upper right")
    ax.set_title("(a) 半径收缩历程", loc="left")

    ax = axes[1]
    style(ax)
    ax.plot(th, dR * 3600.0, "-", color=C_ORANGE, lw=1.3)
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("收缩速率 $dR/dt$ / (cm/h)")
    ax.set_xlim(0, 24)
    ax.axhline(0, color=C_BASE, lw=0.6)
    i = int(np.argmin(dR))
    ax.plot([th[i]], [dR[i] * 3600.0], "o", ms=3.4, color=C_CORAL)
    ax.annotate("峰值 $%.2f$ cm/h\n$t=%.1f$ h" % (dR[i] * 3600.0, th[i]),
                xy=(th[i], dR[i] * 3600.0), xytext=(th[i] + 5.0,
                                                    dR[i] * 3600.0 * 0.72),
                fontsize=6.8, color=C_CORAL,
                arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=0.7))
    ax.set_title("(b) 收缩速率 $dR/dt$", loc="left")

    ax = axes[2]
    style(ax)
    done = 100.0 * (R[0] - Rh) / (R[0] - R[-1])
    ax.plot(th, done, "-", color=C_TEAL, lw=1.4)
    for lev, col in ((50, C_ORANGE), (99, C_PURPLE)):
        k = int(np.argmax(done >= lev))
        ax.plot([th[k]], [lev], "o", ms=3.4, color=col)
        ax.annotate("%d%% 完成于 %.1f h" % (lev, th[k]),
                    xy=(th[k], lev), xytext=(th[k] + 1.2, lev - 5),
                    fontsize=6.8, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.7))
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("收缩完成度 / %")
    ax.set_xlim(0, 30)
    ax.set_ylim(-5, 108)
    ax.set_title("(c) 收缩高度前置", loc="left")
    return fig


def d04_radius_vs_moisture():
    """半径与含水率的对应关系：二者并不同步。"""
    e = load_npz("result4_data.npz")
    t = e["times"] / 3600.0
    R = e["radius"] * 100.0
    cen = e["conc21"][:, 0]

    fig = plt.figure(figsize=(W, 2.7), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], hspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    style(ax, grid="both")
    sc = ax.scatter(R, cen, c=t, s=5, cmap=SEQ, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.03)
    cb.set_label("时间 / h", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel("半径 $R$ / cm")
    ax.set_ylabel("中心含水率 $C(0,t)$ / (kg/kg)")
    ax.set_title("(a) 半径–含水率相轨（问题四解）", loc="left")
    ax.text(0.04, 0.94, "收缩在 22 h 内基本结束，\n含水率却持续下降到 50 h",
            transform=ax.transAxes, fontsize=6.9, va="top")

    # 两个量纲不同的量：拆成上下子图共享时间轴，不做双 Y 轴。
    ax1 = fig.add_subplot(gs[0, 1])
    style(ax1)
    ax1.plot(t, R, "-", color=C_MAIN, lw=1.4)
    ax1.set_ylabel("$R$ / cm")
    ax1.set_title("(b) 几何收缩与水分脱除的时间尺度", loc="left")
    ax1.tick_params(labelbottom=False)
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    style(ax2)
    ax2.plot(t, cen, "-", color=C_TEAL, lw=1.4)
    ax2.axhline(0.15, color=C_CORAL, lw=0.8, ls="-.")
    ax2.set_ylabel("$C(0,t)$ / (kg/kg)")
    ax2.set_xlabel("时间 / h")
    ax2.set_xlim(0, t[-1])
    for a in (ax1, ax2):
        a.axvspan(0, 22, color="#EEF3F9", zorder=0)
    ax1.text(21, R.min() + 0.02, "收缩基本结束", fontsize=6.8, ha="right",
             color=C_BASE)
    ax2.text(23, 2.30, "水分仍持续脱除", fontsize=6.8, ha="left", color=C_BASE)
    return fig


# ======================================================================
# 三、数据图：问题一
# ======================================================================
def _p1():
    return load_npz("p1_full.npz")


def d05_p1_fields():
    """问题一温度场与水分的时空演化（热力图）。"""
    d = _p1()
    tt = d["times"] / 60.0
    r = d["r_out_cm"]
    T = d["temp_out"][1:].T
    C = d["conc_out"][1:].T

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7), layout="constrained")
    for ax, Z, cmap, lab, title in (
            (axes[0], T, SEQ, "温度 / $^\\circ$C", "(a) 温度场 $T(r,t)$"),
            (axes[1], C, DIV, "含水率 / (kg/kg)", "(b) 水分浓度场 $C(r,t)$")):
        im = ax.pcolormesh(tt, r, Z, cmap=cmap, shading="auto", rasterized=True)
        cb = fig.colorbar(im, ax=ax, pad=0.03)
        cb.set_label(lab, fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
        ax.set_xlabel("时间 / min")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("到中心距离 / cm")
    axes[0].axhline(2.0 * 0.75, color="white", lw=0.6, ls=":")
    axes[1].annotate("水分变化仅在最外侧 3–5 mm",
                     xy=(22, 1.52), xytext=(14.0, 0.85), fontsize=6.8,
                     color=C_INK,
                     arrowprops=dict(arrowstyle="->", color=C_INK, lw=0.7))
    return fig


def d06_p1_profiles():
    """问题一各报数时刻的径向剖面。"""
    d = _p1()
    r = d["r_out_cm"]
    idx = d["snap_idx"]
    ts = [100, 300, 600, 900, 1200, 1500, 1800]
    ls = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)),
          (0, (1, 2))]
    mk = ["o", "s", "^", "v", "D", "P", "X"]
    cols = [FS.identity_color(i) for i in range(len(ts))]

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.75), layout="constrained")
    for ax, key, lab, title in (
            (axes[0], "temp_out", "温度 / $^\\circ$C", "(a) 温度剖面"),
            (axes[1], "conc_out", "含水率 / (kg/kg)", "(b) 水分浓度剖面")):
        style(ax)
        for k, (i, tt_) in enumerate(zip(idx, ts)):
            ax.plot(r, d[key][i], linestyle=ls[k], marker=mk[k], ms=3.0,
                    lw=1.1, color=cols[k], label="%d s" % tt_)
        ax.set_xlabel("到中心距离 / cm")
        ax.set_ylabel(lab)
        ax.set_xlim(0, 2.0)
        ax.set_title(title, loc="left")
    axes[1].legend(fontsize=6.6, ncol=2, loc="lower left", handlelength=2.2)
    axes[0].annotate("中心 30 min 内完全未变", xy=(0.05, 2.55), xytext=(0.35, 2.44),
                     fontsize=6.8, color=C_CORAL,
                     arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=0.7))
    axes[0].annotate("径向温差 3.21 $^\\circ$C", xy=(2.0, 36.79),
                     xytext=(1.05, 35.3), fontsize=6.8, color=C_MAIN,
                     arrowprops=dict(arrowstyle="->", color=C_MAIN, lw=0.7))
    return fig


def d07_p1_history():
    """问题一中心/表面时间历程与径向温差。"""
    d = _p1()
    # 守恒量序列含 t=0 的初值，时间轴相应补一个 0
    # 温度/水分历史去掉 t=0 的初始行后再画，时间轴与之等长
    t = d["times"] / 60.0
    T = d["temp_out"][1:]
    C = d["conc_out"][1:]
    tair, Tair, Cair = load_air()

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), layout="constrained")
    ax = axes[0]
    style(ax)
    ax.plot(t, T[:, 0], "-", color=C_MAIN, lw=1.4, label="中心 $T(0,t)$")
    ax.plot(t, T[:, 10], "--", color=C_ORANGE, lw=1.3, label="中间 $T(1.0,t)$")
    ax.plot(t, T[:, 20], "-.", color=C_TEAL, lw=1.4, label="表面 $T(R,t)$")
    ax.plot(tair / 60.0, Tair, ":", color=C_BASE, lw=1.2,
            label="烘房 $T_\\infty(t)$")
    ax.set_xlabel("时间 / min")
    ax.set_ylabel("温度 / $^\\circ$C")
    ax.legend(fontsize=6.8, loc="lower right")
    ax.set_title("(a) 温度：由表及里升温", loc="left")
    ax.set_xlim(0, 30)

    ax = axes[1]
    style(ax)
    ax.plot(t, C[:, 0], "-", color=C_MAIN, lw=1.4, label="中心 $C(0,t)$")
    ax.plot(t, C[:, 10], "--", color=C_ORANGE, lw=1.3, label="中间 $C(1.0,t)$")
    ax.plot(t, C[:, 20], "-.", color=C_TEAL, lw=1.4, label="表面 $C(R,t)$")
    ax.plot(tair / 60.0, Cair, ":", color=C_BASE, lw=1.2,
            label="烘房 $C_\\infty(t)$")
    ax.set_xlabel("时间 / min")
    ax.set_ylabel("含水率 $C$ / (kg/kg)")
    ax.set_xlim(0, 30)
    ax.legend(fontsize=6.8, loc="center right")
    ax.set_title("(b) 水分：表面先降、中心不动", loc="left")
    ax.annotate("$\\Delta C$ 仅 $0.0017$", xy=(24, 2.5495), xytext=(13.5, 2.36),
                fontsize=6.8, color=C_MAIN,
                arrowprops=dict(arrowstyle="->", color=C_MAIN, lw=0.7))
    return fig


def d08_p1_scales():
    """特征扩散长度对比：为什么温度铺满截面而水分只动外层。"""
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), layout="constrained")
    d = _p1()
    t = np.linspace(1, 1800, 400)
    alpha = float(d["alpha"][0])
    D = float(d["d_c0"][0])
    lt = np.sqrt(alpha * t) * 1000.0
    lc = np.sqrt(D * t) * 1000.0

    ax = axes[0]
    style(ax)
    ax.plot(t / 60.0, lt, "-", color=C_ORANGE, lw=1.5,
            label="热量 $\\sqrt{\\alpha t}$")
    ax.plot(t / 60.0, lc, "--", color=C_TEAL, lw=1.5,
            label="水分 $\\sqrt{Dt}$")
    ax.axhline(20.0, color=C_CORAL, lw=1.0, ls="-.")
    ax.text(29, 20.8, "药材半径 $R=20$ mm", fontsize=6.8, color=C_CORAL,
            ha="right")
    ax.set_xlabel("时间 / min")
    ax.set_ylabel("特征扩散长度 / mm")
    ax.legend(fontsize=7.0, loc="center left")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 24)
    ax.set_title("(a) 两个特征长度相差一个量级", loc="left")

    ax = axes[1]
    style(ax)
    ratio = (alpha / D)
    ax.bar([0, 1, 2], [alpha, D, alpha - D],
           color=[C_ORANGE, C_TEAL, C_BASE], width=0.5)
    ax.set_yscale("log")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["$\\alpha$\n热量", "$D(C_0)$\n水分", "$\\alpha-D$\n差值"],
                       fontsize=7.0)
    ax.set_ylabel("扩散系数 / (m$^2$/s)")
    for x, v in zip([0, 1, 2], [alpha, D, alpha - D]):
        ax.text(x, v * 1.35, "%.3e" % v, ha="center", fontsize=6.7,
                color=C_INK)
    ax.set_ylim(1e-9, 2e-6)
    ax.text(0.03, 0.94, "$\\alpha/D=%.1f$ 倍" % ratio, transform=ax.transAxes,
            fontsize=7.0, va="top")
    ax.set_title("(b) 两个扩散系数相差 34 倍", loc="left")
    return fig


def d09_p1_conservation():
    """守恒检验：水分总量曲线与"减少量 − 表面流出量"残差。"""
    d = _p1()
    # 守恒量序列含 t=0 的初值，时间轴相应补一个 0
    t = np.concatenate([[0.0], d["times"]]) / 60.0
    q = d["q_volume"]
    flux = d["surf_flux"]
    area = 2.0 * np.pi * 0.02 * 0.25
    dt = float(d["dt"][0])
    cum = np.concatenate([[0.0], np.cumsum(flux[1:]) * dt]) * area

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), layout="constrained")
    ax = axes[0]
    style(ax)
    ax.plot(t, q, "-", color=C_MAIN, lw=1.5, label="模型守恒量 $Q=\\int C\\,dV$")
    ax.plot(t, q[0] - cum, "--", color=C_ORANGE, lw=1.3,
            label="表面累计流出推算（$Q_0-\\int A\\,j\\,dt$）")
    ax.set_xlabel("时间 / min")
    ax.set_ylabel("$Q$ / (m$^3\\cdot$kg/kg)")
    ax.legend(fontsize=6.7, loc="lower left")
    ax.set_title("(a) 两条曲线重合", loc="left")
    ax.set_xlim(0, 30)

    ax = axes[1]
    style(ax)
    resid = (q[0] - q) - cum
    ax.plot(t, np.abs(resid), "-", color=C_PURPLE, lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("时间 / min")
    ax.set_ylabel("$|\\Delta Q-\\int Aj\\,dt|$ / (m$^3\\cdot$kg/kg)")
    ax.set_xlim(0, 30)
    resid14 = float(d["residual"][0]) * 1e14
    ax.text(0.03, 0.10,
            "末态相对残差 $%.1f\\times10^{-14}$\n（机器精度）" % resid14,
            transform=ax.transAxes, fontsize=7.0, va="bottom")
    ax.set_title("(b) 离散守恒残差", loc="left")
    return fig


# ======================================================================
# 四、数据图：问题二
# ======================================================================
def _p23():
    return load_npz("result23_data.npz")


def d10_p2_profiles():
    """问题二 3 h 内温度/含水率的径向剖面族。"""
    d = _p23()
    r = np.arange(21) * 0.1
    hours = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ls = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
    mk = ["o", "s", "^", "v", "D", "P"]
    cols = [FS.identity_color(i) for i in range(6)]

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.75), layout="constrained")
    for ax, key, lab, title in (
            (axes[0], "temp21", "温度 / $^\\circ$C", "(a) 温度剖面"),
            (axes[1], "conc21", "含水率 / (kg/kg)", "(b) 水分浓度剖面")):
        style(ax)
        for k, h in enumerate(hours):
            i = int(round(h * 3600 / 5.0))
            ax.plot(r, d[key][i], linestyle=ls[k], marker=mk[k], ms=3.0,
                    lw=1.1, color=cols[k], label="%.1f h" % h)
        ax.set_xlabel("到中心距离 / cm")
        ax.set_ylabel(lab)
        ax.set_xlim(0, 2.0)
        ax.set_title(title, loc="left")
    axes[1].legend(fontsize=6.5, ncol=2, loc="lower left")
    axes[0].annotate("3 h 时径向温差仅 0.12 $^\\circ$C", xy=(2.0, 49.97),
                     xytext=(0.75, 47.6), fontsize=6.8, color=C_MAIN,
                     arrowprops=dict(arrowstyle="->", color=C_MAIN, lw=0.7))
    return fig


def d11_p2_fields():
    """问题二 3 h 内温度与含水率的时空热力图。"""
    d = _p23()
    n = int(3 * 3600 / 5.0) + 1
    t = d["times"][:n] / 3600.0
    r = np.arange(21) * 0.1
    T = d["temp21"][:n].T
    C = d["conc21"][:n].T

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7), layout="constrained")
    for ax, Z, cmap, lab, title in (
            (axes[0], T, SEQ, "温度 / $^\\circ$C", "(a) 温度场 $T(r,t)$"),
            (axes[1], C, DIV, "含水率 / (kg/kg)", "(b) 水分浓度场 $C(r,t)$")):
        im = ax.pcolormesh(t, r, Z, cmap=cmap, shading="auto", rasterized=True)
        cb = fig.colorbar(im, ax=ax, pad=0.03)
        cb.set_label(lab, fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
        ax.set_xlabel("时间 / h")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("到中心距离 / cm")
    ref = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    for ax, Z in ((axes[0], T), (axes[1], C)):
        for h in ref:
            ax.axvline(h, color="white", lw=0.5, ls=":")
    return fig


def d12_p2_history():
    """问题二中心/表面历程与径向温差。"""
    d = _p23()
    d = {k: d[k] for k in d.files}
    t = d["times"] / 3600.0
    n = int(3 * 3600 / 5.0) + 1
    T, C = d["temp21"][:n], d["conc21"][:n]
    tair, Tair, Cair = load_air()

    fig, axes = plt.subplots(1, 3, figsize=(W, 2.5), layout="constrained")
    ax = axes[0]
    style(ax)
    ax.plot(t[:n], T[:, 0], "-", color=C_MAIN, lw=1.4, label="中心")
    ax.plot(t[:n], T[:, 20], "-.", color=C_ORANGE, lw=1.4, label="表面")
    ax.plot(tair / 3600.0, Tair, ":", color=C_BASE, lw=1.2, label="烘房")
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("温度 / $^\\circ$C")
    ax.set_xlim(0, 3)
    ax.legend(fontsize=6.8, loc="lower right")
    ax.set_title("(a) 温度快速趋于均匀", loc="left")

    ax = axes[1]
    style(ax)
    ax.plot(t[:n], C[:, 0], "-", color=C_MAIN, lw=1.4, label="中心")
    ax.plot(t[:n], C[:, 10], "--", color=C_TEAL, lw=1.3, label="1.0 cm")
    ax.plot(t[:n], C[:, 20], "-.", color=C_ORANGE, lw=1.4, label="表面")
    ax.plot(tair / 3600.0, Cair, ":", color=C_BASE, lw=1.2, label="烘房")
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("含水率 / (kg/kg)")
    ax.set_xlim(0, 3)
    ax.legend(fontsize=6.8, loc="center left")
    ax.set_title("(b) 含水率由表及里依次下降", loc="left")

    ax = axes[2]
    style(ax)
    ax.plot(t[:n], T[:, 20] - T[:, 0], "-", color=C_ORANGE, lw=1.4)
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("径向温差 $T(R)-T(0)$ / $^\\circ$C")
    ax.set_xlim(0, 3)
    ax.set_title("(c) 径向温差迅速衰减", loc="left")
    ax.annotate("3 h 时 0.12 $^\\circ$C", xy=(3.0, T[n - 1, 20] - T[n - 1, 0]),
                xytext=(1.35, 1.05), fontsize=6.8, color=C_ORANGE,
                arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=0.7))
    return fig


def d13_props():
    """附录 3 / 附录 4 的物性随含水率的变化。"""
    d = load_npz("props_curves.npz")
    c = d["c"]
    fig, axes = plt.subplots(2, 2, figsize=(W, 4.3), layout="constrained")
    specs = [("rho3", "rho4", "$\\rho$ / (kg/m$^3$)", "(a) 体积密度", "linear"),
             ("cp3", "cp4", "$c_p$ / (J/(kg$\\cdot$K))", "(b) 比热容", "linear"),
             ("k3", "k4", "$k$ / (W/(m$\\cdot$K))", "(c) 热传导系数", "linear"),
             ("d3", "d4", "$D$ / (m$^2$/s)", "(d) 水分扩散系数", "log")]
    for ax, (k3, k4, lab, title, sc) in zip(axes.ravel(), specs):
        style(ax)
        ax.plot(c, d[k3], "-", color=C_MAIN, lw=1.5, label="附录 3（问题二、三）")
        ax.plot(c, d[k4], "--", color=C_ORANGE, lw=1.5, label="附录 4（问题四）")
        ax.set_ylabel(lab)
        ax.set_xlabel("含水率 $C$ / (kg/kg)")
        ax.set_title(title, loc="left")
        ax.set_yscale(sc)
        ax.axvline(2.55, color=C_CORAL, lw=0.8, ls="-.")
        ax.axvline(0.15, color=C_PURPLE, lw=0.8, ls="-.")
    axes[0, 0].legend(fontsize=6.7, loc="upper left")
    # 竖直参考线的含义统一在下方说明，避免旋转文字压住曲线
    axes[1, 1].text(0.02, 0.06, "竖直点线：左 $C=0.15$（判据），右 $C=2.55$（初值）",
                    transform=axes[1, 1].transAxes, fontsize=6.1,
                    color=C_BASE, va="bottom")
    return fig


def d14_props_D():
    """扩散系数 D 的温度依赖与含水率依赖（对数坐标）。"""
    d = load_npz("props_curves.npz")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), layout="constrained")
    ax = axes[0]
    style(ax)
    for i, cc in enumerate(d["cs"]):
        ax.semilogy(d["temps"], d["dd3"][i], "-", lw=1.3,
                    color=FS.identity_color(i), label="$C=%.2f$" % cc)
    ax.set_xlabel("药材温度 / $^\\circ$C")
    ax.set_ylabel("$D$ / (m$^2$/s)")
    ax.legend(fontsize=6.8, title="附录 3", title_fontsize=6.8)
    ax.set_title("(a) $D$ 对温度的依赖（弱）", loc="left")

    ax = axes[1]
    style(ax)
    ax.semilogy(d["c"], d["d3"], "-", color=C_MAIN, lw=1.5, label="附录 3")
    ax.semilogy(d["c"], d["d4"], "--", color=C_ORANGE, lw=1.5, label="附录 4")
    ax.set_xlabel("含水率 $C$ / (kg/kg)")
    ax.set_ylabel("$D$ / (m$^2$/s)")
    ax.legend(fontsize=6.9)
    ax.set_title("(b) $D$ 对含水率的依赖（跨两个量级）", loc="left",
                 fontsize=8.6)
    ax.annotate("$C:2.55\\to0.15$ 时 $D$ 降 2 个量级",
                xy=(0.40, d["d3"][np.argmin(abs(d["c"] - 0.40))]),
                xytext=(0.62, 3e-10), fontsize=6.8, color=C_INK,
                arrowprops=dict(arrowstyle="->", color=C_INK, lw=0.7))
    return fig


# ======================================================================
# 五、数据图：问题三
# ======================================================================
def d15_p3_curve():
    """问题三干燥曲线与三段结构。"""
    d = _p23()
    t = d["times"] / 3600.0
    cen = d["conc21"][:, 0]
    sur = d["conc_surf"]
    dry = float(d["dry_time"][0]) / 3600.0

    fig, ax = plt.subplots(figsize=(W, 3.0), layout="constrained")
    style(ax)
    ax.axvspan(0, 12, color="#EFF4F9", zorder=0)
    ax.axvspan(12, 36, color="#E3ECF5", zorder=0)
    ax.axvspan(36, dry, color="#D6E4F0", zorder=0)
    ax.plot(t, cen, "-", color=C_MAIN, lw=1.6, label="中心 $C(0,t)$")
    ax.plot(t, sur, "--", color=C_ORANGE, lw=1.5, label="表面 $C(R,t)$")
    ax.axhline(0.15, color=C_CORAL, lw=1.2, ls="-.",
               label="烘干判据 0.15 kg/kg")
    ax.plot([dry], [cen[-1]], "o", color=C_CORAL, ms=4)
    ax.annotate("中心达标 $t=%.2f$ h\n（= %.3f 天）" % (dry, dry / 24),
                xy=(dry, cen[-1]), xytext=(dry - 27, 0.50), fontsize=7.2,
                color=C_CORAL, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=0.8))
    for x, txt in ((6, "快速干燥段\n(0–12 h)"), (24, "降速干燥段\n(12–36 h)"),
                   (46, "极慢段\n(36–57 h)")):
        ax.text(x, 2.34, txt, fontsize=6.8, color=C_BASE, ha="center")
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("含水率 $C$ / (kg/kg)")
    ax.set_xlim(0, dry + 2)
    ax.set_ylim(0, 2.7)
    # 图例移到坐标框上方，把框内空间完全让给放大子图
    ax.legend(fontsize=6.9, loc="lower left", ncol=3, frameon=False,
              mode="expand", bbox_to_anchor=(0.0, 1.005, 1.0, 0.10))

    # 阶段转换：附件 1 只到 4 h，之后按恒温恒湿工况外推
    ax.axvline(4.0, color=C_BASE, lw=0.8, ls=":")
    ax.text(4.4, 1.72, "4 h：附件 1 结束\n转恒温 50 $^\\circ$C", fontsize=6.6,
            color=C_BASE, va="center")

    # 最后 12 h 的放大：中心含水率只降了很少一点，是全流程最不经济的阶段
    axin = ax.inset_axes([0.52, 0.50, 0.35, 0.28])
    m = t >= 45.0
    axin.plot(t[m], cen[m], "-", color=C_MAIN, lw=1.2)
    axin.axhline(0.15, color=C_CORAL, lw=0.9, ls="-.")
    axin.axvline(dry, color=C_BASE, lw=0.7, ls=":")
    axin.set_ylim(0.142, 0.192)
    axin.set_xlim(45, dry + 0.6)
    axin.tick_params(labelsize=5.6, length=2, pad=1.4)
    for s in ("top", "right"):
        axin.spines[s].set_visible(False)
    axin.set_title("末段放大（45 h 后仅降 %.3f）"
                   % (cen[m][0] - cen[-1]), fontsize=5.8, pad=2.0, loc="left")
    fig.tight_layout()
    return fig


def d16_p3_family():
    """问题三各径向位置的含水率历程族。"""
    d = _p23()
    t = d["times"] / 3600.0
    pos = np.arange(21) * 0.1
    # 用正文报数的 5 个位置，颜色身份一一对应（超过 6 条会与身份色环回冲突）
    sel = [0, 5, 10, 15, 20]
    fig, ax = plt.subplots(figsize=(W, 3.0), layout="constrained")
    style(ax)
    for i, j in enumerate(sel):
        ax.plot(t, d["conc21"][:, j], "-", lw=1.3,
                color=FS.identity_color(i), label="$r=%.1f$ cm" % pos[j])
    ax.axhline(0.15, color=C_CORAL, lw=1.1, ls="-.")
    ax.text(2, 0.19, "判据 0.15", color=C_CORAL, fontsize=6.8)
    dry = float(d["dry_time"][0]) / 3600.0
    ax.axvline(dry, color=C_BASE, lw=0.8, ls=":")
    ax.text(dry - 1.2, 0.36, "中心达标 %.1f h" % dry, fontsize=6.8, ha="right",
            color=C_BASE)
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("含水率 $C$ / (kg/kg)")
    ax.set_xlim(0, dry + 1)
    ax.set_ylim(0, 2.9)
    ax.legend(fontsize=6.8, ncol=3, loc="upper right", handlelength=1.8,
              columnspacing=1.2)
    return fig


def d17_p3_field():
    """问题三全过程的时空热力图（对数刻度以兼顾两个量级）。"""
    d = _p23()
    t = d["times"] / 3600.0
    r = np.arange(21) * 0.1
    C = d["conc21"].T

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.8), layout="constrained")
    ax = axes[0]
    im = ax.pcolormesh(t, r, C, cmap=SEQ, shading="auto", rasterized=True,
                       vmin=0.05, vmax=2.55)
    cb = fig.colorbar(im, ax=ax, pad=0.03)
    cb.set_label("含水率 / (kg/kg)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("到中心距离 / cm")
    ax.set_title("(a) 含水率场（线性刻度）", loc="left")
    dry = float(d["dry_time"][0]) / 3600.0
    ax.axvline(dry, color=C_CORAL, lw=1.0, ls="--")
    ax.text(dry - 1, 1.92, "%.1f h" % dry, color=C_CORAL, fontsize=6.8,
            ha="right")

    ax = axes[1]
    Z = np.log10(np.maximum(C, 1e-3))
    im = ax.pcolormesh(t, r, Z, cmap=DIV, shading="auto", rasterized=True,
                       vmin=np.log10(0.05), vmax=np.log10(2.55))
    cb = fig.colorbar(im, ax=ax, pad=0.03)
    cb.set_label("$\\lg(C/(\\mathrm{kg/kg}))$", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.contour(t[::40], r, C[:, ::40], levels=[0.15],
               colors=[C_CORAL], linewidths=1.0, linestyles="--")
    ax.set_xlabel("时间 / h")
    ax.set_title("(b) 对数刻度（含 0.15 等值线）", loc="left")
    return fig


def d18_p3_rate():
    """干燥速率：时间历程与速率–含水率曲线。"""
    d = _p23()
    t = d["times"] / 3600.0
    cen = d["conc21"][:, 0]
    rate = -np.gradient(cen, t)
    dry = float(d["dry_time"][0]) / 3600.0

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), layout="constrained")
    ax = axes[0]
    style(ax)
    ax.plot(t, rate, "-", color=C_MAIN, lw=1.4)
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("中心干燥速率 $-dC/dt$ / (1/h)")
    ax.set_xlim(0, dry)
    ax.set_title("(a) 干燥速率的时间历程", loc="left")
    imax = int(np.argmax(rate))
    ax.plot([t[imax]], [rate[imax]], "o", ms=3.6, color=C_CORAL)
    ax.annotate("峰值 %.3f /h（$t=%.1f$ h）" % (rate[imax], t[imax]),
                xy=(t[imax], rate[imax]), xytext=(t[imax] + 4, rate[imax] * 0.93),
                fontsize=6.8, color=C_CORAL,
                arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=0.7))
    ax.axvspan(36, dry, color="#F2F6FA", zorder=0)
    ax.text(46, rate.max() * 0.42, "36 h 后速率趋零\n$37\\%$ 的时间只换来 $0.036$",
            fontsize=6.8, color=C_BASE, ha="center")

    ax = axes[1]
    style(ax)
    ax.plot(cen, rate, "-", color=C_MAIN, lw=1.4)
    ax.invert_xaxis()
    ax.set_xlabel("中心含水率 $C(0,t)$ / (kg/kg)")
    ax.set_ylabel("干燥速率 $-dC/dt$ / (1/h)")
    ax.set_title("(b) 干燥速率曲线（对含水率）", loc="left")
    for lev, txt, col in ((2.0, "恒速段", C_TEAL), (0.5, "降速段", C_PURPLE)):
        k = int(np.argmin(abs(cen - lev)))
        ax.plot([cen[k]], [rate[k]], "o", ms=3.4, color=col)
        ax.annotate(txt, xy=(cen[k], rate[k]),
                    xytext=(cen[k] + 0.45, rate[k] + rate.max() * 0.10),
                    fontsize=6.9, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.7))
    return fig


# ======================================================================
# 六、数据图：问题四
# ======================================================================
def d19_p4_shrink():
    """问题四：收缩历程与"含/不含收缩"的烘干时长对照。"""
    e = load_npz("result4_data.npz")
    ns = load_npz("p4_noshrink.npz")
    t, R = load_radius()
    dp = np.loadtxt(FIGDATA / "radius_pchip.csv", delimiter=",", skiprows=1)

    fig, axes = plt.subplots(2, 1, figsize=(W, 4.0), sharex=True,
                             layout="constrained")
    ax = axes[0]
    style(ax)
    ax.plot(dp[:, 0] / 3600.0, dp[:, 1], "-", color=C_MAIN, lw=1.5,
            label="PCHIP 插值 $R(t)$")
    ax.plot(t / 3600.0, R, "o", color=C_INK, ms=2.2,
            label="附件 2 实测点（$n=145$）")
    ax.set_ylabel("半径 / cm")
    ax.set_xlim(0, 60)
    ax.legend(fontsize=6.9, loc="upper right")
    ax.set_title("(a) 药材半径收缩历程", loc="left")
    ax.annotate("22 h 内完成 99% 的收缩", xy=(22, 1.26), xytext=(30, 1.42),
                fontsize=6.9, color=C_MAIN,
                arrowprops=dict(arrowstyle="->", color=C_MAIN, lw=0.7))

    ax = axes[1]
    style(ax)
    t4 = e["times"] / 3600.0
    cen = e["conc21"][:, 0]
    dry4 = float(e["dry_time"][0]) / 3600.0
    tn = ns["times"] / 3600.0
    cn = ns["conc"][:, 0]
    dryn = float(ns["dry_time"][0]) / 3600.0
    ax.plot(tn, cn, "--", color=C_ORANGE, lw=1.5,
            label="不含收缩（同物性对照，%.1f h = %.2f 天）" % (dryn, dryn / 24))
    ax.plot(t4, cen, "-", color=C_MAIN, lw=1.6,
            label="含收缩（本文，%.1f h = %.2f 天）" % (dry4, dry4 / 24))
    ax.axhline(0.15, color=C_CORAL, lw=1.1, ls="-.")
    ax.text(2, 0.19, "判据 0.15 kg/kg", color=C_CORAL, fontsize=6.8)
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("中心含水率 / (kg/kg)")
    ax.set_xlim(0, tn[-1])
    ax.legend(fontsize=6.8, loc="upper right")
    ax.set_title("(b) 忽视收缩的时长为计入收缩的 %.2f 倍" % (dryn / dry4),
                 loc="left")
    return fig


def d20_p4_field():
    """问题四含水率时空场：半径收缩导致外侧位置失去定义。"""
    e = load_npz("result4_data.npz")
    t = e["times"] / 3600.0
    r = np.arange(21) * 0.1
    C = np.ma.masked_invalid(e["conc21"].T)
    dry = float(e["dry_time"][0]) / 3600.0

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), layout="constrained")
    ax = axes[0]
    cmap = SEQ.copy()
    cmap.set_bad("#EFEFEF")
    im = ax.pcolormesh(t, r, C, cmap=cmap, shading="auto", rasterized=True, vmin=0.05, vmax=2.55)
    cb = fig.colorbar(im, ax=ax, pad=0.03)
    cb.set_label("含水率 / (kg/kg)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("到中心距离 / cm")
    ax.axvline(dry, color=C_CORAL, lw=1.0, ls="--")
    ax.text(dry - 1, 1.9, "%.1f h" % dry, color=C_CORAL, fontsize=6.8, ha="right")
    ax.set_title("(a) 固定位置含水率（空白＝已出计算域）", loc="left")

    ax = axes[1]
    style(ax)
    ax.plot(t, e["radius"] * 100.0, "-", color=C_MAIN, lw=1.6,
            label="药材表面 $R(t)$")
    ax.plot([0, 60], [2.0, 2.0], ":", color=C_BASE, lw=1.0,
            label="1.5 / 2.0 cm 位置")
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("半径 / cm")
    ax.set_xlim(0, 60)
    ax.set_ylim(1.0, 2.12)
    ax.legend(fontsize=6.9, loc="lower left")
    for lev, col in ((1.5, C_ORANGE), (2.0, C_CORAL)):
        k = int(np.argmax(e["radius"] * 100.0 <= lev))
        ax.axhline(lev, color=col, lw=0.7, ls="--")
        ax.annotate("%.1f cm 处 %.1f h 后无定义" % (lev, t[k]),
                    xy=(t[k], lev), xytext=(t[k] + 3, lev + 0.03),
                    fontsize=6.7, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.6))
    ax.set_title("(b) 哪些位置会在何时离开计算域", loc="left")
    return fig


def d21_p4_surface():
    """问题四表面/中心含水率与半径–含水率耦合。"""
    e = load_npz("result4_data.npz")
    t = e["times"] / 3600.0
    R = e["radius"] * 100.0
    cen = e["conc21"][:, 0]
    sur = e["conc_surf"]
    dry = float(e["dry_time"][0]) / 3600.0

    fig, axes = plt.subplots(1, 3, figsize=(W, 2.5), layout="constrained")
    ax = axes[0]
    style(ax)
    ax.plot(t, cen, "-", color=C_MAIN, lw=1.4, label="中心")
    ax.plot(t, sur, "--", color=C_ORANGE, lw=1.4, label="表面")
    ax.axhline(0.15, color=C_CORAL, lw=0.9, ls="-.")
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("含水率 / (kg/kg)")
    ax.set_xlim(0, dry)
    ax.legend(fontsize=6.8)
    ax.set_title("(a) 中心与表面历程", loc="left")

    ax = axes[1]
    style(ax)
    dif = sur - cen
    ax.plot(t, dif, "-", color=C_PURPLE, lw=1.4)
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("$C(R)-C(0)$ / (kg/kg)")
    ax.set_xlim(0, dry)
    ax.set_title("(b) 内外含水率之差", loc="left")
    # 表面比中心干，所以差值为负；"最大梯度"应对应绝对值最大处
    k = int(np.argmin(dif))
    ax.annotate("梯度最大 $%.2f$（$t=%.1f$ h）" % (dif[k], t[k]),
                xy=(t[k], dif[k]), xytext=(t[k] + 4, dif[k] * 0.55),
                fontsize=6.7, color=C_PURPLE,
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=0.7))

    ax = axes[2]
    style(ax, grid="both")
    sc = ax.scatter(R, cen, c=t, s=5, cmap=SEQ, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.03)
    cb.set_label("时间 / h", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel("半径 $R$ / cm")
    ax.set_ylabel("中心含水率 / (kg/kg)")
    ax.set_title("(c) 半径–含水率相轨", loc="left")
    return fig


# ======================================================================
# 七、数据图：数值验证与灵敏度
# ======================================================================
def d22_convergence():
    """径向网格数 N 与时间步 Δt 对烘干时长的影响。"""
    from matplotlib.ticker import NullLocator

    g = load_grid_csv()
    ns = [20, 50, 100, 200, 400]
    dts = [2, 5, 10, 20, 60, 120]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), layout="constrained")

    ax = axes[0]
    style(ax, grid="both")
    v = np.array([g[("fick", n, 5.0)] / 3600.0 for n in ns])
    ax.plot(ns, v, "-o", color=C_MAIN, lw=1.4, ms=4, label="本文口径（Fick）")
    vv = np.array([g[("cons", n, 5.0)] / 3600.0 for n in ns])
    ax.plot(ns, vv, "--s", color=C_ORANGE, lw=1.4, ms=4,
            label="把 $\\rho$ 写回水分方程")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.xaxis.set_minor_locator(NullLocator())     # 去掉 log 的次刻度标注，避免与主刻度打架
    ax.axhline(v[-1], color=C_BASE, lw=0.7, ls=":")
    ax.annotate("N=20（题目输出网格）偏低 %.2f%%" % (100 * (v[-1] - v[0]) / v[-1]),
                xy=(20, v[0]), xytext=(30, 58.9), fontsize=6.8,
                color=C_CORAL,
                arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=0.7))
    ax.set_xlabel("径向控制体数 $N$")
    ax.set_ylabel("烘干时长 / h")
    ax.legend(fontsize=6.8, loc="lower right")
    ax.set_title("(a) 网格收敛（$\\Delta t=5$ s）", loc="left")

    ax = axes[1]
    style(ax, grid="both")
    w = np.array([g[("fick", 100, d)] / 3600.0 for d in dts])
    ax.plot(dts, w, "-o", color=C_MAIN, lw=1.4, ms=4)
    ax.set_xscale("log")
    ax.set_xticks(dts)
    ax.set_xticklabels([("%g" % d) for d in dts])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("时间步长 $\\Delta t$ / s")
    ax.set_ylabel("烘干时长 / h")
    ax.set_ylim(w.min() - 0.12, w.max() + 0.12)
    ax.set_title("(b) 时间步收敛（$N=100$）", loc="left")
    ax.annotate("$\\Delta t$ 放大 24 倍\n烘干时长只改 %.2f%%"
                % (100 * (w[-1] - w[0]) / w[0]),
                xy=(120, w[-1]), xytext=(3.4, w.min() - 0.09), fontsize=6.8,
                color=C_INK,
                arrowprops=dict(arrowstyle="->", color=C_INK, lw=0.7))
    return fig


def d23_closures():
    """水分方程三种口径的对照。"""
    g = load_grid_csv()
    base = g[("fick", 100, 5.0)] / 3600.0
    cons = g[("cons", 100, 5.0)] / 3600.0
    fick20 = g[("fick", 20, 5.0)] / 3600.0
    cons20 = g[("cons", 20, 5.0)] / 3600.0

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), layout="constrained")
    ax = axes[0]
    style(ax, grid="x")
    names = ["本文：经典 Fick\n$\\partial_tC=\\nabla\\!\\cdot\\!(D\\nabla C)$",
             "变密度守恒\n$\\partial_t(\\rho C)=\\nabla\\!\\cdot\\!(\\rho D\\nabla C)$"]
    vals = [base, cons]
    y = np.arange(2)
    ax.barh(y, vals, height=0.5, color=[C_MAIN, C_ORANGE])
    for i, v in enumerate(vals):
        ax.text(v + 0.25, i, "%.2f h" % v, va="center", fontsize=7.2)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.9)
    ax.set_xlabel("问题三烘干时长 / h")
    ax.set_xlim(45, 66)
    ax.set_title("(a) 口径不同的绝对影响", loc="left")
    ax.text(0.98, 0.06, "相差 $+%.1f\\%%$" % (100 * (cons / base - 1)),
            transform=ax.transAxes, ha="right", fontsize=7.2, color=C_NEG)

    ax = axes[1]
    style(ax)
    ratio = 100 * (cons / base - 1)
    ratio20 = 100 * (cons20 / fick20 - 1)
    ax.plot([0, 1], [ratio, ratio], "-", color=C_ORANGE, lw=1.6)
    ax.plot([0, 1], [ratio20, ratio20], "--o", color=C_MAIN, lw=1.2, ms=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["$N=100,\\ \\Delta t=5$ s", "$N=20,\\ \\Delta t=5$ s"],
                       fontsize=7.0)
    ax.set_ylabel("变密度口径相对本文 / %")
    ax.set_ylim(0, 9)
    ax.set_title("(b) 口径差与数值参数无关", loc="left")
    ax.text(0.5, (ratio + ratio20) / 2 + 0.45,
            "两组网格给出同一口径差\n$\\Rightarrow$ 差异来自模型口径而非离散误差",
            fontsize=6.9, ha="center", va="bottom", color=C_BASE)
    return fig


def d24_sensitivity():
    """h、h_m、D 的 ±20% 灵敏度。"""
    s, rows = load_sens()
    base = s["base1.00"]

    def rel(key):
        return 100.0 * (s[key] - base) / base

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), layout="constrained")
    ax = axes[0]
    style(ax, grid="x")
    labels = ["$D$", "$h_m$", "$h$"]
    lo = np.array([rel("D0.80"), rel("hm0.80"), rel("h0.80")])
    hi = np.array([rel("D1.20"), rel("hm1.20"), rel("h1.20")])
    y = np.arange(3)
    ax.barh(y, hi, height=0.42, color=C_ORANGE, label="参数 $+20\\%$")
    ax.barh(y, lo, height=0.42, color=C_MAIN, label="参数 $-20\\%$")
    ax.axvline(0, color=C_INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("烘干时长相对变化 / %")
    ax.legend(fontsize=6.8, loc="upper right")
    ax.set_xlim(-26, 44)
    ax.set_title("(a) 三类参数的灵敏度", loc="left")
    for i in range(3):
        if abs(hi[i]) < 0.4 and abs(lo[i]) < 0.4:
            ax.text(-1.2, y[i], "$\\approx0\\%$", va="center", ha="right",
                    fontsize=7.0)

    ax = axes[1]
    style(ax)
    order = ["h0.80", "h1.20", "hm0.80", "hm1.20", "base1.00", "D0.80", "D1.20"]
    vals = [s[k] for k in order]
    cols = [C_MAIN, C_MAIN, C_TEAL, C_TEAL, C_BASE, C_ORANGE, C_ORANGE]
    x = np.arange(len(order))
    ax.bar(x, vals, color=cols, width=0.62)
    ax.axhline(base, color=C_BASE, lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["$h$−", "$h$+", "$h_m$−", "$h_m$+", "基准", "$D$−",
                        "$D$+"], fontsize=6.8)
    ax.set_ylabel("烘干时长 / h")
    ax.set_ylim(40, 76)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.8, "%.1f" % v, ha="center", fontsize=6.6)
    ax.set_title("(b) 绝对烘干时长", loc="left")
    return fig


def d25_dry_mass():
    """用附件 2 实测半径检验干物质守恒。"""
    r = np.arange(21) * 0.001
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), layout="constrained")

    for ax, tag, a, b, note in (
            (axes[0], "result23_data.npz", 650.0, 128.0, "问题三（固定半径，附录 3）"),
            (axes[1], "result4_data.npz", 760.0, 90.0, "问题四（含收缩，附录 4）")):
        d = load_npz(tag)
        t = d["times"] / 3600.0
        C = d["conc21"]
        m = np.empty(len(t))
        for k in range(len(t)):
            row = C[k]
            ok = np.isfinite(row)
            cc, rr = row[ok], r[ok]
            # numpy 2.x 把 trapz 改名为 trapezoid
            y = (a + b * cc) / (1 + cc) * 2 * np.pi * rr
            m[k] = np.trapezoid(y, rr)
            if "4" in tag:
                # 问题四的半径在收缩：最外侧采样点到 r=R(t) 之间还有一个薄壳
                # 没有采样点覆盖。不补这一项，半径每越过一条 0.1 cm 网格线，
                # 积分域就会突跳一次，曲线出现锯齿（纯属采样伪影）。
                Rk = d["radius"][k]
                if Rk > rr[-1]:
                    cs = float(C[k][np.isfinite(C[k])][-1])
                    shell = (a + b * cs) / (1 + cs) * 2 * np.pi * \
                        (rr[-1] + Rk) / 2.0 * (Rk - rr[-1])
                    m[k] += shell
        if "4" in tag:                       # 再做 2 h 滑动平均压掉残余抖动
            w = max(int(2 * 3600 / float(d["times"][1])), 1)
            kk = np.ones(w) / w
            num = np.convolve(m, kk, mode="same")
            den = np.convolve(np.ones_like(m), kk, mode="same")
            m = num / den                    # 端点按有效窗口长度归一，避免边界塌陷
        rel = 100.0 * (m / m[0] - 1.0)
        style(ax)
        ax.plot(t, rel, "-", color=C_MAIN if "23" in tag else C_TEAL, lw=1.5)
        ax.axhline(0, color=C_CORAL, lw=1.0, ls="-.")
        ax.set_xlabel("时间 / h")
        ax.set_ylabel("干物质质量相对初值的偏差 / %")
        ax.set_title(note, loc="left")
        ax.text(0.03, 0.92, "末态 $%+.0f\\%%$" % rel[-1],
                transform=ax.transAxes, fontsize=7.4, va="top",
                color=C_MAIN if "23" in tag else C_TEAL)
    axes[0].text(0.5, 0.55, "题面给定的 $\\rho(C)$ 与\n固定半径假设的固有张力",
                 transform=axes[0].transAxes, fontsize=6.9, ha="center",
                 color=C_BASE)
    return fig


# ======================================================================
# 八、数据图：物性关联式的自洽性（rho 专题）
# ======================================================================
def _rho_model(c, a, beta):
    """线性收缩模型给出的 rho(C) = a(1+C)/(1 + beta*a*C/rho_w)。"""
    return a * (1 + c) / (1 + beta * a * c / 1000.0)


def d26_rho_hypothesis():
    """不同收缩假设下的 rho–C 关系（对应文档图 1）。"""
    R = load_rho()
    c = np.linspace(0, 2.55, 400)
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), layout="constrained")
    for ax, key, lab in ((axes[0], "appendix3", "(a) 附录 3：$\\rho=650+128C$"),
                         (axes[1], "appendix4", "(b) 附录 4：$\\rho=760+90C$")):
        cc = R["cases"][key]
        a, b = cc["a"], cc["b"]
        style(ax)
        ax.plot(c, a + b * c, "-", color=C_MAIN, lw=2.0, label="题目经验式")
        ax.plot(c, _rho_model(c, a, 0.0), "--", color=C_ORANGE, lw=1.3,
                label="无收缩理想式（$\\beta=0$）")
        ax.plot(c, _rho_model(c, a, 1.0), "-.", color=C_TEAL, lw=1.3,
                label="理想收缩式（$\\beta=1$）")
        ax.plot(c, _rho_model(c, a, cc["beta_lsq"]), ":", color=C_PURPLE,
                lw=1.8, label="反演收缩式（$\\beta=%.3f$）" % cc["beta_lsq"])
        ax.axhline(1000, color=C_BASE, lw=0.8, ls=(0, (1, 3)))
        ax.axhline(1400, color=C_BASE, lw=0.8, ls=(0, (1, 3)))
        # 参考密度标签放左侧：右侧要留给"斜率=截距"的反证标注
        ax.text(0.06, 1035, "水的真密度 1000", fontsize=6.5, ha="left",
                color=C_BASE)
        ax.text(0.06, 1435, "干物质真密度 1400", fontsize=6.5, ha="left",
                color=C_BASE)
        if key == "appendix3":
            ax.plot([2.55], [R["naive_rho_C0"]], "*", ms=8, color=C_CORAL,
                    zorder=5)
            ax.annotate("若强行要求斜率=截距：\n$%.0f\\times(1+2.55)=%.0f$ kg/m$^3$\n"
                        "超过任何可能的混合物密度" % (a, R["naive_rho_C0"]),
                        xy=(2.55, R["naive_rho_C0"]),
                        xytext=(1.30, 1290), fontsize=6.6, color=C_CORAL,
                        arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=0.7))
        ax.plot([2.55], [cc["rho_end"]], "o", ms=4, color=C_MAIN, zorder=5)
        ax.set_xlabel("干基含水率 $C$ / (kg/kg)")
        ax.set_ylabel("体积密度 $\\rho$ / (kg/m$^3$)")
        ax.set_xlim(0, 2.6)
        ax.set_ylim(600, 2700)
        ax.set_title(lab, loc="left")
    axes[0].legend(fontsize=6.5, loc="upper left", borderpad=0.3,
                   labelspacing=0.32, handlelength=2.0)
    return fig


def d27_rho_d():
    """表观干基密度 rho_d = rho/(1+C)：它并非常数（对应文档图 2）。"""
    R = load_rho()
    c = np.linspace(0, 2.55, 400)
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.6), layout="constrained")
    style(ax)
    for key, col, ls, lab in (("appendix3", C_MAIN, "-", "附录 3：$650\\to275$"),
                              ("appendix4", C_ORANGE, "--", "附录 4：$760\\to279$")):
        cc = R["cases"][key]
        ax.plot(c, (cc["a"] + cc["b"] * c) / (1 + c), ls, color=col, lw=1.6,
                label=lab)
    ax.axhline(650, color=C_BASE, lw=0.9, ls="-.")
    ax.text(0.06, 668, "若无收缩，$\\rho_d$ 应为常数", fontsize=6.8, color=C_BASE)
    ax.set_xlabel("干基含水率 $C$ / (kg/kg)")
    ax.set_ylabel("$\\rho_d=\\rho/(1+C)$ / (kg/m$^3$)")
    ax.set_xlim(0, 2.55)
    ax.set_ylim(200, 820)
    ax.legend(fontsize=6.8, loc="upper right")
    ax.annotate("单位干物质的体积在 $C_0$ 处只有初始值的 42%",
                xy=(2.55, 275), xytext=(1.05, 320), fontsize=6.8, color=C_INK,
                arrowprops=dict(arrowstyle="->", color=C_INK, lw=0.7))
    return fig


def d28_beta_check():
    """收缩系数反演（最小二乘曲线）与三相体积分数非负性检验。"""
    R = load_rho()
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7), layout="constrained")

    ax = axes[0]
    style(ax)
    cg = np.linspace(0, 2.55, 600)
    betas = np.linspace(0.5, 1.25, 400)
    for key, col, ls, lab in (("appendix3", C_MAIN, "-", "附录 3"),
                              ("appendix4", C_ORANGE, "--", "附录 4")):
        cc = R["cases"][key]
        rmse = [float(np.sqrt(np.mean((_rho_model(cg, cc["a"], bt)
                                       - (cc["a"] + cc["b"] * cg)) ** 2)))
                for bt in betas]
        ax.plot(betas, rmse, ls, color=col, lw=1.5, label=lab)
        ax.plot([cc["beta_lsq"]], [cc["rmse"]], "o", ms=4.2, color=col)
        # 标注挪到曲线不经过的左下角，避免压线
        ax.annotate("$\\beta_{LSQ}=%.3f$（%s）"
                    % (cc["beta_lsq"], "附录 3" if key == "appendix3"
                       else "附录 4"),
                    xy=(cc["beta_lsq"], cc["rmse"]),
                    xytext=(0.53, 9.0 if key == "appendix3" else 1.5),
                    fontsize=6.8, color=col, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.7))
    ax.axvspan(0.7, 1.0, color="#EAF1F8", zorder=0)
    ax.text(0.85, 91, "文献常见区间 $\\beta\\in[0.7,1.0]$", fontsize=6.6,
            ha="center", va="center", color=C_BASE,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
    ax.set_xlabel("收缩系数 $\\beta$")
    ax.set_ylabel("拟合 RMSE / (kg/m$^3$)")
    ax.set_ylim(0, 98)
    ax.legend(fontsize=6.8, loc="lower right")
    ax.set_title("(a) 收缩系数的最小二乘反演", loc="left")

    ax = axes[1]
    style(ax)
    labs, ws, ss, aas = [], [], [], []
    for key, short in (("appendix3", "附录 3"), ("appendix4", "附录 4")):
        for tag, tlab in (("C0", "$C_0=2.55$"), ("dry", "$C=0$")):
            p = R["cases"][key]["phases"][tag]
            labs.append("%s\n%s" % (short, tlab))
            ws.append(100 * p["eps_w"])
            ss.append(100 * p["eps_s"])
            aas.append(100 * p["eps_a"])
    x = np.arange(4)
    b1 = ax.bar(x, ws, color=C_MAIN, label="水 $\\varepsilon_w$", width=0.6)
    b2 = ax.bar(x, ss, bottom=ws, color=C_TEAL, label="固 $\\varepsilon_s$",
                width=0.6)
    b3 = ax.bar(x, aas, bottom=np.array(ws) + np.array(ss), color=C_ORANGE,
                label="气（孔隙）$\\varepsilon_a$", width=0.6)
    for xi, (w_, s_, a_) in enumerate(zip(ws, ss, aas)):
        for val, base_, b in ((w_, 0, b1), (s_, w_, b2), (a_, w_ + s_, b3)):
            if val > 6:
                ax.text(xi, base_ + val / 2, "%.0f%%" % val, ha="center",
                        va="center", fontsize=6.6,
                        color="white" if b is b1 else C_INK)
        ax.text(xi, 103, "✓", ha="center", fontsize=8, color=C_POS)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=6.7)
    ax.set_ylabel("体积分数 / %")
    ax.set_ylim(0, 112)
    ax.legend(fontsize=6.6, loc="lower left", ncol=3,
              bbox_to_anchor=(0.0, -0.34, 1.0, 0.1), mode="expand")
    ax.set_title("(b) 三相体积分数全程非负（$\\rho_s=1400$）", loc="left")
    return fig


def d29_geo_shrink():
    """由经验式推算的几何收缩，与附件 2 实测收缩的对照。"""
    R = load_rho()
    e = load_npz("result4_data.npz")
    c = np.linspace(0.01, 2.55, 400)
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7), layout="constrained")

    ax = axes[0]
    style(ax)
    cc = R["cases"]["appendix3"]
    a, b = cc["a"], cc["b"]
    v = ((1 + c) / (a + b * c)) / ((1 + 2.55) / (a + b * 2.55))
    ax.plot(c, v, "-", color=C_MAIN, lw=1.6, label="体积比 $V/V_0$")
    ax.plot(c, v ** 0.5, "--", color=C_ORANGE, lw=1.5,
            label="线收缩比 $L/L_0$")
    ax.plot(c, 2.0 * v ** 0.5, "-.", color=C_TEAL, lw=1.5,
            label="半径 $R$ / cm")
    for lev in (2.0, 1.0, 0.5, 0.15):
        k = int(np.argmin(abs(c - lev)))
        ax.plot([c[k]], [v[k]], "o", ms=3.2, color=C_MAIN)
        ax.text(c[k] + 0.03, v[k] + 0.012, "%.3f" % v[k], fontsize=6.4,
                color=C_INK)
    ax.axvline(2.0, color=C_BASE, lw=0.7, ls=":")
    ax.annotate("预热段：$C\\geq2.0$\n线收缩 $<5\\%$",
                xy=(2.0, 1.94), xytext=(1.28, 1.46), fontsize=6.6,
                color=C_BASE,
                arrowprops=dict(arrowstyle="->", color=C_BASE, lw=0.7))
    ax.set_xlabel("终点含水率 $C$ / (kg/kg)")
    ax.set_ylabel("相对初始态的比例")
    ax.set_xlim(2.55, 0)
    ax.set_ylim(0.35, 2.05)
    ax.legend(fontsize=6.7, loc="lower left")
    ax.set_title("(a) 由附录 3 经验式推算的收缩（径向等长）", loc="left")

    ax = axes[1]
    style(ax)
    rad = e["radius"] * 100.0
    # 体积加权平均含水率 \bar C(t) = ∫C·r dr / ∫r dr
    # 与正文 §11.5 的口径一致（同一含水率下比较）；预先向量化，避免逐时刻循环。
    r_grid = np.arange(21) * 0.1          # cm，与 result4 的报告网格一致
    ok = e["valid21"].astype(bool)        # (nt, 21)
    idx_last = ok.sum(axis=1) - 1         # 每行最后一个有效节点的下标
    rows = np.arange(len(rad))
    seg = 0.5 * (e["conc21"][:, 1:] * r_grid[1:]
                 + e["conc21"][:, :-1] * r_grid[:-1]) * 0.1
    seg = np.where(ok[:, 1:] & ok[:, :-1], seg, 0.0)
    cum = np.cumsum(seg, axis=1)          # 前 j 段梯形积分之和
    full = np.where(idx_last >= 1,
                    cum[rows, np.maximum(idx_last - 1, 0)], 0.0)
    r_last = r_grid[np.maximum(idx_last, 0)]
    c_last = e["conc21"][rows, np.maximum(idx_last, 0)]
    gap = np.maximum(rad - r_last, 0.0)   # 最后一个节点到表面的余段
    tail = 0.5 * (e["conc_surf"] * rad + c_last * r_last) * gap
    cbar = (full + tail) / (0.5 * rad ** 2)
    a4, b4 = R["cases"]["appendix4"]["a"], R["cases"]["appendix4"]["b"]
    cg = np.linspace(0.05, 2.55, 400)
    vg = ((1 + cg) / (a4 + b4 * cg)) / ((1 + 2.55) / (a4 + b4 * 2.55))
    ax.plot(cbar, rad, "-", color=C_MAIN, lw=1.6, label="附件 2 实测轨迹")
    ax.plot(cg, 2.0 * vg ** 0.5, "--", color=C_ORANGE, lw=1.5,
            label="附录 4 经验式推算")
    ax.invert_xaxis()
    ax.set_xlabel("体积平均含水率 $\\bar C(t)$ / (kg/kg)")
    ax.set_ylabel("半径 / cm")
    ax.set_ylim(1.15, 2.05)
    ax.legend(fontsize=6.8, loc="upper right")
    ax.set_title("(b) 实测收缩 vs 经验式推算", loc="left")
    # 端点差异必须在**同一含水率**下比：用末态中心含水率代入经验式，而不是取 cg 的右端
    c_end = cbar[-1]
    v_end = ((1 + c_end) / (a4 + b4 * c_end)) / \
        ((1 + 2.55) / (a4 + b4 * 2.55))
    r_th = 2.0 * v_end ** 0.5
    dd = 100 * (rad[-1] - r_th) / r_th
    ax.plot([c_end], [r_th], "o", ms=4, color=C_ORANGE)
    ax.annotate("同含水率下\n经验式 $%.2f$ cm vs 实测 $%.2f$ cm\n（$%+.1f\\%%$）"
                % (r_th, rad[-1], dd),
                xy=(c_end, r_th), xytext=(1.55, 1.30), fontsize=6.6,
                color=C_BASE, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_BASE, lw=0.7))
    ax.text(0.03, 0.06, "两条线并不同源：经验式只隐含平均收缩效应",
            transform=ax.transAxes, fontsize=6.8, va="bottom", color=C_BASE)
    return fig


# ======================================================================
# 图目录与入口
# ======================================================================
def d30_threshold():
    """烘干判据阈值的敏感性：阈值取多少，烘干时长变多少。"""
    d = np.loadtxt(FIGDATA / "threshold.csv", delimiter=",", skiprows=1)
    t, c = d[:, 0] / 3600.0, d[:, 1]
    # 扫描只算到中心含水率 0.1266，故阈值下限取 0.13
    thr = np.array([0.25, 0.22, 0.20, 0.18, 0.16, 0.15, 0.14, 0.13])
    got = np.array([t[int(np.argmax(c <= x))] if (c <= x).any() else np.nan
                    for x in thr])

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), layout="constrained")
    ax = axes[0]
    style(ax, grid="both")
    ax.plot(thr, got, "-o", color=C_MAIN, lw=1.5, ms=4)
    base = got[thr == 0.15][0]
    ax.plot([0.15], [base], "o", ms=6.5, mfc="none", mec=C_CORAL, mew=1.4)
    ax.axvline(0.15, color=C_CORAL, lw=0.9, ls="-.")
    ax.annotate("题面判据 0.15 kg/kg\n$t=%.2f$ h（$N=100$ 扫描）" % base,
                xy=(0.15, base), xytext=(0.208, 41.0), fontsize=6.8,
                color=C_CORAL, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=0.7))
    ax.set_xlabel("判据阈值 $C^*$ / (kg/kg)")
    ax.set_ylabel("中心达标所需时间 / h")
    ax.invert_xaxis()
    ax.set_title("(a) 阈值越严，时长增长越快", loc="left")

    ax = axes[1]
    style(ax)
    rel = 100 * (got / base - 1)
    ax.bar(np.arange(len(thr)), rel, width=0.62,
           color=[C_MAIN if x >= 0.15 else C_ORANGE for x in thr])
    ax.axhline(0, color=C_INK, lw=0.8)
    ax.set_xticks(np.arange(len(thr)))
    ax.set_xticklabels(["%.2f" % x for x in thr], fontsize=6.8,
                       rotation=45)
    ax.set_xlabel("判据阈值 $C^*$ / (kg/kg)")
    ax.set_ylabel("相对 0.15 的时长变化 / %")
    ax.set_title("(b) 阈值敏感性的量级", loc="left")
    for i, v in enumerate(rel):
        ax.text(i, v + (1.2 if v >= 0 else -2.6), "%+.0f%%" % v,
                ha="center", fontsize=6.3)
    ax.set_ylim(min(rel) - 7, max(rel) + 7)
    return fig


def d31_grid_order():
    """网格收敛的观测阶：误差对 $N$ 的双对数图 + 一阶参考斜率。"""
    g = load_grid_csv()
    ns = np.array([20, 50, 100, 200, 400], dtype=float)
    v = np.array([g[("fick", int(n), 5.0)] / 3600.0 for n in ns])
    # Richardson 外推（一阶）：用最细两档估计极限值
    t_inf = 2.0 * v[-1] - v[-2]
    err = np.abs(v - t_inf)

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), layout="constrained")
    ax = axes[0]
    style(ax, grid="both")
    ax.loglog(ns, err, "-o", color=C_MAIN, lw=1.5, ms=4, label="本文口径")
    ref = err[0] * (ns / ns[0]) ** -1.0
    ax.loglog(ns, ref, "--", color=C_BASE, lw=1.0,
              label="参考斜率 $O(N^{-1})$")
    obs = np.log(err[-2] / err[-1]) / np.log(ns[-1] / ns[-2])
    ax.set_xlabel("径向控制体数 $N$")
    ax.set_ylabel("$|t(N)-t_\\infty|$ / h")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(int(n)) for n in ns])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.legend(fontsize=6.8, loc="lower left")
    ax.set_title("(a) 误差随网格的衰减（双对数）", loc="left")
    ax.text(0.97, 0.94, "观测阶 $p=%.2f$" % obs, transform=ax.transAxes,
            ha="right", va="top", fontsize=7.0)

    ax = axes[1]
    style(ax)
    ax.plot(ns, v, "-o", color=C_MAIN, lw=1.5, ms=4)
    ax.axhline(t_inf, color=C_ORANGE, lw=1.1, ls="--")
    ax.text(22, t_inf + 0.012, "Richardson 外推 $t_\\infty=%.3f$ h" % t_inf,
            fontsize=6.8, color=C_ORANGE)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(int(n)) for n in ns])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("径向控制体数 $N$")
    ax.set_ylabel("烘干时长 / h")
    ax.set_ylim(v.min() - 0.12, max(t_inf, v.max()) + 0.10)
    ax.set_title("(b) 烘干时长向极限值单调收敛", loc="left")
    ax.annotate("$N=20$（题目输出网格）\n偏低 $%.2f\\%%$"
                % (100 * (t_inf - v[0]) / t_inf),
                xy=(20, v[0]), xytext=(28, v.min() + 0.16), fontsize=6.8,
                color=C_CORAL,
                arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=0.7))
    return fig


def d32_energy():
    """能量自洽性核算：蒸发潜热需求 vs 对流供热能力。"""
    d = _p1()
    t = d["times"] / 60.0
    T = d["temp_out"][1:]
    C = d["conc_out"][1:]
    t_air, Tair, Cair = load_air()
    Ts, Cs = T[:, 20], C[:, 20]
    # 附件 1 是 60 s 采样，模型是 1 s 步长，把边界条件插值到模型时间轴上
    Tinf = np.interp(d["times"], t_air, Tair)
    Cinf = np.interp(d["times"], t_air, Cair)

    A = 2.0 * np.pi * 0.02 * 0.25          # 侧面换热面积 m^2
    RHO, H, HM, LV = 820.0, 25.0, 8.0e-7, 2.26e6
    q_conv = H * (Tinf - Ts) * A                      # 对流供热 W
    rho_d = RHO / (1.0 + Cs)                          # 表面干基密度
    j_w = rho_d * HM * (Cs - Cinf)                    # 表面失水通量 kg/(m^2 s)
    q_lat = LV * j_w * A                              # 潜热需求 W

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), layout="constrained")
    ax = axes[0]
    style(ax)
    ax.plot(t, q_conv, "-", color=C_ORANGE, lw=1.6, label="对流供热 $hA(T_\\infty-T_s)$")
    ax.plot(t, q_lat, "--", color=C_TEAL, lw=1.6,
            label="蒸发潜热需求 $L_v A j_w$")
    ax.set_xlabel("时间 / min")
    ax.set_ylabel("功率 / W")
    ax.set_xlim(0, 30)
    ax.legend(fontsize=6.7, loc="center left")
    ax.set_title("(a) 潜热需求远大于对流供热", loc="left")

    ax = axes[1]
    style(ax)
    ax.plot(t, q_lat / q_conv, "-", color=C_PURPLE, lw=1.6)
    ax.axhline(1.0, color=C_CORAL, lw=1.0, ls="-.")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 1.2e4)
    ax.text(0.6, 1.25, "收支平衡线", fontsize=6.6, color=C_CORAL, ha="left")
    ax.set_xlabel("时间 / min")
    ax.set_ylabel("$L_v A j_w\\,/\\,hA(T_\\infty-T_s)$")
    ax.set_xlim(0, 30)
    ax.set_title("(b) 比值 $\\gg1$：潜热无法自洽", loc="left")
    ax.text(0.03, 0.94,
            "若强行计入潜热，表面温度会被压到\n湿球温度（%.1f $^\\circ$C）以下，"
            "违反热力学第二定律" % 34.69,
            transform=ax.transAxes, fontsize=6.7, va="top")
    return fig


DATA_FIGS = [
    ("d01_air_series", d01_air_series),
    ("d02_air_diag", d02_air_diag),
    ("d03_radius", d03_radius),
    ("d04_radius_vs_moisture", d04_radius_vs_moisture),
    ("d05_p1_fields", d05_p1_fields),
    ("d06_p1_profiles", d06_p1_profiles),
    ("d07_p1_history", d07_p1_history),
    ("d08_p1_scales", d08_p1_scales),
    ("d09_p1_conservation", d09_p1_conservation),
    ("d10_p2_profiles", d10_p2_profiles),
    ("d11_p2_fields", d11_p2_fields),
    ("d12_p2_history", d12_p2_history),
    ("d13_props", d13_props),
    ("d14_props_D", d14_props_D),
    ("d15_p3_curve", d15_p3_curve),
    ("d16_p3_family", d16_p3_family),
    ("d17_p3_field", d17_p3_field),
    ("d18_p3_rate", d18_p3_rate),
    ("d19_p4_shrink", d19_p4_shrink),
    ("d20_p4_field", d20_p4_field),
    ("d21_p4_surface", d21_p4_surface),
    ("d22_convergence", d22_convergence),
    ("d23_closures", d23_closures),
    ("d24_sensitivity", d24_sensitivity),
    ("d25_dry_mass", d25_dry_mass),
    ("d26_rho_hypothesis", d26_rho_hypothesis),
    ("d27_rho_d", d27_rho_d),
    ("d28_beta_check", d28_beta_check),
    ("d29_geo_shrink", d29_geo_shrink),
    ("d30_threshold", d30_threshold),
    ("d31_grid_order", d31_grid_order),
    ("d32_energy", d32_energy),
]

ALL = SCHEMATICS + DATA_FIGS


def main():
    global ONLY
    ap = argparse.ArgumentParser(description="A 题论文插图生成器")
    ap.add_argument("--only", default=None, help="逗号分隔的图 id")
    ap.add_argument("--list", action="store_true", help="列出全部图 id")
    args = ap.parse_args()
    if args.list:
        for name, fn in ALL:
            print("%-24s %s" % (name, (fn.__doc__ or "").strip().split("\n")[0]))
        return
    ONLY = set(x.strip() for x in (args.only or "").split(",") if x.strip())
    FIGDIR.mkdir(parents=True, exist_ok=True)
    print("输出目录: %s" % FIGDIR)
    run(ALL)
    print("\n=== 版面自检汇总（scipilot-figure-skill :: visual_qa）===")
    visual_qa.print_report(AUDIT)


if __name__ == "__main__":
    main()
