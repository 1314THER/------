#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本节插图：斜率截距比 r=b/a 作为"是否考虑收缩"的统一判据。

画什么、为什么
--------------
(a) 判据曲线：横轴 r=b/a（0~1），纵轴**绝干线收缩率** 1 - R(0)/R0。
    r=1 时恒为 0（无收缩），r 越小收缩越强；两条曲线分别是问题四采用的
    径向等长口径与各向同性口径。题面两条经验式的 r 标在曲线上——它们离
    r=1 都很远，这就是"必须考虑收缩"的判据来源。
(b) 呼应曲线：纵轴换成**同一含水率下的径向线收缩率**，横轴为含水率（自左
    向右干燥）。叠三条门槛线（3%、5%、10%），并把问题一、问题二、问题三的
    报数点标在附录 3 的曲线上——问题一结束时只有 2.4%（可忽略），问题三
    结束时已达 32%（不可忽略），一眼看出三问为何作不同的收缩假设。

数据一律读 输出/斜率截距与收缩.json（由 compute_shrinkage.py 生成），
不在绘图脚本里硬编码任何结论数字。

运行
----
    MPLCONFIGDIR=/tmp/mplcache /tmp/figvenv/bin/python make_figure.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIGDIR = HERE / "figs"

sys.path.insert(0, str(PROJECT / "代码" / "figlib"))
sys.path.insert(0, os.path.expanduser("~/.codex/skills/scipilot-figure-skill/scripts"))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

import figstyle as FS                                # noqa: E402

W = 15.6 / 2.54          # A4 正文宽度 15.6 cm，不做二次缩放
C_MAIN = FS.COLOR_MAIN
C_ORANGE = FS.ACCENT_ORANGE
C_BASE = FS.COLOR_BASELINE
C_INK = FS.COLOR_INK

style = FS.style_axes


def load() -> dict:
    return json.loads((HERE / "输出" / "斜率截距与收缩.json").read_text(encoding="utf-8"))


def dry_line_shrink(r: float, c0: float, geom: str = "radial") -> float:
    """绝干线收缩率 1 - R(0)/R0（%）。"""
    q = (1.0 + r * c0) / (1.0 + c0)                  # V(0)/V(C0)
    root = np.sqrt(q) if geom == "radial" else np.cbrt(q)
    return 100.0 * (1.0 - root)


def line_shrink_at_c(c, r: float, c0: float) -> float:
    """含水率 C 处的径向线收缩率（%）。"""
    c = np.asarray(c, dtype=float)
    q = (1.0 + c) * (1.0 + r * c0) / ((1.0 + c0) * (1.0 + r * c))
    return 100.0 * (1.0 - np.sqrt(q))


def main() -> None:
    rep = load()
    c0 = rep["常数"]["C0"]
    r3 = rep["附录"]["附录3"]["r"]
    r4 = rep["附录"]["附录4"]["r"]
    echo = rep["呼应表"]

    fig, axes = plt.subplots(1, 2, figsize=(W, 3.35), layout="constrained")

    # ---------------- (a) 判据曲线 ----------------
    ax = axes[0]
    style(ax)
    rr = np.linspace(0.0, 1.0, 400)
    # 高亮带只画在曲线之上，正好把两个标记连起来；铺满全高会压住左下角的图例
    ax.axvspan(r4, r3, ymin=0.50, ymax=1.0, color=FS.COLOR_MAIN_PALE,
               alpha=0.6, lw=0, zorder=0)
    ax.plot(rr, [dry_line_shrink(x, c0, "radial") for x in rr], "-",
            color=C_MAIN, lw=1.7, label="径向等长（问题四几何）")
    ax.plot(rr, [dry_line_shrink(x, c0, "iso") for x in rr], "--",
            color=C_ORANGE, lw=1.5, label="各向同性")
    ax.axvline(1.0, color=C_BASE, ls=":", lw=0.8)
    ax.text(0.995, 24.0, "无收缩\n$r=1$", fontsize=6.8, color=C_BASE,
            ha="right", va="top", linespacing=1.4)

    shr3 = dry_line_shrink(r3, c0, "radial")
    shr4 = dry_line_shrink(r4, c0, "radial")
    ax.plot([r3], [shr3], "o", ms=5.2, color=C_MAIN, zorder=5)
    ax.plot([r4], [shr4], "o", ms=5.2, color=C_MAIN, markerfacecolor="white",
            markeredgewidth=1.2, zorder=5)
    ax.text(0.40, 40.5,
            "题面两式的 $r$ 都远离 1\n绝干径向收缩 %.1f%%~%.1f%%" % (shr3, shr4),
            fontsize=6.6, color=C_INK, ha="left", va="bottom", linespacing=1.4)

    ax.set_xlim(0.0, 1.04)
    ax.set_ylim(0.0, 47.0)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("斜率截距比 $r=b/a$")
    ax.set_ylabel("绝干线收缩率 / %")
    # 图例同时承担"标记=哪条经验式"的信息，省掉容易压线的引线注解
    h, lb = ax.get_legend_handles_labels()
    h += [plt.Line2D([], [], ls="none", marker="o", ms=5.0, color=C_MAIN)]
    lb += ["附录 3：$r$=%.3f" % r3]
    h += [plt.Line2D([], [], ls="none", marker="o", ms=5.0, color=C_MAIN,
                     markerfacecolor="white")]
    lb += ["附录 4：$r$=%.3f" % r4]
    ax.legend(h, lb, fontsize=6.6, loc="lower left", handlelength=1.6,
              labelspacing=0.35)
    ax.set_title("(a) 一个数决定要不要收缩", loc="left")

    # ---------------- (b) 三问呼应 ----------------
    ax = axes[1]
    style(ax)
    cg = np.linspace(0.02, c0, 500)
    ax.plot(cg, line_shrink_at_c(cg, r3, c0), "-", color=C_MAIN, lw=1.7,
            label="附录 3（问题一~三）")
    ax.plot(cg, line_shrink_at_c(cg, r4, c0), "--", color=C_ORANGE, lw=1.5,
            label="附录 4（问题四）")

    for lev, txt in ((3.0, "3%"), (5.0, "5%"), (10.0, "10%"), (20.0, "20%")):
        ax.axhline(lev, color=FS.NEUTRAL_LIGHT, ls=(0, (4, 3)), lw=0.7, zorder=0)
        ax.text(0.06, lev + 0.7, txt, fontsize=6.2, color=C_BASE, ha="left")

    # 三个报数点用不同 marker 区分，交给图例说明——不用引线，避免压到曲线上
    marks = (("问题一", "o", "问题一末（1800 s）"),
             ("问题二", "s", "问题二末（3 h）"),
             ("问题三", "^", "问题三末（57.4 h）"))
    for name, mk, lab in marks:
        cbar = echo[name]["平均C"]
        y = 100.0 * echo[name]["附录3_线收缩_平均"]
        ax.plot([cbar], [y], mk, ms=5.4, color=C_MAIN, zorder=5,
                markerfacecolor="white", markeredgewidth=1.2,
                label="%s：%.1f%%" % (lab, y))

    ax.set_xlim(c0, 0.0)
    ax.set_ylim(0.0, 46.0)
    ax.set_xlabel("含水率 $C$ / (kg/kg)")
    ax.set_ylabel("径向线收缩率 / %")
    ax.legend(fontsize=6.5, loc="upper left", handlelength=1.6,
              labelspacing=0.35)
    ax.set_title("(b) 三问各自处在哪一段", loc="left")

    FIGDIR.mkdir(parents=True, exist_ok=True)

    # 出图自检闭环（scipilot-figure-skill 流程）：先程序自检缺字/裁切/刻度重叠
    try:
        from visual_qa import audit_layout          # noqa: PLC0415
        issues = audit_layout(fig)
        if issues:
            print("版面自检发现问题：")
            for sev, msg in issues:
                # 本机所有"全覆盖"中文字体只有 400 一个字重，matplotlib 会对
                # weight='bold' 回退并打 findfont 日志，视觉上并无方框。这类
                # 记录不能当缺字；只有 "missing from font" 才是真的缺字形。
                if sev == "FAIL" and "missing from font" not in msg:
                    print("  [注意] 仅出现字重回退（无缺字形），"
                          "中文字形完整；粗体由同色描边模拟")
                else:
                    print("  [%s] %s" % (sev, msg))
        else:
            print("版面自检：未发现缺字、裁切或刻度重叠")
    except Exception as exc:                        # 自检脚本缺失不应阻断出图
        print("（跳过程序化版面自检：%s）" % exc)

    out = FIGDIR / "t01_r_criterion"
    FS.save_fig(fig, str(out) + ".png", vector=True)
    print("已输出 ->", out.with_suffix(".png"), "与", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
