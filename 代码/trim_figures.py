#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按保留清单裁剪正文插图：只从 .tex 里移除图块，磁盘上的图片文件一律保留。

为什么用清单而不是删文件
------------------------
正文 40 张图对四问而言过密（约 18 页），压到 30 页以内必须减图；但图片本身
（论文/figs/ 下的 PDF/PNG）是资产，删掉就不可逆。所以本脚本只把 .tex 里的
\\begin{figure}...\\end{figure} 整块摘掉，figs/ 目录一个字节都不动。

副作用（需要人工处理）
----------------------
被移除的图带走的 \\label{fig:...}，正文里对应的 \\ref{fig:...} 会变成 "??"。
本脚本把每一处受影响的引用连行号列出来，写到 输出/插图裁剪报告.md，供后续
清理行文时逐一处理。

跑法：
    python3 代码/trim_figures.py --dry-run     # 只出报告，不改文件
    python3 代码/trim_figures.py               # 实际裁剪
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
TEX = PROJECT / "论文" / "论文全文.tex"
REPORT = PROJECT / "输出" / "插图裁剪报告.md"
FIGDIR = PROJECT / "论文" / "figs"

# 保留的 15 张：路线图、两套原始数据、ρ–C 与收缩假设、几何降维、
# 四问各一张主结果图、收敛性两张、口径对照、几何收缩
KEEP = {
    "s07_roadmap", "d01_air_series", "d03_radius", "d26_rho_hypothesis",
    "s01_geometry", "d05_p1_fields", "d11_p2_fields", "d13_props",
    "d15_p3_curve", "d19_p4_shrink", "d20_p4_field", "d22_convergence",
    "d31_grid_order", "d23_closures", "d29_geo_shrink",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = TEX.read_text(encoding="utf-8")
    blocks = list(re.finditer(r"[ \t]*\\begin\{figure\}.*?\\end\{figure\}[ \t]*\n*",
                              src, re.S))

    removed, kept_names = [], []
    spans = []
    for m in blocks:
        img = re.search(r"includegraphics\[[^\]]*\]\{([^}]*)\}", m.group(0))
        name = img.group(1).split("/")[-1].rsplit(".", 1)[0] if img else ""
        lab = re.search(r"\\label\{([^}]*)\}", m.group(0))
        if name in KEEP:
            kept_names.append(name)
            continue
        spans.append(m.span())
        removed.append({"name": name,
                        "label": lab.group(1) if lab else "",
                        "line": src[: m.start()].count("\n") + 1})

    out = src
    for a, b in reversed(spans):
        out = out[:a] + out[b:]
    out = re.sub(r"\n{3,}", "\n\n", out)

    # 受影响的行文引用（按行号列出，供人工清理）
    refs = {}
    for r in removed:
        if not r["label"]:
            continue
        pat = "\\ref{%s}" % r["label"]
        hit = [(i, l.strip()) for i, l in enumerate(out.splitlines(), 1)
               if pat in l]
        if hit:
            refs[r["label"]] = hit

    missing = KEEP - set(kept_names)
    lines = ["# 插图裁剪报告", "",
             "保留图 %d 张，移除图块 %d 个（论文/figs/ 下的文件全部保留）。"
             % (len(kept_names), len(removed)), ""]
    if missing:
        lines += ["> 注意：保留清单里有 %d 个文件名没在正文中出现：%s"
                  % (len(missing), ", ".join(sorted(missing))), ""]
    lines += ["## 已移除的图", "", "| 图片 | 标签 | 原行号 |", "| --- | --- | --- |"]
    for r in removed:
        lines.append("| %s | `%s` | %d |" % (r["name"], r["label"], r["line"]))
    lines += ["", "## 需要人工清理的引用", "",
              "下列 \\ref 指向已移除的图，编译后会显示为 ??。", ""]
    for lab, hit in refs.items():
        for i, l in hit:
            lines.append("- 第 %d 行 `%s`：%s" % (i, lab, l[:90]))
    if not refs:
        lines += ["（无——正文未直接引用被删图的标签）"]
    lines += ["", "## 磁盘文件状态", "",
              "论文/figs/ 共 %d 个文件，本次未删任何文件。"
              % len([p for p in FIGDIR.iterdir() if p.is_file()]), ""]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    if not args.dry_run:
        TEX.write_text(out, encoding="utf-8")
    print("保留 %d 张，移除 %d 个图块%s"
          % (len(kept_names), len(removed), "（dry-run，未写文件）" if args.dry_run else ""))
    print("报告 ->", REPORT)


if __name__ == "__main__":
    main()
