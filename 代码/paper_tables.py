#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出论文正文用的表 3、表 4、表 5、表 6（Markdown 格式）。

输入：输出/result23_data.npz、输出/result4_data.npz
      （由 代码/run_all.py 生成）
输出：输出/问题二三四-表格.md

四张表的采样规则（与题面要求一致）：
    表 3 / 表 4（问题二）：3 h 内每 0.5 h，位置 0/0.5/1/1.5/2 cm
    表 5（问题三）：每 6 h 直至烘干结束，位置同上
    表 6（问题四）：同表 5，但末列改为"药材表面"（R(t) 本身在变），
                    且超出当前半径的固定位置以"——"表示

row_at() 用 np.argmin 找最近的**已存解**时间层：因为求解时间步是 5 s，
而表中时间点都是 5 s 的整数倍，取到的是精确对应的那一层，不会引入误差。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "输出"


def row_at(times, values, t_query, pos_idx):
    k = int(np.argmin(np.abs(times - t_query)))
    return k, [values[k][j] for j in pos_idx]


def fmt(v):
    return "——" if v is None or not np.isfinite(v) else "%.4f" % v


def table(lines, title, times, grid, t_list, pos_cm, extra=None, extra_t=None):
    head = "| 时间/h | " + " | ".join(
        p if isinstance(p, str) else "%g" % p for p in pos_cm
    ) + " |"
    lines.append("## " + title)
    lines.append("")
    lines.append(head)
    lines.append("| --- |" + " --- |" * len(pos_cm))
    for label, t in t_list:
        k = int(np.argmin(np.abs(times - t)))
        cells = []
        for p in pos_cm:
            if p == "药材表面":
                cells.append(fmt(extra[k] if extra is not None else np.nan))
            else:
                j = int(round(p / 0.1))
                v = grid[k][j]
                cells.append(fmt(v if np.isfinite(v) else np.nan))
        lines.append("| %s | " % label + " | ".join(cells) + " |")
    lines.append("")


def main():
    lines = ["# 问题二、三、四结果表", ""]

    # ---------------- 问题二 ----------------
    d = np.load(OUT / "result23_data.npz")
    t = d["times"]
    pos_idx = [0, 5, 10, 15, 20]
    pos_cm = [0, 0.5, 1.0, 1.5, 2.0]

    t3 = [("%.1f" % h, h * 3600) for h in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)]
    table(lines, "表 3　3 小时内药材的温度（单位：°C）",
          t, d["temp21"], t3, pos_cm)
    table(lines, "表 4　3 小时内药材的水分浓度（单位：kg/kg）",
          t, d["conc21"], t3, pos_cm)

    # ---------------- 问题三 ----------------
    dry3 = float(d["dry_time"][0])
    t5 = [("%d" % (h // 3600), float(h))
          for h in np.arange(6 * 3600, dry3, 6 * 3600)]
    t5.append(("烘干结束时间", dry3))
    table(lines, "表 5　药材烘干过程的水分浓度（单位：kg/kg）",
          t, d["conc21"], t5, pos_cm)
    lines.append("> 问题三烘干结束时间：%.0f s = %.2f h = %.3f 天。"
                 % (dry3, dry3 / 3600, dry3 / 86400))
    lines.append("")

    # ---------------- 问题四 ----------------
    e = np.load(OUT / "result4_data.npz")
    te = e["times"]
    dry4 = float(e["dry_time"][0])
    t6 = [("%d" % (h // 3600), float(h))
          for h in np.arange(6 * 3600, dry4, 6 * 3600)]
    t6.append(("烘干结束时间", dry4))
    table(lines, "表 6　药材烘干过程的水分浓度（含收缩，单位：kg/kg）",
          te, e["conc21"], t6, [0, 0.5, 1.0, 1.5, 2.0, "药材表面"],
          extra=e["conc_surf"])
    lines.append("> 问题四烘干结束时间：%.0f s = %.2f h = %.3f 天。"
                 % (dry4, dry4 / 3600, dry4 / 86400))
    lines.append("> 半径由 2.000 cm 收缩至 %.3f cm，故 1.5 cm 与 2.0 cm 处的"
                 "数据在半径收缩到该值以下后无定义，以——表示。"
                 % (e["radius"][-1] * 100))
    lines.append("")

    out = OUT / "问题二三四-表格.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
