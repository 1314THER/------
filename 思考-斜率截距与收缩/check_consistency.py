#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一致性自检：正文/README 里出现的每个数字，都必须等于 compute_shrinkage.py 的计算结果。

为什么需要这个脚本
------------------
本节的全部数字应当来自 `输出/斜率截距与收缩.json`，但 `.tex` 与 `README.md`
里的数字是手写的——中间断了一层。这个脚本把那层补上：从 JSON 取数、按正文
所用的位数格式化，再回到 `.tex` / `README.md` 里逐条比对。

改了 `compute_shrinkage.py` 或换了结果文件之后，先跑它：
任何一条 FAIL 都说明"文中的数字"与"算出来的数字"已经不一致。

    PY=/Users/sqz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
    $PY check_consistency.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def f3(x: float) -> str:
    return "%.3f" % x


def f1(x: float) -> str:
    return "%.1f" % x


def pct1(x: float) -> str:
    """小数 -> 百分数字符串（正文写作 $35.0\\%$，比对时只查数字部分）。"""
    return "%.1f" % (100.0 * x)


def main() -> int:
    rep = json.loads((HERE / "输出" / "斜率截距与收缩.json").read_text(encoding="utf-8"))
    tex = (HERE / "斜率截距与收缩.tex").read_text(encoding="utf-8")
    rdm = (HERE / "README.md").read_text(encoding="utf-8")

    a3, a4 = rep["附录"]["附录3"], rep["附录"]["附录4"]
    at2 = rep["附件2"]
    thr3, thr4 = rep["门槛表"]["附录3"], rep["门槛表"]["附录4"]
    echo = rep["呼应表"]
    path = {round(p["t_h"], 1): p for p in rep["路径对照"]}
    rng = rep["路径偏差范围_pct"]

    # (说明, 期望字面量, 计算值, 正文中用于定位的片段)
    checks: list[tuple[str, str, str, str]] = []

    def add(label: str, literal: str, calc: str, doc_lit: str | None = None) -> None:
        checks.append((label, literal, calc, literal if doc_lit is None else doc_lit))

    # ---- 判据参数
    add("r 附录3", "0.197", f3(a3["r"]))
    add("r 附录4", "0.118", f3(a4["r"]))
    add("无收缩反例 rho(C0)", "2307.5", f1(a3["rho_if_no_shrink"]))

    # ---- 绝干半径与 beta
    add("绝干半径 附录3", "1.301", f3(a3["R_dry_cm_radial"]))
    add("绝干半径 附录4", "1.211", f3(a4["R_dry_cm_radial"]))
    add("实测终半径 附件2", "1.198", f3(at2["R_end_cm"]))
    add("beta 附录3", "0.822", f3(a3["beta_endpoint"]))
    add("beta 附录4", "0.891", f3(a4["beta_endpoint"]))
    b_at2 = at2["反标定beta"]["附录4"]
    add("beta 附件2", "0.922", f3(b_at2))
    add("beta 相对差", "3.4", f1(100.0 * (b_at2 - a4["beta_endpoint"]) / b_at2))
    add("半径偏差 附录4", "1.1", f1(at2["绝干半径对照"]["附录4"]["偏差_pct"]))
    add("半径偏差 附录3", "8.6", f1(at2["绝干半径对照"]["附录3"]["偏差_pct"]))

    # ---- 表 1
    add("V(0)/V(C0) 附录3", "0.423", f3(a3["V_dry_over_V_C0"]))
    add("V(0)/V(C0) 附录4", "0.367", f3(a4["V_dry_over_V_C0"]))
    add("V_end/V0 附件2", "0.359", f3(at2["V_end_over_V0_radial"]))
    add("容许带下界 附录3", "85.6", f1(a3["b_band_lo"]))
    add("容许带下界 附录4", "62.1", f1(a4["b_band_lo"]))

    # ---- 表 3 门槛
    for lv, lit3, lit4 in (("3%", "2.177", "2.249"), ("5%", "1.953", "2.062"),
                           ("10%", "1.467", "1.637"), ("20%", "0.730", "0.939")):
        add(f"门槛 {lv} 附录3", lit3, f3(thr3[lv]))
        add(f"门槛 {lv} 附录4", lit4, f3(thr4[lv]))
    add("正文门槛 3%", "2.18", "%.2f" % thr3["3%"])
    add("正文门槛 10%", "1.47", "%.2f" % thr3["10%"])
    add("正文门槛 20%", "0.73", "%.2f" % thr3["20%"])

    # ---- 表 4 三问呼应
    for name, lit_c, lit3, lit4 in (("问题一", "2.253", "2.4", "2.9"),
                                    ("问题二", "1.364", "11.2", "13.6"),
                                    ("问题三", "0.120", "32.0", "36.4")):
        add(f"{name} 平均C", lit_c, f3(echo[name]["平均C"]))
        add(f"{name} 线收缩 附录3", lit3, f1(100 * echo[name]["附录3_线收缩_平均"]))
        add(f"{name} 线收缩 附录4", lit4, f1(100 * echo[name]["附录4_线收缩_平均"]))

    # ---- 图注：绝干线收缩
    add("绝干线收缩 附录3 径向", "35.0", f1(100 * (1 - a3["V_dry_over_V_C0"] ** 0.5)))
    add("绝干线收缩 附录4 径向", "39.4", f1(100 * (1 - a4["V_dry_over_V_C0"] ** 0.5)))
    add("绝干线收缩 附录3 各向同性", "24.9",
        f1(100 * (1 - a3["V_dry_over_V_C0"] ** (1.0 / 3.0))))

    # ---- 表 2 路径反馈
    for th, lit_r, lit_c, lit_d in ((5.0, "1.424", "0.959", "21.4"),
                                    (10.0, "1.273", "0.510", "22.4"),
                                    (20.0, "1.210", "0.221", "16.1"),
                                    (50.0, "1.200", "0.112", "10.6")):
        p = path[th]
        add(f"路径 {th:g}h 半径", lit_r, f3(p["R_cm"]))
        add(f"路径 {th:g}h 平均C", lit_c, f3(p["C_bar"]))
        add(f"路径 {th:g}h 偏差", lit_d, f1(abs(p["dev_pct"])))
    add("路径偏差范围下界", "10.6", f1(abs(rng["max"])))
    add("路径偏差范围上界", "22.4", f1(abs(rng["min"])))

    # ---- 附件 2 收缩完成度（正文"22 h 内基本完成"）
    add("收缩 99% 完成时刻", "22", "%.0f" % at2["收缩完成度"]["99%"]["t_h"],
        doc_lit="22$\\,h 内基本完成")

    # ================= 执行比对 =================
    bad = 0
    width = max(len(c[0]) for c in checks)
    for label, literal, calc, doc_lit in checks:
        in_doc = doc_lit in tex
        ok_calc = literal == calc
        if in_doc and ok_calc:
            flag = "OK  "
        else:
            flag = "FAIL"
            bad += 1
        detail = ""
        if not ok_calc:
            detail += f" 计算值={calc}"
        if not in_doc:
            detail += " 正文中未找到该字面量"
        print(f"[{flag}] {label:<{width}}  正文={literal:<8}{detail}")

    # ---- README 里的两处范围表述
    for label, literal, where in (("README 路径范围", "10.6%~22.4%", rdm),
                                  ("README 总量范围", "1%~3%", rdm)):
        ok = literal in where
        if not ok:
            bad += 1
        print(f"[{'OK  ' if ok else 'FAIL'}] {label:<{width}}  文本={literal}")

    print()
    print("共 %d 项，失败 %d 项。" % (len(checks) + 2, bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
