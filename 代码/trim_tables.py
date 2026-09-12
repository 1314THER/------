#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按清单裁剪正文表格：只从 .tex 里移除表块，数据与生成脚本一律不动。

分成两档，避免误删承载结论的表：
  SAFE   —— 与其它表/图严格重复，或纯事务性（删了不丢信息）
  MAYBE  —— 冗余但仍有独立信息，列出供人工决定

副作用：被移除的表带走的 \\label{tab:...}，正文对应的 \\ref 会变成 "??"，
报告里逐条列出。

跑法：
    python3 代码/trim_tables.py --dry-run
    python3 代码/trim_tables.py                 # 只删 SAFE
    python3 代码/trim_tables.py --with-maybe    # SAFE + MAYBE 一起删
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
TEX = PROJECT / "论文" / "论文全文.tex"
REPORT = PROJECT / "输出" / "表格裁剪报告.md"

# tab:face-grid 与 tab:face-mean 严格重复（后者含 N=100/200/400 三档，前者只有两档）；
# tab:code 是程序清单，附录里已有完整代码，属事务性表格
SAFE = {"tab:face-grid", "tab:code"}

# 冗余但各自还有一点独立信息，供人工定夺
MAYBE = {"tab:overview", "tab:phase", "tab:geo-shrink", "tab:err2",
         "tab:p1-conv", "tab:files"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--with-maybe", action="store_true")
    args = ap.parse_args()

    drop = set(SAFE) | (set(MAYBE) if args.with_maybe else set())
    src = TEX.read_text(encoding="utf-8")
    blocks = list(re.finditer(r"[ \t]*\\begin\{table\}.*?\\end\{table\}[ \t]*\n*",
                              src, re.S))
    removed, spans, kept = [], [], 0
    for m in blocks:
        lab = re.search(r"\\label\{([^}]*)\}", m.group(0))
        name = lab.group(1) if lab else ""
        if name in drop:
            spans.append(m.span())
            removed.append({"label": name, "line": src[: m.start()].count("\n") + 1})
        else:
            kept += 1

    out = src
    for a, b in reversed(spans):
        out = out[:a] + out[b:]
    out = re.sub(r"\n{3,}", "\n\n", out)

    refs = {}
    for r in removed:
        pat = "\\ref{%s}" % r["label"]
        hit = [(i, l.strip()) for i, l in enumerate(out.splitlines(), 1) if pat in l]
        if hit:
            refs[r["label"]] = hit

    lines = ["# 表格裁剪报告", "",
             "保留 %d 张，移除 %d 张。" % (kept, len(removed)), "",
             "## 已移除", "", "| 标签 | 原行号 |", "| --- | --- |"]
    for r in removed:
        lines.append("| `%s` | %d |" % (r["label"], r["line"]))
    lines += ["", "## 候选（未删除，供人工决定）", ""]
    for lab in sorted(MAYBE):
        lines.append("- `%s`" % lab)
    lines += ["", "## 需要人工清理的引用", ""]
    for lab, hit in refs.items():
        for i, l in hit:
            lines.append("- 第 %d 行 `%s`：%s" % (i, lab, l[:88]))
    if not refs:
        lines.append("（无）")
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    if not args.dry_run:
        TEX.write_text(out, encoding="utf-8")
    print("移除 %d 张表%s" % (len(removed),
                             "（dry-run）" if args.dry_run else ""))
    print("报告 ->", REPORT)


if __name__ == "__main__":
    main()
