#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把论文全文重排成 B 档结构（四问连续 + 统一检验章 + 收缩判据收口）。

重排规则
--------
原结构的问题：物性自洽性检验（7 页）排在所有问题之前；网格/时间步/界面系数/
径向效应四章（9 页）插在问题三与问题四之间，把"四问"的叙事切断。

新结构：
    1 问题重述   2 问题分析   3 模型假设   4 符号说明
    5 数据预处理与物性数据的可容许性检验      （原 §5 + 原 §6 主体，层级下沉一级）
    6 问题一     7 问题二     8 问题三     9 问题四   （四问连续）
    10 模型的检验与灵敏度分析                 （原 §10–§13 合并，层级下沉一级）
    11 收缩的建模判据与代价                   （原 §6.7 迁入，正文另补）
    12 结果汇总与交付说明                     （原 §15）
    13 模型的评价与推广                       （原 §16）
    参考文献、附录

只做"搬家与层级调整"，不改一个数字；缩写在正文写作阶段另行处理。

跑法：
    python3 代码/restructure_paper.py --in 论文/论文全文.tex --out 论文/论文全文.tex
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

MARK = re.compile(r"^\\section\*?\{.*\}$", re.M)


def split_blocks(src: str) -> tuple[str, list[tuple[str, str]], str]:
    """切成 (前言, [(标题行, 正文), ...], 尾部的 \\end{document})。"""
    marks = [(m.start(), m.group(0)) for m in MARK.finditer(src)]
    head = src[: marks[0][0]]
    blocks = []
    for i, (pos, line) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(src)
        blocks.append((line, src[pos:end]))
    tail = ""
    if "\\end{document}" in blocks[-1][1]:
        body, _, tail = blocks[-1][1].partition("\\end{document}")
        blocks[-1] = (blocks[-1][0], body)
        tail = "\\end{document}" + tail
    return head, blocks, tail


def demote(text: str, n: int = 1) -> str:
    """把 \\section 降为 \\subsection 等，用于把若干章合并成更大的一章。"""
    order = ["section", "subsection", "subsubsection", "paragraph", "subparagraph"]
    for _ in range(n):
        for lv in reversed(range(len(order) - 1)):
            text = re.sub(r"\\%s\*?\{" % order[lv], r"\\%s{" % order[lv + 1], text)
    return text


def blank_abstract(head: str) -> str:
    """按要求把摘要正文与关键词留空。"""
    head = re.sub(
        r"(\\begin\{center\}\s*\{\\zihao\{4\}\\bfseries 摘\\quad 要\}\s*\\end\{center\}\n)"
        r"(.*?)(\\vspace\{1em\}\n\\noindent\\textbf\{关键词\}[^\n]*\n(?:[^\n]*\n)*?)(\\newpage)",
        r"\1\n% 摘要正文：按要求留空，待撰写。\n\n\\vspace{1em}\n\\noindent\\textbf{关键词}：\n\n\4",
        head, flags=re.S)
    return head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="论文/论文全文.tex")
    ap.add_argument("--out", dest="dst", default=None)
    args = ap.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst or args.src)
    src = src_path.read_text(encoding="utf-8")
    head, blocks, tail = split_blocks(src)
    titles = [b[0] for b in blocks]
    if len(blocks) < 18:
        raise SystemExit("章节数与预期不符：%d" % len(blocks))

    B = {i + 1: blocks[i][1] for i in range(len(blocks))}

    # 原 §6 拆成三段：主体（6.1–6.6）、张力对照（6.7）、本节结论（6.8）
    s6 = B[6]
    m67 = re.search(r"\\subsection\{一个应当诚实说明的张力.*", s6, re.S)
    if not m67:
        raise SystemExit("找不到 §6.7")
    s6_body, s6_rest = s6[: m67.start()], s6[m67.start():]
    m68 = re.search(r"\\subsection\{本节结论\}", s6_rest)
    s67 = s6_rest[: m68.start()] if m68 else s6_rest

    # §5 与 §6 主体合并：§6 整体下沉一级，标题改为覆盖两部分内容
    b5 = re.sub(r"^\\section\{数据预处理\}",
                "\\\\section{数据预处理与物性数据的可容许性检验}",
                B[5], count=1, flags=re.M)
    merged5 = b5.rstrip() + "\n\n" + demote(s6_body, 1).strip() + "\n\n"

    # §10–§13 合并成"模型的检验与灵敏度分析"：整块下沉一级
    merged10 = (B[10].rstrip() + "\n\n" + B[11].rstrip() + "\n\n"
                + B[12].rstrip() + "\n\n" + B[13].rstrip() + "\n\n")
    merged10 = demote(merged10, 1)
    merged10 = ("\\section{模型的检验与灵敏度分析}\n"
                "本节把与四个问题都有关的四类检验集中说明：径向网格数与时间步长的\n"
                "数值收敛性、水分方程口径的取法、界面扩散系数的平均方式，以及一维\n"
                "径向假设本身的适用范围。这些结论不改变四个问题的求解流程，但决定了\n"
                "结果的有效位数与可复核性，故单列一章。\n\n"
                + merged10)

    # §11：收缩的建模判据与代价（先放原 §6.7，正文随后补写）
    s11 = ("\\section{收缩的建模判据与代价}\n"
           "\\label{sec:shrink-criterion}\n\n"
           + s67.strip() + "\n\n"
           "\\subsection{小结}\n"
           "本节给出的判据可复核：收缩几何只由斜率截距比 $r=b/a$ 决定，\n"
           "忽略收缩是否安全由临界含水率 $C^\\ast$ 与容差 $\\varepsilon$ 判定。\n\n")

    # 结果文件说明 → 结果汇总与交付说明
    b12 = re.sub(r"^\\section\{结果文件说明\}",
                 "\\\\section{结果汇总与交付说明}", B[15], count=1, flags=re.M)

    new = (blank_abstract(head)
           + B[1] + B[2] + B[3] + B[4] + merged5
           + B[7] + B[8] + B[9] + B[14]
           + merged10 + s11 + b12 + B[16] + B[17] + B[18]
           + tail)
    dst_path.write_text(new, encoding="utf-8")

    order = ["问题重述", "问题分析", "模型假设", "符号说明", "数据预处理",
             "问题一", "问题二", "问题三", "问题四", "模型的检验与灵敏度分析",
             "收缩的建模判据", "结果汇总", "模型的评价", "参考文献", "程序清单"]
    got = re.findall(r"^\\section\*?\{([^}]*)\}", new, re.M)
    print("重排后的一级标题：")
    for t in got:
        print("   ", t[:40])


if __name__ == "__main__":
    main()
