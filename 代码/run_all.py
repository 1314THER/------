#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性求解问题二/三/四并导出结果数据（供生成 result2-4.xlsx）。

问题二与问题三使用同一套模型（附录 3 物性、无收缩），一次求解即可同时满足；
问题四使用附录 4 物性并计入附件 2 给出的收缩。

输出（npz）:
    times      时间序列 s
    conc21     固定物理位置 0,0.1,...,2.0 cm 处的水分浓度（超出当前半径处为 NaN）
    temp21     同上，温度
    valid21    该位置是否在药材内部（bool）
    conc_surf  药材表面（r = R(t)）处的水分浓度
    temp_surf  药材表面温度
    radius     R(t)  m
    dry_time   中心含水率首次低于 0.15 的时刻 s

运行:
    PY=/Users/sqz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
    $PY 代码/run_all.py --n 200 --dt 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

from solve_p234 import solve  # noqa: E402

POSITIONS = [round(0.1 * k, 1) for k in range(21)]     # 0, 0.1, ..., 2.0 cm


def series(sol, positions):
    """把贴体坐标解插值到固定物理位置，并标出"已经不在药材内部"的点。

    为什么需要这一步
    ----------------
    solve() 返回的解定义在贴体坐标 ξ=r/R(t) 上，网格节点随时间跟着材料走。
    但题目要求的输出位置是**固定的物理距离**（0, 0.1, …, 2.0 cm），
    所以每个时刻都要重新换算：物理位置 r_k 对应的贴体坐标是 r_k/R(t)。

    为什么要 valid 掩码
    ------------------
    问题四里药材半径会收缩到 1.198 cm。此后 1.3～2.0 cm 这些位置已经在
    药材外部，含水率**没有定义**（不是"等于某个值"）。这些格子必须留空，
    而不能填 0 或用最近邻外推——所以先算出 valid，再只对内部点插值，
    外部点保持 NaN，最终在 Excel 里体现为空白单元格。
    """
    xi = sol["xi"]
    n_t = len(sol["times"])
    n_p = len(positions)
    temp = np.full((n_t, n_p), np.nan)      # 先全填 NaN，只覆盖有效的格子
    conc = np.full((n_t, n_p), np.nan)
    valid = np.zeros((n_t, n_p), dtype=bool)
    for k in range(n_t):
        rad = sol["radius"][k]
        target = np.asarray(positions, float) / 100.0
        inside = target <= rad + 1e-12       # 加 1e-12 容忍浮点比较误差
        valid[k] = inside
        if inside.any():
            # 物理位置 → 贴体坐标，再在 ξ 网格上线性插值
            t_xi = np.clip(target[inside] / rad, 0.0, 1.0)
            temp[k, inside] = np.interp(t_xi, xi, sol["temp"][k])
            conc[k, inside] = np.interp(t_xi, xi, sol["conc"][k])
    return temp, conc, valid


def export(sol, tag, n, dt):
    temp, conc, valid = series(sol, POSITIONS)
    temp_surf = sol["temp"][:, -1]
    conc_surf = sol["conc"][:, -1]
    out = PROJECT / "输出" / ("%s_data.npz" % tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        times=sol["times"],
        conc21=conc,
        temp21=temp,
        valid21=valid,
        conc_surf=conc_surf,
        temp_surf=temp_surf,
        radius=sol["radius"],
        dry_time=np.array(
            [sol["dry_time"] if sol["dry_time"] is not None else np.nan]
        ),
        n=np.array([n]),
        dt=np.array([dt]),
    )
    print("已导出: %s  (%d 个时间点)" % (out, len(sol["times"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dt", type=float, default=5.0)
    ap.add_argument("--only", type=int, default=0, help="仅跑指定题号")
    args = ap.parse_args()

    if args.only in (0, 2, 3):
        # 问题二与问题三共用同一套模型（附录 3 物性、半径恒为 2 cm），
        # 差别只在于"取到什么时候"：问题二只要 3 h 内的结果，
        # 问题三要一直算到中心含水率低于 0.15。算一次全长即可同时满足，
        # 无需重复求解。
        print("=== 问题二/三：附录 3 物性，无收缩 ===")
        sol23 = solve(
            prop="p23", shrink=False, n=args.n, dt=args.dt, tend=259200.0
        )
        print(
            "  烘干时间: %.0f s = %.2f h = %.3f 天"
            % (
                sol23["dry_time"],
                sol23["dry_time"] / 3600,
                sol23["dry_time"] / 86400,
            )
        )
        export(sol23, "result23", args.n, args.dt)

    if args.only in (0, 4):
        print("=== 问题四：附录 4 物性，含收缩 ===")
        sol4 = solve(
            prop="p4", shrink=True, n=args.n, dt=args.dt, tend=259200.0
        )
        print(
            "  烘干时间: %.0f s = %.2f h = %.3f 天"
            % (
                sol4["dry_time"],
                sol4["dry_time"] / 3600,
                sol4["dry_time"] / 86400,
            )
        )
        export(sol4, "result4", args.n, args.dt)


if __name__ == "__main__":
    main()
