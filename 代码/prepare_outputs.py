#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把问题二/三/四的解整理成 result2/3/4.xlsx 所需的中间数据。

输出到 输出/ 下的二进制（float64 小端、行主序）与元数据 JSON：
    result2.bin  : times[N2], temp[N2*21], conc[N2*21]
    result3.bin  : times[N3], conc[N3*21]
    result4.bin  : times[N4], conc[N4*21], conc_surf[N4]
    result_meta.json
其中 result2 按题面要求为 1 s 间隔（由 dt 解线性插值）；result3/4 为 60 s 间隔。
超出当前半径的位置在 result4 中写为 NaN，生成 xlsx 时留空。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "输出"


def resample(times, values, targets):
    """把 (n_t, n_p) 的时空解按时间线性插值到 targets 时刻。

    逐列处理（而不是整体插值），是因为问题四的某些列含 NaN——
    那些位置超出了当前半径，无定义。插值前先剔除 NaN，只在该列有效的
    时间范围内插值，避免 NaN 污染整列。
    """
    out = np.empty((len(targets), values.shape[1]))
    for j in range(values.shape[1]):
        col = values[:, j]
        good = ~np.isnan(col)                # 该列有效的时间点
        out[:, j] = np.interp(targets, times[good], col[good])
    return out


def main():
    # ---------------- 问题二 / 三 ----------------
    d23 = np.load(OUT / "result23_data.npz")
    t23 = d23["times"]
    dry23 = float(d23["dry_time"][0])
    # 题面要求 result2 是**1 s 间隔**，而求解时用的是 dt=5 s 的粗时间步
    # （为了控制 61 h 的计算量）。所以这里把 5 s 解线性插值到 1 s 网格。
    # 时间步收敛性检验表明 dt=5 s 与 2 s 的烘干时长只差 12 s，
    # 故该插值不引入有意义的误差。
    n2 = int(round(dry23))                       # 1 s 一个点
    t2 = np.arange(1, n2 + 1, dtype=float)
    temp2 = resample(t23, d23["temp21"], t2)
    conc2 = resample(t23, d23["conc21"], t2)
    with open(OUT / "result2.bin", "wb") as fh:
        fh.write(np.ascontiguousarray(t2).astype("<f8").tobytes())
        fh.write(np.ascontiguousarray(temp2).astype("<f8").tobytes())
        fh.write(np.ascontiguousarray(conc2).astype("<f8").tobytes())
    print("result2: %d 行 x 21 列（1 s 间隔，插值自 %g s 解）" % (n2, t23[1]))

    n3 = int(dry23 // 60)
    t3 = np.arange(60, n3 * 60 + 1, 60, dtype=float)
    # result3 要 60 s 间隔。因为求解时间步是 5 s，60 一定是 5 的整数倍，
    # 所以 60 s 网格点**恰好落在**已保存的解上，searchsorted 取到的是
    # 精确对应的时间层，不需要再做插值。
    idx3 = np.searchsorted(t23, t3)
    idx3 = np.clip(idx3, 0, len(t23) - 1)
    conc3 = d23["conc21"][idx3]
    with open(OUT / "result3.bin", "wb") as fh:
        fh.write(np.ascontiguousarray(t3).astype("<f8").tobytes())
        fh.write(np.ascontiguousarray(conc3).astype("<f8").tobytes())
    print("result3: %d 行 x 21 列（60 s 间隔）" % len(t3))

    # ---------------- 问题四 ----------------
    d4 = np.load(OUT / "result4_data.npz")
    t4all = d4["times"]
    dry4 = float(d4["dry_time"][0])
    n4 = int(dry4 // 60)
    t4 = np.arange(60, n4 * 60 + 1, 60, dtype=float)
    idx4 = np.clip(np.searchsorted(t4all, t4), 0, len(t4all) - 1)
    conc4 = d4["conc21"][idx4].copy()
    surf4 = d4["conc_surf"][idx4].copy()
    # 超出当前半径的位置留空
    rad4 = d4["radius"][idx4]
    pos_cm = np.array([round(0.1 * k, 1) for k in range(21)])
    outside = pos_cm[None, :] > rad4[:, None] * 100.0 + 1e-9
    conc4[outside] = np.nan
    with open(OUT / "result4.bin", "wb") as fh:
        fh.write(np.ascontiguousarray(t4).astype("<f8").tobytes())
        fh.write(np.ascontiguousarray(conc4).astype("<f8").tobytes())
        fh.write(np.ascontiguousarray(surf4).astype("<f8").tobytes())
    print(
        "result4: %d 行 x 21 列 + 药材表面（60 s 间隔，末年半径 %.4f cm）"
        % (len(t4), rad4[-1] * 100)
    )

    meta = {
        "positions_cm": [round(0.1 * k, 1) for k in range(21)],
        "n2": n2,
        "n3": len(t3),
        "n4": len(t4),
        "dry2_s": dry23,
        "dry4_s": dry4,
        "dry2_days": dry23 / 86400.0,
        "dry4_days": dry4 / 86400.0,
    }
    (OUT / "result_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("烘干时间: 问题三 %.0f s (%.2f 天), 问题四 %.0f s (%.2f 天)"
          % (dry23, dry23 / 86400, dry4, dry4 / 86400))


if __name__ == "__main__":
    main()
