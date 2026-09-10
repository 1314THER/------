#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2026 高教社杯全国大学生数学建模竞赛 A 题　问题二 / 三 / 四求解.

控制方程（圆柱坐标，一维径向轴对称）:
    热量:  d(rho*cp*T)/dt = (1/r) * d/dr ( k * r * dT/dr )
    水分:  d(rho*C)/dt    = (1/r) * d/dr ( rho * D * r * dC/dr )

与问题一的区别：物性 rho、cp、k、D 均随 C 变化（D 还随 T），两个方程真正耦合，
必须联立迭代。附录 3、4 把 rho 写成 C 的函数，说明它参与方程（问题一中 rho 为
常数才可约去），故此处保留 rho 的守恒形式。

定解条件:
    T(r,0) = 28 C,  C(r,0) = 2.55 kg/kg
    r = 0 :  dT/dr = 0,  dC/dr = 0
    r = R :  -k dT/dr = h (T_s - T_inf(t)),
             -D dC/dr = hm (C_s - C_inf(t))

烘房边界：0~14400 s 由附件 1 线性插值；此后为恒温干燥阶段，取
T_inf = 50.0 C、C_inf = 0.05 kg/kg —— 附件 1 末段 1 h 的平均值恰为
49.9989 / 0.04999，可见这正是题设的恒温段工况。

问题四引入收缩，R(t) 由附件 2 经单调三次 Hermite 插值（PCHIP）给出，
计算在贴体坐标 xi = r/R(t) 上进行，方程增加对流项
    rho*cp*xi*R*Rdot*dT/dxi （水分方程同理）。

数值方法：节点中心有限体积（离散守恒）+ 后向 Euler（无条件稳定）
+ Picard 迭代（物性滞后），每步化为三对角方程组用追赶法求解。

运行（系统 python3 缺 openpyxl，请用打包运行时）:
    PY=/Users/sqz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
    $PY 代码/solve_p234.py --problem 2 --dt 5 --save 输出/result2_data.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openpyxl


PROJECT = Path(__file__).resolve().parent.parent
ATT1 = PROJECT / "附件" / "附件1-烘房温度与水分浓度.xlsx"
ATT2 = PROJECT / "附件" / "附件2-药材半径.xlsx"

R0 = 0.02                 # 初始半径 2 cm                 m
L_LEN = 0.25              # 药材长度 25 cm                m
H_HEAT = 25.0             # 对流换热系数（附录 2）         W/(m^2 K)
H_MASS = 8.0e-7           # 对流传质系数（附录 2）         m/s

T_INIT, C_INIT = 28.0, 2.55
T_CONST, C_CONST = 50.0, 0.05     # 恒温干燥段工况
T_PREHEAT_END = 14400.0           # 附件 1 覆盖的时间上限 s

C_DRY = 0.15              # 烘干判据 kg/kg

N_FINE = 100              # 计算网格单元数（xi 方向，h = 0.01）


# --------------------------------------------------------------------------
# 物性：附录 3（问题二、三）与附录 4（问题四）
# --------------------------------------------------------------------------
def props_p23(conc, temp):
    c = np.maximum(conc, 1e-12)
    t_k = temp + 273.15
    rho = 650.0 + 128.0 * c
    drho = np.full_like(c, 128.0)
    cp = 1450.0 + 2736.0 * c / (c + 1.0)
    k = 0.21 + 0.38 * c / (c + 1.0)
    d = 2.4e-3 * np.exp(-0.45 / c) * np.exp(-3850.0 / t_k)
    return rho, drho, cp, k, d


def props_p4(conc, temp):
    c = np.maximum(conc, 1e-12)
    t_k = temp + 273.15
    rho = 760.0 + 90.0 * c
    drho = np.full_like(c, 90.0)
    cp = 1850.0 + 2150.0 * c / (c + 1.0)
    k = 0.12 + 0.20 * c / (c + 1.0)
    d = 4.2e-4 * np.exp(-0.30 / c) * np.exp(-3850.0 / t_k)
    return rho, drho, cp, k, d


PROPERTY_SETS = {"p23": props_p23, "p4": props_p4}


# --------------------------------------------------------------------------
# 题目附件数据
# --------------------------------------------------------------------------
def _read(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    return [
        r
        for r in wb.active.iter_rows(min_row=2, values_only=True)
        if r[0] is not None
    ]


_AIR = _read(ATT1)
_AIR_T = np.array([r[0] for r in _AIR], float)
_AIR_TEMP = np.array([r[1] for r in _AIR], float)
_AIR_C = np.array([r[2] for r in _AIR], float)

_RAD = _read(ATT2)
_RAD_T = np.array([r[0] for r in _RAD], float)
_RAD_V = np.array([r[1] for r in _RAD], float) / 100.0


def air_conditions(t_query):
    """烘房温度与水分浓度：预热段插值，恒温段取常数。"""
    tq = np.asarray(t_query, float)
    scalar = tq.ndim == 0
    tq = np.atleast_1d(tq)
    temp = np.where(
        tq <= T_PREHEAT_END, np.interp(tq, _AIR_T, _AIR_TEMP), T_CONST
    )
    humid = np.where(
        tq <= T_PREHEAT_END, np.interp(tq, _AIR_T, _AIR_C), C_CONST
    )
    return (float(temp[0]), float(humid[0])) if scalar else (temp, humid)


# --------------------------------------------------------------------------
# 单调三次 Hermite 插值（Fritsch-Carlson），用于 R(t)
# --------------------------------------------------------------------------
class Pchip:
    """保单调的 C1 插值，避免直接对附件 2 做线性插值导致 Rdot 跳变。"""

    def __init__(self, x, y):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.h = np.diff(self.x)
        delta = np.diff(self.y) / self.h
        n = len(self.x)
        m = np.zeros(n)
        m[0] = delta[0]
        m[-1] = delta[-1]
        for i in range(1, n - 1):
            if delta[i - 1] * delta[i] <= 0.0:
                m[i] = 0.0
            else:
                w1 = 2.0 * self.h[i] + self.h[i - 1]
                w2 = self.h[i] + 2.0 * self.h[i - 1]
                m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
        self.m = m

    def _locate(self, xq):
        return np.clip(
            np.searchsorted(self.x, xq, side="right") - 1, 0, len(self.x) - 2
        )

    def __call__(self, xq):
        xq = np.asarray(xq, float)
        scalar = xq.ndim == 0
        xq = np.atleast_1d(xq)
        idx = self._locate(xq)
        h = self.h[idx]
        s = (xq - self.x[idx]) / h
        y0, y1 = self.y[idx], self.y[idx + 1]
        m0, m1 = self.m[idx], self.m[idx + 1]
        out = (
            (2 * s**3 - 3 * s**2 + 1) * y0
            + (s**3 - 2 * s**2 + s) * h * m0
            + (-2 * s**3 + 3 * s**2) * y1
            + (s**3 - s**2) * h * m1
        )
        return float(out[0]) if scalar else out

    def deriv(self, xq):
        xq = np.asarray(xq, float)
        scalar = xq.ndim == 0
        xq = np.atleast_1d(xq)
        idx = self._locate(xq)
        h = self.h[idx]
        s = (xq - self.x[idx]) / h
        y0, y1 = self.y[idx], self.y[idx + 1]
        m0, m1 = self.m[idx], self.m[idx + 1]
        out = (
            (6 * s**2 - 6 * s) * y0 / h
            + (3 * s**2 - 4 * s + 1) * m0
            + (-6 * s**2 + 6 * s) * y1 / h
            + (3 * s**2 - 2 * s) * m1
        )
        return float(out[0]) if scalar else out


def thomas(lower, diag, upper, rhs):
    """追赶法求解三对角方程组。"""
    n = len(diag)
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i] * cp[i - 1]
        cp[i] = upper[i] / m
        dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _assemble(lam, face, adv, value, alpha_const, far_value, n):
    """组装一个扩散-对流-对流边界的隐式三对角系统。"""
    lower = np.zeros(n + 1)
    diag = np.zeros(n + 1)
    upper = np.zeros(n + 1)
    rhs = np.zeros(n + 1)

    # 中心节点：无左侧界面，对流项因 xi_0 = 0 而消失
    diag[0] = lam[0] + face[0]
    upper[0] = -face[0]
    rhs[0] = lam[0] * value[0]

    for i in range(1, n):
        bm = face[i - 1]
        bp = face[i]
        a = adv[i]
        lower[i] = -bm - a
        diag[i] = lam[i] + bm + bp
        upper[i] = -bp + a
        rhs[i] = lam[i] * value[i]

    # 表面节点
    beta = face[n - 1]
    gamma = adv[n]
    lower[n] = -beta + gamma
    diag[n] = lam[n] + alpha_const + beta - gamma
    rhs[n] = lam[n] * value[n] + alpha_const * far_value
    return lower, diag, upper, rhs


def solve(
    prop="p23",
    shrink=False,
    n=N_FINE,
    dt=5.0,
    tend=259200.0,
    picard=6,
    picard_tol=1e-11,
):
    """求解问题二/三/四，返回完整时空解（贴体坐标）。"""
    props = PROPERTY_SETS[prop]
    h = 1.0 / n
    xi = np.arange(n + 1) * h
    xif = (np.arange(n) + 0.5) * h

    # 控制体权重 w_i = ∫ xi dxi
    weight = xi * h
    weight[0] = h * h / 8.0
    weight[n] = h * (1.0 - h / 4.0) / 2.0

    radius = Pchip(_RAD_T, _RAD_V) if shrink else None

    temp = np.full(n + 1, T_INIT)
    conc = np.full(n + 1, C_INIT)

    n_steps = int(round(tend / dt))
    times = np.arange(n_steps + 1) * dt
    temp_hist = np.empty((n_steps + 1, n + 1))
    conc_hist = np.empty((n_steps + 1, n + 1))
    radius_hist = np.empty(n_steps + 1)
    temp_hist[0] = temp
    conc_hist[0] = conc
    radius_hist[0] = radius(0.0) if shrink else R0

    dry_step = None
    for step in range(1, n_steps + 1):
        t_now = step * dt
        t_inf, c_inf = air_conditions(t_now)
        r_now = radius(t_now) if shrink else R0
        r_dot = radius.deriv(t_now) if shrink else 0.0

        # 时间步开始时的状态：右端项必须始终用它，
        # 否则 Picard 每迭代一次就再推进一个 dt，解会快数倍。
        conc_old = conc.copy()
        temp_old = temp.copy()

        for _ in range(picard):
            rho, drho, cp, k_cond, diff = props(conc, temp)
            rho_cp = rho * cp
            mass_face = (0.5 * (rho[:-1] + rho[1:])) * (
                0.5 * (diff[:-1] + diff[1:])
            )
            heat_face = 0.5 * (k_cond[:-1] + k_cond[1:])

            # 水分方程的左端是 d(rho*C)/dt，按链式法则其系数为 rho + C*rho'
            accum = rho + conc * drho

            lam = accum * weight * r_now**2
            fc = dt * mass_face * xif / h
            adv = dt * accum * r_now * r_dot * xi / 2.0
            alpha = dt * accum[n] * r_now * H_MASS
            lower, diag, upper, rhs = _assemble(
                lam, fc, adv, conc_old, alpha, c_inf, n
            )
            conc_new = thomas(lower, diag, upper, rhs)

            lam = rho_cp * weight * r_now**2
            fc = dt * heat_face * xif / h
            adv = dt * rho_cp * r_now * r_dot * xi / 2.0
            alpha = dt * r_now * H_HEAT
            lower, diag, upper, rhs = _assemble(
                lam, fc, adv, temp_old, alpha, t_inf, n
            )
            temp_new = thomas(lower, diag, upper, rhs)

            change = max(
                float(np.max(np.abs(conc_new - conc))),
                float(np.max(np.abs(temp_new - temp))),
            )
            temp, conc = temp_new, conc_new
            if change < picard_tol:
                break

        temp_hist[step] = temp
        conc_hist[step] = conc
        radius_hist[step] = r_now

        if conc[0] < C_DRY:
            dry_step = step
            break

    last = dry_step if dry_step is not None else n_steps
    return {
        "xi": xi,
        "times": times[: last + 1],
        "temp": temp_hist[: last + 1],
        "conc": conc_hist[: last + 1],
        "radius": radius_hist[: last + 1],
        "dry_time": float(times[last]) if dry_step is not None else None,
        "dt": dt,
        "n": n,
    }


def physical_positions(sol, positions_cm):
    """把贴体坐标解映射到给定的物理径向位置（cm）。"""
    xi = sol["xi"]
    times = sol["times"]
    n_pos = len(positions_cm)
    temp_out = np.empty((len(times), n_pos))
    conc_out = np.empty((len(times), n_pos))
    for k in range(len(times)):
        target = np.clip(
            np.asarray(positions_cm, float) / 100.0 / sol["radius"][k], 0.0, 1.0
        )
        temp_out[k] = np.interp(target, xi, sol["temp"][k])
        conc_out[k] = np.interp(target, xi, sol["conc"][k])
    return times, temp_out, conc_out


def main():
    parser = argparse.ArgumentParser(description="A 题 问题二/三/四 求解")
    parser.add_argument("--problem", type=int, required=True, choices=(2, 3, 4))
    parser.add_argument("--n", type=int, default=N_FINE)
    parser.add_argument("--dt", type=float, default=5.0)
    parser.add_argument("--tend", type=float, default=259200.0)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    shrink = args.problem == 4
    prop = "p4" if shrink else "p23"
    print(
        "求解问题%d: 物性=%s 收缩=%s N=%d dt=%g s tend=%g s"
        % (args.problem, prop, shrink, args.n, args.dt, args.tend)
    )
    sol = solve(
        prop=prop, shrink=shrink, n=args.n, dt=args.dt, tend=args.tend
    )
    print(
        "中心含水率首次低于 %.2f 的时刻: %s s = %.2f h = %.3f 天"
        % (
            C_DRY,
            sol["dry_time"],
            (sol["dry_time"] or 0) / 3600,
            (sol["dry_time"] or 0) / 86400,
        )
    )
    for hours in (0.5, 1, 2, 3):
        k = int(round(hours * 3600 / sol["dt"]))
        if k < len(sol["times"]):
            c = sol["conc"][k]
            print(
                "  t=%4.1f h: 中心C=%.4f  表面C=%.4f  中心T=%.2f  表面T=%.2f"
                % (
                    hours,
                    c[0],
                    c[-1],
                    sol["temp"][k][0],
                    sol["temp"][k][-1],
                )
            )

    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            xi=sol["xi"],
            times=sol["times"],
            temp=sol["temp"],
            conc=sol["conc"],
            radius=sol["radius"],
        )
        print("已保存: %s" % path)


if __name__ == "__main__":
    main()
