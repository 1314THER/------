#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2026 A题 问题二/三/四 求解

跟问题一最大的区别是物性不再恒定。问题一里 rho、cp、k 都是常数、D 只看含水率，
所以温度和水分两个方程是解耦的，先后各推一步就行。问题二开始，附录3、4 给的
rho、cp、k 全都随含水率变化，D 还随温度变化，两个方程真正耦合，只能来回迭代。

   热量:  rho*cp*dT/dt = (1/r) * d/dr ( k * r * dT/dr )   （体积热容口径）
   水分:  dC/dt = (1/r) * d/dr ( D * r * dC/dr )          （经典 Fick，C 为干基含水率）

  四个问题共用上面两式；问题四的差别只在几何：半径 R(t) 随时间收缩，计算域
  随之缩小。把时间导数换到随材料运动的贴体坐标 xi = r/R(t) 上时有

      d/dt|_r = d/dt|_xi - (xi*Rdot/R) * d/dxi

  而对流项 u*d/dr（u = xi*Rdot 为骨架速度）恰好与坐标变换项抵消：

      d/dt|_r + u*d/dr = d/dt|_xi

  因此方程里**不出现**额外的"挤压"对流项，收缩只通过度量因子 1/R^2 与
  表面边界条件中的 R 进入模型。这一点与问题二、三的经典 Fick 口径完全一致，
  也避免了把体积密度 rho(C) 直接写进水分累积项（见论文 5.6 节）。

定解条件跟问题一一样：
   T(r,0) = 28 C,  C(r,0) = 2.55 kg/kg
   r = 0 : dT/dr = dC/dr = 0            （轴对称，不能写成定值边界）
   r = R : -k dT/dr = h*(T_s - T_inf)   （对流换热）
           -D dC/dr = hm*(C_s - C_inf)  （对流传质）

烘房条件是分两段的：附件1 只给到 14400 s，这段用线性插值；之后算恒温干燥段，
取 50 C 和 0.05 kg/kg。这两个数不是随手定的——附件1 最后 1 小时的平均值就是
49.9989 和 0.04999，而且这段的升温斜率只有 0.003 C/h，说明烘房已经稳住了。

问题四还要处理收缩。半径 R(t) 用附件2 的数据，走 PCHIP 插值（不能用差分，实测
相邻差分估出来的斜率信噪比只有 2:1，噪声会被放大）。为了不让网格跟着动，把计算
放到贴体坐标 xi = r/R(t) 上做；如上面推导，方程形式不变，只有度量因子与表面
边界条件带上 R。

数值方法还是老一套：节点中心有限体积 + 后向 Euler + Picard 迭代，
每步组装成三对角方程组，用追赶法解。

跑法：
   PY=/Users/sqz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
   $PY 代码/solve_p234.py --problem 3 --n 200 --dt 5 --save 输出/result3_data.npz
   注意 --n 不写就用下面 N_FINE 的默认值，论文里的结果是 200 跑的。
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import openpyxl

#文件
PROJECT = Path(__file__).resolve().parent.parent
ATT1 = PROJECT / "附件" / "附件1-烘房温度与水分浓度.xlsx"
ATT2 = PROJECT / "附件" / "附件2-药材半径.xlsx"

#常数
R0 = 0.02 #初始半径 2 cm
L_LEN = 0.25 #药材长度 25 cm 
H_HEAT = 25.0 #对流换热系数（附录 2)
H_MASS = 8.0e-7 #对流传质系数（附录 2)
T_INIT, C_INIT = 28.0, 2.55 #初始温度与含水率
T_CONST, C_CONST = 50.0, 0.05 #恒温干燥
T_PREHEAT_END = 14400.0 #附件 1 时间上限 s
C_DRY = 0.15 #烘干判据
N_FINE = 200 #默认网格单元数。论文结果就是 200 跑的；换成 100 烘干时间会差 0.16%（219340 vs 219685 s）

#物性关系
def props_p23(conc, temp):
    c = np.maximum(conc, 1e-12)      # 防除零
    t_k = temp + 273.15              # °C → K
    rho = 650.0 + 128.0 * c
    drho = np.full_like(c, 128.0)    # d(650+128C)/dC = 128
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


def _face_mean(a, kind):
    """由两侧节点值构造界面值。

    "arith"：算术平均（本文默认）；
    "harm" ：调和平均（分片常数介质的串联阻力）；
    "log"  ：对数平均（面内线性变化时的精确值）。
    """
    lo = np.maximum(a[:-1], 1e-30)
    hi = np.maximum(a[1:], 1e-30)
    if kind == "arith":
        return 0.5 * (lo + hi)
    if kind == "harm":
        return 2.0 * lo * hi / (lo + hi)
    if kind == "log":
        ratio = hi / lo
        close = np.isclose(ratio, 1.0)
        out = (hi - lo) / np.log(ratio)
        return np.where(close, lo, out)
    raise ValueError("unknown face_mean: %r" % kind)


#读取附件数据
def _read(path):
    #读取时跳过表头
    wb = openpyxl.load_workbook(path, data_only=True)
    return [
        r
        for r in wb.active.iter_rows(min_row=2, values_only=True)
        if r[0] is not None
    ]
_AIR = _read(ATT1)
_AIR_T = np.array([r[0] for r in _AIR], float) #时间
_AIR_TEMP = np.array([r[1] for r in _AIR], float) #烘房温度
_AIR_C = np.array([r[2] for r in _AIR], float) #烘房水分浓度
_RAD = _read(ATT2)
_RAD_T = np.array([r[0] for r in _RAD], float) #时间
_RAD_V = np.array([r[1] for r in _RAD], float) / 100.0 #半径


def air_conditions(t_query):
    tq = np.asarray(t_query, float)
    scalar = tq.ndim == 0 #输入是不是标量
    tq = np.atleast_1d(tq) #向量化
    #t≤14400 s时对附件1线性插值；np.where一次处理整个数组;np.interp在t>14400时会做线性外推       
    temp = np.where(
        tq <= T_PREHEAT_END, np.interp(tq, _AIR_T, _AIR_TEMP), T_CONST
    )
    humid = np.where(
        tq <= T_PREHEAT_END, np.interp(tq, _AIR_T, _AIR_C), C_CONST
    )
    return (float(temp[0]), float(humid[0])) if scalar else (temp, humid)



#追赶法（同 solve_p1.py，此处不再重复注释）
def thomas(lower, diag, upper, rhs):
    """追赶法解三对角方程组，O(n)。"""
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

#组装三对角系统
def _assemble(lam, face, value, alpha_const, far_value, n):
    #把一个"扩散 + 第三类边界"的隐式方程组装成三对角系统。
    """被组装的方程（逐节点）:lam_i·(u_i - value_i)（后向 Euler)+face_i 类项·(u_i - u_{i±1})(扩散项)+alpha·(u_N - far_value)(表面第三类边界)=0
    统一记成:lower[i]·u_{i-1} + diag[i]·u_i + upper[i]·u_{i+1} = rhs[i]。
    """

    #参数:
    #lam : (n+1,)         时间导数项系数(含控制体权重、物性、R²)
    #face : (n,)          界面扩散系数（含 dt、D 或 k、ξ_face、1/h)
    #value : (n+1,)       本时间步开始时刻的场（右端项只用它）
    #alpha_const : float  表面第三类边界的换热/传质系数（含 dt、R、h 或 hm)
    #far_value : float    远场值(T_inf 或 C_inf)
    #n : int              控制体个数

    #三段的组装依据
    #· 中心节点 i=0:控制体是[0, h/2],没有左侧界面(对称性已经体现在weight[0]=h²/8里），所以只有右界面的face[0]。
    #· 内部节点：左界面 face[i-1]、右界面 face[i]。
    #· 表面节点 i=n:β=face[n-1] 是内界面扩散，alpha_const 是第三类边界项。
    #  贴体坐标下对流项与坐标变换项精确抵消，因此不出现对流贡献。
    lower = np.zeros(n + 1)
    diag = np.zeros(n + 1)
    upper = np.zeros(n + 1)
    rhs = np.zeros(n + 1)
    #中心节点
    diag[0] = lam[0] + face[0]
    upper[0] = -face[0]
    rhs[0] = lam[0] * value[0]
    #内部节点
    for i in range(1, n):
        bm = face[i - 1]
        bp = face[i]
        lower[i] = -bm
        diag[i] = lam[i] + bm + bp
        upper[i] = -bp
        rhs[i] = lam[i] * value[i]
    #表面节点
    beta = face[n - 1]
    lower[n] = -beta
    diag[n] = lam[n] + alpha_const + beta
    rhs[n] = lam[n] * value[n] + alpha_const * far_value
    return lower, diag, upper, rhs

#保单调三次Hermite插值（问题四）
class Pchip:
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
        self.m = m #保存节点导数备用

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

#主求解流程
def solve(
    prop="p23",
    shrink=False,
    n=N_FINE,
    dt=5.0,
    tend=259200.0,
    picard=6,
    picard_tol=1e-11,
    face_mean="arith",
):
    """参数
    prop : str        "p23" 用附录 3 物性（问题二、三）；"p4" 用附录 4
    shrink : bool     True 时启用附件 2 的收缩（问题四）
    n : int          贴体坐标 ξ 方向的网格单元数
    dt : float       时间步长 [s]
    tend : float     最长求解时间 [s]（达标即提前结束）
    picard : int     Picard 最大迭代次数
    picard_tol : float  Picard 收敛判据（温度与水分的变化量）
    face_mean : str  界面物性的平均方式 "arith"（算术）/ "harm"（调和）/ "log"（对数）
    """
    #返回字典
    #xi          贴体网格节点
    #times       实际推进到的时间序列（达标时会被截断）
    #temp/conc   两个场的完整时空解（贴体坐标）
    #radius      每个时刻的半径 R(t)
    #dry_time    中心含水率首次低于 0.15 的时刻 [s]，未达标为 None
    #dt, n       实际使用的步长与网格
    props = PROPERTY_SETS[prop] #取物性函数

    #贴体网格
    h = 1.0 / n #ξ方向步长（无量纲）
    xi = np.arange(n + 1) * h #节点ξ_i
    xif = (np.arange(n) + 0.5) * h #界面ξ_{i+1/2}

    #控制体权重w_i=∫ξdξ，与问题一的w=∫rdr同源，还需要修正中心与表面
    weight = xi * h
    weight[0] = h * h / 8.0
    weight[n] = h * (1.0 - h / 4.0) / 2.0

    #收缩时构造R(t)的PCHIP插值对象；不收缩时为None。
    radius = Pchip(_RAD_T, _RAD_V) if shrink else None

    #初值
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

    dry_step = None #记录首次达标
    for step in range(1, n_steps + 1):
        t_now = step * dt
        t_inf, c_inf = air_conditions(t_now) 
        r_now = radius(t_now) if shrink else R0 #当前半径 R(t)

        #把时间步开始时的状态固定下来
        conc_old = conc.copy()
        temp_old = temp.copy()

        #Picard迭代
        for _ in range(picard):
            rho, _, cp, k_cond, diff = props(conc, temp)
            rho_cp = rho * cp
            heat_face = _face_mean(k_cond, face_mean)
            # 水分方程：四个问题统一用经典 Fick 口径（干基含水率，ρ 不进入）。
            # 收缩只通过度量因子 R² 与边界条件中的 R 进入，见文件头推导。
            mass_face = _face_mean(diff, face_mean)
            lam = weight * r_now**2
            fc = dt * mass_face * xif / h
            alpha = dt * r_now * H_MASS
            lower, diag, upper, rhs = _assemble(
                lam, fc, conc_old, alpha, c_inf, n
            )
            conc_new = thomas(lower, diag, upper, rhs)

            #温度方程
            lam = rho_cp * weight * r_now**2
            fc = dt * heat_face * xif / h
            alpha = dt * r_now * H_HEAT
            lower, diag, upper, rhs = _assemble(
                lam, fc, temp_old, alpha, t_inf, n
            )
            temp_new = thomas(lower, diag, upper, rhs)

            # 两个场的最大变化量都小于判据才算收敛
            change = max(
                float(np.max(np.abs(conc_new - conc))),
                float(np.max(np.abs(temp_new - temp))),
            )
            temp, conc = temp_new, conc_new
            if change < picard_tol:
                break

        #记录结果
        temp_hist[step] = temp
        conc_hist[step] = conc
        radius_hist[step] = r_now

        #烘干判据
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

#把贴体坐标解映射到固定物理位置
def physical_positions(sol, positions_cm):
    xi = sol["xi"]
    times = sol["times"]
    n_pos = len(positions_cm)
    temp_out = np.empty((len(times), n_pos))
    conc_out = np.empty((len(times), n_pos))
    for k in range(len(times)):
        # 当前半径 R(t) 把物理位置换算成贴体坐标
        target = np.clip(
            np.asarray(positions_cm, float) / 100.0 / sol["radius"][k], 0.0, 1.0
        )
        temp_out[k] = np.interp(target, xi, sol["temp"][k])
        conc_out[k] = np.interp(target, xi, sol["conc"][k])
    return times, temp_out, conc_out

#程序入口
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
    #快速核对
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
