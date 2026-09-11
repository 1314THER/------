#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2026 高教社杯全国大学生数学建模竞赛 A 题 —— 问题二 / 三 / 四求解器。

===============================================================================
一、与问题一的本质区别
===============================================================================
问题一的 rho、cp、k 是常数、D 只依赖 C，所以温度与水分两个方程解耦。
问题二起，附录 3、4 给出的物性都是含水率（乃至温度）的函数：

    附录 3（问题二、三）              附录 4（问题四）
    rho = 650 + 128C                  rho = 760 + 90C
    cp  = 1450 + 2736·C/(C+1)         cp  = 1850 + 2150·C/(C+1)
    k   = 0.21 + 0.38·C/(C+1)         k   = 0.12 + 0.20·C/(C+1)
    D   = 2.4e-3·e^(-0.45/C)·e^(-3850/T)   D = 4.2e-4·e^(-0.30/C)·e^(-3850/T)

于是：温度方程里的 rho、cp、k 依赖 C；水分方程里的 D 依赖 T。
两个方程**真正耦合**，必须联立迭代求解。

控制方程（圆柱坐标、一维径向、守恒形式）：
    热量：  d(rho·cp·T)/dt = (1/r)·d/dr( k·r·dT/dr )
    水分：  d(rho·C)/dt    = (1/r)·d/dr( rho·D·r·dC/dr )

定解条件与问题一相同：
    T(r,0)=28 °C，C(r,0)=2.55 kg/kg
    r=0：dT/dr=dC/dr=0（轴对称）
    r=R：-k dT/dr = h(T_s-T_inf)，-D dC/dr = hm(C_s-C_inf)

===============================================================================
二、烘房边界条件：预热段 + 恒温段
===============================================================================
附件 1 只覆盖 0～14400 s（4 h），而整个烘干要 61 h。故边界取分段形式：
    t ≤ 14400 s ：由附件 1 实测序列线性插值；
    t > 14400 s ：取常数 50 °C / 0.05 kg/kg。
依据：附件 1 末 1 h 的平均值是 49.9989 °C / 0.04999 kg/kg，末 1 h 的
温度线性拟合斜率仅 +0.003 °C/h——说明烘房已进入恒温恒湿状态，
取常数外推是数据支持的结果，不是主观假定（详见论文第 5.5 节）。

===============================================================================
三、问题四的动边界（收缩）
===============================================================================
附件 2 给出药材外半径随时间的变化：2.000 cm → 1.198 cm。
收缩使计算区域本身在变，属于动边界问题。处理办法是引入**贴体坐标**
    ξ = r / R(t) ∈ [0, 1]
把变区域 [0, R(t)] 映射成固定区域 [0,1]。由复合求导：
    ∂/∂t|_r = ∂/∂t|_ξ - (ξ·Ṙ/R)·∂/∂ξ ，  ∂/∂r = (1/R)·∂/∂ξ
代入后（以热量方程为例）：
    rho·cp·ξ·R²·∂T/∂t = ∂/∂ξ( k·ξ·∂T/∂ξ ) + rho·cp·ξ·R·Ṙ·∂T/∂ξ
右端最后一项是坐标变换带来的**对流项**：Ṙ<0（收缩）时它把表层的水分
与热量向内"挤压"。若令 Ṙ≡0，方程退化为问题二的形式。

Ṙ 由附件 2 的 PCHIP 插值**解析求导**得到，而不是对观测做数值差分——
差分会把测量噪声放大 1/Δt 倍（见论文 5.4 节的信噪比论证）。

===============================================================================
四、数值方法
===============================================================================
空间：节点中心有限体积，但在贴体坐标 ξ 上进行，控制体权重是 ∫ξ dξ。
时间：后向 Euler（无条件稳定）。
耦合：Picard 迭代——用上一轮的温度算水分方程、用上一轮的水分算温度方程，
      反复交替直到两个场都不再变化。
线性方程组：三对角，追赶法 O(N) 求解。

===============================================================================
五、怎么运行
===============================================================================
    PY=/Users/sqz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
    $PY 代码/solve_p234.py --problem 2 --dt 5 --save 输出/result2_data.npz
    $PY 代码/solve_p234.py --problem 3 --dt 5 --save 输出/result3_data.npz
    $PY 代码/solve_p234.py --problem 4 --dt 5 --save 输出/result4_data.npz
参数：--problem {2,3,4}（决定用哪套物性、是否收缩）
      --n 网格数、--dt 时间步、--tend 终止时刻、--save 输出路径
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openpyxl


# =============================================================================
# 0. 路径与常数
# =============================================================================

PROJECT = Path(__file__).resolve().parent.parent
ATT1 = PROJECT / "附件" / "附件1-烘房温度与水分浓度.xlsx"
ATT2 = PROJECT / "附件" / "附件2-药材半径.xlsx"

R0 = 0.02                 # 初始半径 2 cm                m
L_LEN = 0.25              # 药材长度 25 cm               m（体积换算用）
H_HEAT = 25.0             # 对流换热系数（附录 2）        W/(m²·K)
H_MASS = 8.0e-7           # 对流传质系数（附录 2）        m/s

T_INIT, C_INIT = 28.0, 2.55       # 初始温度与含水率
T_CONST, C_CONST = 50.0, 0.05     # 恒温干燥段工况（见模块 docstring）
T_PREHEAT_END = 14400.0           # 附件 1 覆盖的时间上限 s

C_DRY = 0.15              # 烘干判据：各处含水率低于该值 kg/kg

N_FINE = 100              # 默认网格单元数（ξ 方向，h = 0.01）


# =============================================================================
# 1. 物性关系：附录 3（问题二、三）与附录 4（问题四）
# =============================================================================

def props_p23(conc, temp):
    """附录 3 的物性，返回 (rho, drho/dC, cp, k, D)。

    参数
    ----
    conc : ndarray   含水率 C [kg/kg]
    temp : ndarray   温度 T [°C]

    返回
    ----
    rho   密度            kg/m³
    drho  密度对含水率的导数 dρ/dC = 128（常数），后面链式法则要用
    cp    比热容          J/(kg·K)
    k     热传导系数      W/(m·K)
    d     水分扩散系数    m²/s

    两个细节
    --------
    1. c = np.maximum(conc, 1e-12)：防除零。因为 e^(-0.45/C) 在 C→0 时
       会下溢，迭代中间步可能出现极小值。
    2. t_k = temp + 273.15：D 的 Arrhenius 项 e^(-3850/T) 里的 T 必须是
       **绝对温标 K**。若误把摄氏度代进去，D 会差几个数量级——
       这是本题最容易犯的单位错误。
    """
    c = np.maximum(conc, 1e-12)      # 防除零
    t_k = temp + 273.15              # °C → K
    rho = 650.0 + 128.0 * c
    drho = np.full_like(c, 128.0)    # d(650+128C)/dC = 128
    cp = 1450.0 + 2736.0 * c / (c + 1.0)
    k = 0.21 + 0.38 * c / (c + 1.0)
    d = 2.4e-3 * np.exp(-0.45 / c) * np.exp(-3850.0 / t_k)
    return rho, drho, cp, k, d


def props_p4(conc, temp):
    """附录 4 的物性（问题四专用），返回格式同 props_p23。

    与附录 3 的区别：系数不同，且 D 的前因子从 2.4e-3 降到 4.2e-4、
    指数从 e^(-0.45/C) 变成 e^(-0.30/C)。总体效果是问题四的扩散系数
    小得多，若不加收缩，烘干时长会长达 5.71 天（论文 §9.6 的对照）。
    """
    c = np.maximum(conc, 1e-12)
    t_k = temp + 273.15
    rho = 760.0 + 90.0 * c
    drho = np.full_like(c, 90.0)
    cp = 1850.0 + 2150.0 * c / (c + 1.0)
    k = 0.12 + 0.20 * c / (c + 1.0)
    d = 4.2e-4 * np.exp(-0.30 / c) * np.exp(-3850.0 / t_k)
    return rho, drho, cp, k, d


PROPERTY_SETS = {"p23": props_p23, "p4": props_p4}
# 用字符串选物性，方便 solve() 按题目编号切换，也方便外部脚本做灵敏度
# 分析时替换掉某一套物性（例如把 D 整体乘 0.8 或 1.2）。


# =============================================================================
# 2. 读附件数据
# =============================================================================

def _read(path):
    """读一个 xlsx 的第一个工作表，跳过表头与空行，返回行的列表。"""
    wb = openpyxl.load_workbook(path, data_only=True)
    return [
        r
        for r in wb.active.iter_rows(min_row=2, values_only=True)
        if r[0] is not None
    ]


# 模块导入时就把两个附件读进内存，避免每次调用重复读盘。
_AIR = _read(ATT1)
_AIR_T = np.array([r[0] for r in _AIR], float)       # 时间 s
_AIR_TEMP = np.array([r[1] for r in _AIR], float)    # 烘房温度 °C
_AIR_C = np.array([r[2] for r in _AIR], float)       # 烘房水分浓度 kg/kg

_RAD = _read(ATT2)
_RAD_T = np.array([r[0] for r in _RAD], float)       # 时间 s
_RAD_V = np.array([r[1] for r in _RAD], float) / 100.0   # 半径 cm → m


def air_conditions(t_query):
    """返回某个时刻的烘房工况 (T_inf, C_inf)。

    分段定义：t ≤ 14400 s 时对附件 1 线性插值；之后取恒温段常数。
    np.where 一次处理整个数组；np.interp 在 t > 14400 时会做线性外推，
    所以必须先判条件再取常数，不能直接插值。

    输入标量时返回两个 float；输入数组时返回两个数组。
    """
    tq = np.asarray(t_query, float)
    scalar = tq.ndim == 0            # 记录输入是不是标量，决定返回形式
    tq = np.atleast_1d(tq)           # 统一转成一维数组方便向量化
    temp = np.where(
        tq <= T_PREHEAT_END, np.interp(tq, _AIR_T, _AIR_TEMP), T_CONST
    )
    humid = np.where(
        tq <= T_PREHEAT_END, np.interp(tq, _AIR_T, _AIR_C), C_CONST
    )
    return (float(temp[0]), float(humid[0])) if scalar else (temp, humid)


# =============================================================================
# 3. PCHIP：保单调三次 Hermite 插值（给问题四的 R(t) 用）
# =============================================================================

class Pchip:
    """Fritsch--Carlson 单调三次 Hermite 插值（PCHIP）。

    为什么不能直接用线性插值或三次样条
    ----------------------------------
    问题四需要收缩速率 Ṙ 进入对流项。
    · 分段线性插值：Ṙ 在节点处跳变，等于把间断喂进方程；
    · 普通三次样条：会在数据上产生过冲，得到"先膨胀再收缩"的非物理结果；
    · PCHIP：既保证 C¹ 连续（导数连续），又保持数据单调、不产生过冲。

    构造方法（Fritsch--Carlson）
    ---------------------------
    记区间斜率 delta_i = (y_{i+1}-y_i)/h_i，节点导数记为 m_i。
    内部节点：
        若 delta_{i-1} 与 delta_i 异号（数据在此处有极值）→ m_i = 0，
            强制把拐点压平，避免插值曲线冲出去；
        否则取加权调和平均
            m_i = (w1+w2) / (w1/delta_{i-1} + w2/delta_i)
            w1 = 2h_i + h_{i-1}，w2 = h_i + 2h_{i-1}
        调和平均（而不是算术平均）正是"不过冲"的关键。
    端点：用单侧斜率 m_0 = delta_0、m_{n-1} = delta_{n-2}。
    有了节点值和节点导数，区间内就用标准的三次 Hermite 基函数求值。
    """

    def __init__(self, x, y):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.h = np.diff(self.x)                       # 各区间宽度 h_i
        delta = np.diff(self.y) / self.h               # 各区间斜率
        n = len(self.x)
        m = np.zeros(n)
        m[0] = delta[0]                                # 左端点单侧斜率
        m[-1] = delta[-1]                              # 右端点单侧斜率
        for i in range(1, n - 1):
            if delta[i - 1] * delta[i] <= 0.0:
                m[i] = 0.0                             # 极值点：压平
            else:
                w1 = 2.0 * self.h[i] + self.h[i - 1]
                w2 = self.h[i] + 2.0 * self.h[i - 1]
                m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
        self.m = m                                     # 保存节点导数备用

    def _locate(self, xq):
        """找到查询点落在哪个区间，返回区间左端索引（裁剪到 [0, n-2]）。"""
        return np.clip(
            np.searchsorted(self.x, xq, side="right") - 1, 0, len(self.x) - 2
        )

    def __call__(self, xq):
        """求插值值 R(t)。"""
        xq = np.asarray(xq, float)
        scalar = xq.ndim == 0
        xq = np.atleast_1d(xq)
        idx = self._locate(xq)
        h = self.h[idx]
        s = (xq - self.x[idx]) / h                     # 区间内的归一化坐标
        y0, y1 = self.y[idx], self.y[idx + 1]
        m0, m1 = self.m[idx], self.m[idx + 1]
        # 标准三次 Hermite 基函数展开
        out = (
            (2 * s**3 - 3 * s**2 + 1) * y0
            + (s**3 - 2 * s**2 + s) * h * m0
            + (-2 * s**3 + 3 * s**2) * y1
            + (s**3 - s**2) * h * m1
        )
        return float(out[0]) if scalar else out

    def deriv(self, xq):
        """求插值函数的导数 Ṙ(t)（解析求导，不是数值差分）。"""
        xq = np.asarray(xq, float)
        scalar = xq.ndim == 0
        xq = np.atleast_1d(xq)
        idx = self._locate(xq)
        h = self.h[idx]
        s = (xq - self.x[idx]) / h
        y0, y1 = self.y[idx], self.y[idx + 1]
        m0, m1 = self.m[idx], self.m[idx + 1]
        # 对上面的 Hermite 表达式关于 x 求导（链式法则带出 1/h 与 s'=1/h）
        out = (
            (6 * s**2 - 6 * s) * y0 / h
            + (3 * s**2 - 4 * s + 1) * m0
            + (-6 * s**2 + 6 * s) * y1 / h
            + (3 * s**2 - 2 * s) * m1
        )
        return float(out[0]) if scalar else out


# =============================================================================
# 4. 追赶法（同 solve_p1.py，此处不再重复注释）
# =============================================================================

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


# =============================================================================
# 5. 组装三对角系统（本文件最核心的一段）
# =============================================================================

def _assemble(lam, face, adv, value, alpha_const, far_value, n):
    """把一个"扩散 + 对流 + 第三类边界"的隐式方程组装成三对角系统。

    被组装的方程（逐节点）
    ----------------------
        lam_i·(u_i - value_i)                       ← 时间导数项（后向 Euler）
      + face_i 类项·(u_i - u_{i±1})                 ← 扩散项
      + adv_i·(u_{i+1} - u_{i-1})                   ← 对流项（贴体坐标带来）
      + alpha·(u_N - far_value)                     ← 表面第三类边界
      = 0
    统一记成  lower[i]·u_{i-1} + diag[i]·u_i + upper[i]·u_{i+1} = rhs[i]。

    参数
    ----
    lam : (n+1,)         时间导数项系数（含控制体权重、物性、R²）
    face : (n,)          界面扩散系数（含 dt、D 或 k、ξ_face、1/h）
    adv : (n+1,)         对流项系数（贴体坐标下 ∝ R·Ṙ·ξ；不收缩时全为 0）
    value : (n+1,)       本时间步**开始时刻**的场（右端项只用它）
    alpha_const : float  表面第三类边界的换热/传质系数（含 dt、R、h 或 hm）
    far_value : float    远场值（T_inf 或 C_inf）
    n : int              控制体个数

    三段的组装依据
    --------------
    · 中心节点 i=0：控制体是 [0, h/2]，没有左侧界面（对称性已经体现在
      weight[0]=h²/8 里），所以只有右界面的 face[0]。
    · 内部节点：左界面 face[i-1]、右界面 face[i]，对流项用中心差分
      ∫ξ∂ξu dξ ≈ ξ_i(u_{i+1}-u_{i-1})/2，故 lower 里出现 -adv、upper 里 +adv。
    · 表面节点 i=n：β=face[n-1] 是内界面扩散，γ=adv[n] 是对流项在边界上的
      单侧(迎风)离散。因为收缩时 Ṙ<0、流动方向朝内，迎风离散取 u_{n-1}-u_n。
    """
    lower = np.zeros(n + 1)
    diag = np.zeros(n + 1)
    upper = np.zeros(n + 1)
    rhs = np.zeros(n + 1)

    # ---- 中心节点：无左邻居，对流项因 ξ_0=0 而消失 ----
    diag[0] = lam[0] + face[0]
    upper[0] = -face[0]
    rhs[0] = lam[0] * value[0]

    # ---- 内部节点：扩散（左 bm / 右 bp）＋ 对流（±a） ----
    for i in range(1, n):
        bm = face[i - 1]
        bp = face[i]
        a = adv[i]
        lower[i] = -bm - a
        diag[i] = lam[i] + bm + bp
        upper[i] = -bp + a
        rhs[i] = lam[i] * value[i]

    # ---- 表面节点：第三类边界 + 迎风对流 ----
    beta = face[n - 1]
    gamma = adv[n]
    lower[n] = -beta + gamma
    diag[n] = lam[n] + alpha_const + beta - gamma
    rhs[n] = lam[n] * value[n] + alpha_const * far_value
    return lower, diag, upper, rhs


# =============================================================================
# 6. 主求解流程
# =============================================================================

def solve(
    prop="p23",
    shrink=False,
    n=N_FINE,
    dt=5.0,
    tend=259200.0,
    picard=6,
    picard_tol=1e-11,
):
    """求解问题二/三/四，返回完整时空解（贴体坐标）。

    参数
    ----
    prop : str        "p23" 用附录 3 物性（问题二、三）；"p4" 用附录 4
    shrink : bool     True 时启用附件 2 的收缩（问题四）
    n : int          贴体坐标 ξ 方向的网格单元数
    dt : float       时间步长 [s]
    tend : float     最长求解时间 [s]（达标即提前结束）
    picard : int     Picard 最大迭代次数
    picard_tol : float  Picard 收敛判据（温度与水分的变化量）

    返回字典
    --------
    xi          贴体网格节点
    times       实际推进到的时间序列（达标时会被截断）
    temp/conc   两个场的完整时空解（贴体坐标）
    radius      每个时刻的半径 R(t)
    dry_time    中心含水率首次低于 0.15 的时刻 [s]，未达标为 None
    dt, n       实际使用的步长与网格
    """
    props = PROPERTY_SETS[prop]       # 取物性函数

    # ---- 6.1 贴体网格 ----
    h = 1.0 / n                       # ξ 方向步长（ξ∈[0,1]，所以无量纲）
    xi = np.arange(n + 1) * h         # 节点 ξ_i
    xif = (np.arange(n) + 0.5) * h    # 界面 ξ_{i+1/2}

    # 控制体权重 w_i = ∫ξ dξ，与问题一的 w=∫r dr 同源，
    # 同样要单独修正中心与表面（否则守恒检验出现虚假残差）。
    weight = xi * h
    weight[0] = h * h / 8.0
    weight[n] = h * (1.0 - h / 4.0) / 2.0

    # 收缩时构造 R(t) 的 PCHIP 插值对象；不收缩时为 None。
    radius = Pchip(_RAD_T, _RAD_V) if shrink else None

    # ---- 6.2 初值 ----
    temp = np.full(n + 1, T_INIT)
    conc = np.full(n + 1, C_INIT)

    n_steps = int(round(tend / dt))
    times = np.arange(n_steps + 1) * dt
    temp_hist = np.empty((n_steps + 1, n + 1))   # 预留历史，便于事后取任意时刻
    conc_hist = np.empty((n_steps + 1, n + 1))
    radius_hist = np.empty(n_steps + 1)
    temp_hist[0] = temp
    conc_hist[0] = conc
    radius_hist[0] = radius(0.0) if shrink else R0

    dry_step = None                   # 记录首次达标的步号
    for step in range(1, n_steps + 1):
        t_now = step * dt
        t_inf, c_inf = air_conditions(t_now)               # 本时刻的烘房工况
        r_now = radius(t_now) if shrink else R0            # 当前半径 R(t)
        r_dot = radius.deriv(t_now) if shrink else 0.0     # 当前收缩速率 Ṙ(t)

        # ★ 关键：把时间步开始时的状态固定下来。Picard 迭代的右端项必须
        #   始终用它。若误用"上一轮迭代的结果"，每迭代一次就相当于又推进了
        #   一个 dt，最终解会比真解快好几倍。
        conc_old = conc.copy()
        temp_old = temp.copy()

        # ---- 6.3 Picard 迭代：交替求解两个耦合方程 ----
        for _ in range(picard):
            # 用当前 C、T 算出全场物性
            rho, drho, cp, k_cond, diff = props(conc, temp)
            rho_cp = rho * cp
            # 界面上的系数取相邻两节点的算术平均（面值，而不是节点值）
            mass_face = (0.5 * (rho[:-1] + rho[1:])) * (
                0.5 * (diff[:-1] + diff[1:])
            )
            heat_face = 0.5 * (k_cond[:-1] + k_cond[1:])

            # 水分方程左端是 ∂(ρC)/∂t。按链式法则：
            #   ∂(ρC)/∂t = (ρ + C·dρ/dC)·∂C/∂t
            # 所以时间导数项的系数是 rho + C·ρ'，而不是 rho。
            accum = rho + conc * drho

            # -------- 水分方程 --------
            # 贴体坐标下每一项的系数：
            #   lam   = accum·w·R²         （时间项：R² 来自 dξ 与 dr 的换算）
            #   fc    = dt·ρD·ξ_face/h     （扩散项）
            #   adv   = dt·accum·R·Ṙ·ξ/2   （对流项，∫ξ∂ξu dξ 的中心差分）
            #   alpha = dt·accum_N·R·hm    （表面传质边界）
            lam = accum * weight * r_now**2
            fc = dt * mass_face * xif / h
            adv = dt * accum * r_now * r_dot * xi / 2.0
            alpha = dt * accum[n] * r_now * H_MASS
            lower, diag, upper, rhs = _assemble(
                lam, fc, adv, conc_old, alpha, c_inf, n
            )
            conc_new = thomas(lower, diag, upper, rhs)

            # -------- 温度方程 --------
            # 结构完全相同，只是把 accum 换成 ρcp、把 ρD 换成 k、
            # 把 hm 换成 h。注意这里 alpha 不再乘额外系数。
            lam = rho_cp * weight * r_now**2
            fc = dt * heat_face * xif / h
            adv = dt * rho_cp * r_now * r_dot * xi / 2.0
            alpha = dt * r_now * H_HEAT
            lower, diag, upper, rhs = _assemble(
                lam, fc, adv, temp_old, alpha, t_inf, n
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

        # ---- 6.4 记录本时刻结果 ----
        temp_hist[step] = temp
        conc_hist[step] = conc
        radius_hist[step] = r_now

        # ---- 6.5 烘干判据 ----
        # 判据是"各处含水率低于 0.15"。由于水分由表及里脱除，中心处始终
        # 最高，所以"各处达标"等价于"中心达标"，只需检查 conc[0]。
        # 一旦达标就停止，省掉后面的计算。
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


# =============================================================================
# 7. 把贴体坐标解映射到固定物理位置
# =============================================================================

def physical_positions(sol, positions_cm):
    """把贴体坐标解映射到给定的物理径向位置（单位 cm）。

    为什么要映射
    ------------
    解是在 ξ=r/R(t) 上算的，网格节点随时间"跟着材料走"。但题目要求给的是
    固定物理位置（到中心轴 0、0.5、1.0 … cm）处的值，所以每个时刻都要做一次
    反查：某物理位置 r_k 对应的贴体坐标是 ξ_k = r_k / R(t)。

    np.clip(..., 0.0, 1.0) 的作用
    -----------------------------
    药材收缩后半径可能小于 1.5 cm，此时 r=2.0 cm 这些位置已经在药材外部，
    ξ_k > 1 没有物理意义。裁剪到 1.0 意味着把它们当作"表面"处理；
    输出到 Excel 时，调用方会依据 valid 掩码把这些格子留空。
    """
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


# =============================================================================
# 8. 命令行入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="A 题 问题二/三/四 求解")
    parser.add_argument("--problem", type=int, required=True, choices=(2, 3, 4))
    parser.add_argument("--n", type=int, default=N_FINE)
    parser.add_argument("--dt", type=float, default=5.0)
    parser.add_argument("--tend", type=float, default=259200.0)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    # 问题四才启用收缩、才用附录 4 的物性；问题二、三用附录 3。
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
    # 打印几个关键时刻的剖面，便于快速核对
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
