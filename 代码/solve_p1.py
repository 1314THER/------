#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2026 高教社杯全国大学生数学建模竞赛 A 题 —— 问题一求解器。

===============================================================================
一、要解的物理问题
===============================================================================
药材是半径 R=2 cm、长 L=25 cm 的圆柱。长径比 6.25，两端面只占侧面积的 8%，
所以忽略轴向输运，把问题降成"一维径向轴对称"问题：未知场只有
    T(r,t)  药材温度        [°C]
    C(r,t)  干基含水率      [kg 水 / kg 干料]
其中 r ∈ [0, R] 是到中心轴的距离。

控制方程（圆柱坐标下的守恒律，注意 1/r 和 r 这两个因子是柱坐标特有的）：
    热量：  rho*cp*dT/dt = (1/r) * d/dr ( k * r * dT/dr )
    水分：  dC/dt        = (1/r) * d/dr ( D(C) * r * dC/dr )

定解条件：
    初始：  T(r,0) = 28 °C，C(r,0) = 2.55 kg/kg
    中心：  dT/dr = dC/dr = 0                    （轴对称，不能写 Dirichlet！）
    表面：  -k dT/dr = h  * (T_s - T_inf(t))     （对流换热，第三类边界）
            -D dC/dr = hm * (C_s - C_inf(t))     （对流传质，第三类边界）

烘房空气状态 T_inf(t)、C_inf(t) 由附件 1 的实测序列线性插值得到。
物性取自附录 2，其中水分扩散系数是含水率的函数：
    D(C) = 7e-9 * exp(-0.89 / C)     [m^2/s]
含水率越低，D 越小（指数是负的），所以水分方程是非线性的。

===============================================================================
二、三处题面没给、必须自己定的建模选择
===============================================================================
1. 传质边界的密度口径：把 hm 与含水率差相乘要变成 kg/(m²·s) 的通量，需要
   一个参考密度。取药材干基密度 rho_d 为参考密度时，rho_d 在等式两侧约去，
   于是边界条件就是上面写的最简形式。本文采用该口径。
2. 忽略汽化潜热：完整的表面能量平衡还应含蒸发吸热项 L_v*j_w。题面没给 L_v；
   若按同一密度口径计入，表面温度会被压到约 6.1 °C，低于由空气状态算出的
   湿球温度 34.7 °C——这在物理上不可能。故判定题面参数体系隐含"忽略该项"。
3. 水分方程取简单扩散形式（而不是把 rho(C) 也塞进时间导数里）。
   理由与三种口径的定量对比见论文 §7.6.5。

问题一的特殊性：rho、cp、k 都是常数，且 D 只依赖 C 不依赖 T，
所以温度方程与水分方程**完全解耦**，可以各自独立推进（先后顺序无所谓）。
本文件就利用了这一性质。

===============================================================================
三、数值方法
===============================================================================
空间：节点中心有限体积法（FVM）。把 [0,R] 均分成 N 个控制体，节点在控制体
      中心，界面在节点之间。FVM 的好处是"离散格式本身守恒"——总水量的
      变化严格等于表面流出的水量，可以拿来当正确性判据。
时间：后向 Euler。无条件稳定，不怕 D(C) 在干燥后期剧烈变化。
非线性：D(C) 用 Picard 迭代（把系数滞后一步，反复解线性方程组直到收敛）。
线性方程组：三对角，用追赶法（Thomas 算法）在 O(N) 时间内解。

默认细网格 N=400（hx = 0.005 cm），时间步 1 s。题目要求输出
r = 0, 0.1, ..., 2.0 cm 共 21 个节点，这些点正好是细网格的子集
（每 20 个细网格点取一个），所以输出时不需要插值，避免引入额外误差。

===============================================================================
四、怎么运行
===============================================================================
系统自带的 python3 缺 openpyxl，请用打包运行时：
    PY=/Users/sqz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
    $PY 代码/solve_p1.py --save 输出/result1_data.npz --tables 输出/问题一-表1表2.md
命令行参数：
    --n      细网格单元数（默认 400，必须是 20 的整数倍）
    --dt     时间步长，秒（默认 1）
    --save   把解存成 .npz 与同名 .json（后者供 build_result1.mjs 生成 Excel）
    --tables 把表 1、表 2 导出成 Markdown
"""

from __future__ import annotations

import argparse                      # 解析命令行参数
from pathlib import Path             # 跨平台路径拼接

import numpy as np                   # 数组与向量化运算
import openpyxl                      # 读附件 1 的 xlsx


# =============================================================================
# 0. 路径与物理常数
# =============================================================================

PROJECT = Path(__file__).resolve().parent.parent
# __file__ 是本文件的路径；.resolve() 展开成绝对路径；.parent 是 代码/，
# 再 .parent 就是项目根目录。这样无论从哪里调用脚本都能找到附件。

ATTACHMENT1 = PROJECT / "附件" / "附件1-烘房温度与水分浓度.xlsx"

# ---- 附录 2 给定的物性参数（全部换成 SI 基本单位，避免单位混用）----
RHO = 820.0          # 药材体积密度            kg/m^3
CP = 2600.0          # 比热容                  J/(kg·K)
K_COND = 0.36        # 热传导系数              W/(m·K)
H_HEAT = 25.0        # 表面对流换热系数        W/(m^2·K)
H_MASS = 8.0e-7      # 表面对流传质系数        m/s

# ---- 几何尺寸（题面给的是 cm，这里统一换算成 m）----
RADIUS = 0.02        # 半径 2 cm  -> 0.02 m
LENGTH = 0.25        # 长度 25 cm -> 0.25 m（只在算体积/表面积时用到）

# ---- 初始条件 ----
T_INIT = 28.0        # 初始温度                °C
C_INIT = 2.55        # 初始干基含水率          kg/kg

# ---- 求解区间与时间步 ----
T_END = 1800.0       # 问题一求解到 1800 s（30 min）
DT = 1.0             # 时间步长 1 s

# ---- 网格 ----
N_FINE = 400         # 计算网格单元数（必须能被 N_OUT 整除）
N_OUT = 20           # 输出网格单元数：20 段 -> 21 个节点，间隔 0.1 cm

ALPHA = K_COND / (RHO * CP)
# 热扩散率 alpha = k/(rho*cp) = 1.689e-7 m²/s。
# 它和 D 相差约 34 倍，正是"温度很快铺满截面、水分只在表层移动"的原因。

REPORT_TIMES = [100, 300, 600, 900, 1200, 1500, 1800]   # 表 1/表 2 的时间点
REPORT_POS_CM = [0.0, 0.5, 1.0, 1.5, 2.0]               # 表 1/表 2 的位置点


# =============================================================================
# 1. 物性关系
# =============================================================================

def diffusion_coefficient(c):
    """附录 2 的水分扩散系数经验式 D(C) = 7e-9 * exp(-0.89/C)，单位 m²/s。

    参数
    ----
    c : ndarray 或 float
        干基含水率 [kg/kg]。

    返回
    ----
    与 c 同形状的 D 值 [m²/s]。

    说明
    ----
    np.maximum(c, 1e-12) 是防除零保护：含水率理论上恒大于 0，但在数值迭代
    的中间步里可能瞬时出现 0 或极小值，若不保护会得到 inf/nan 并污染整场。
    该保护只在 C < 1e-12 时才起作用，对本题 C ∈ [0.05, 2.55] 的取值区间
    没有任何影响。

    D 随 C 单调增：C 从 2.55 降到 0.15 时，D 从 4.94e-9 掉到 1.86e-11，
    跨了两个数量级。这就是干燥后期越来越慢的根本原因，也是必须用
    隐式格式（后向 Euler）而不是显式格式的原因。
    """
    return 7.0e-9 * np.exp(-0.89 / np.maximum(c, 1e-12))


# =============================================================================
# 2. 网格权重（有限体积法的核心）
# =============================================================================

def cell_weights(n_fine, hx):
    """返回每个控制体的径向权重 w_i，使得 ∫f dV ≈ 2πL * Σ w_i f_i。

    参数
    ----
    n_fine : int
        控制体个数 N。
    hx : float
        网格步长 [m]。

    返回
    ----
    w : ndarray, shape (N+1,)
        径向权重，满足 ∫_{CV_i} r dr = w_i。

    推导
    ----
    体积微元 dV = 2πL·r·dr，所以任意函数 f 的体积分是
        ∫ f dV = 2πL ∫ f·r dr ≈ 2πL Σ f_i ∫_{CV_i} r dr = 2πL Σ w_i f_i，
    即 w_i = ∫_{CV_i} r dr，其中 CV_i 是第 i 个控制体。

    · 内部节点 i：控制体 [r_i-hx/2, r_i+hx/2]，故
          w_i = [(r_i+hx/2)² - (r_i-hx/2)²]/2 = r_i·hx
    · 中心节点 i=0：控制体是 [0, hx/2]（左半边因对称性不存在），故
          w_0 = (hx/2)²/2 = hx²/8
    · 表面节点 i=N：控制体是 [R-hx/2, R]，故
          w_N = [R² - (R-hx/2)²]/2 = hx(R - hx/4)/2

    为什么必须单独处理这两端
    ------------------------
    r_0 = 0，如果偷懒用 w_i = r_i·hx 统一计算，中心节点的权重会被记成 0，
    相当于把中心控制体的"容量"当成零；表面节点则会被当成完整的 hx 宽度，
    权重放大近一倍。这两个错误叠加会让守恒检验出现约 1% 的虚假残差
    （见 conservation_report），从而掩盖真实的数值问题。
    """
    r = np.arange(n_fine + 1) * hx        # 节点坐标 r_i = i*hx，共 N+1 个
    w = r * hx                            # 先用内部节点公式填满整个数组
    w[0] = hx * hx / 8.0                  # 修正中心控制体权重
    w[n_fine] = hx * (RADIUS - hx / 4.0) / 2.0   # 修正表面控制体权重
    return w


# =============================================================================
# 3. 读附件 1
# =============================================================================

def load_air_conditions(path=ATTACHMENT1):
    """读取附件 1，返回 (时间, 烘房温度, 烘房水分浓度) 三个一维数组。

    附件 1 的格式：第 1 行是表头，A 列时间[s]、B 列温度[°C]、C 列水分浓度。
    共 241 行数据，间隔 60 s，覆盖 0～14400 s。

    这里**不做任何平滑**：噪声诊断（残差 σ=0.116 °C、高频交替型、无异常点）
    与"为什么不平滑"的量化论证见论文第 5 节。读取时只做两件事：
      1. 跳过表头（min_row=2）；
      2. 丢掉空行（row[0] is None），防止 Excel 末尾的空白被当成数据。
    """
    wb = openpyxl.load_workbook(path, data_only=True)
    # data_only=True 表示读取"数值"而不是单元格里的公式。
    rows = [
        row
        for row in wb.active.iter_rows(min_row=2, values_only=True)
        if row[0] is not None
    ]
    # values_only=True 直接给值而不是 Cell 对象；wb.active 取第一个工作表。
    t = np.array([row[0] for row in rows], dtype=float)       # 时间 [s]
    temp = np.array([row[1] for row in rows], dtype=float)    # 温度 [°C]
    humid = np.array([row[2] for row in rows], dtype=float)   # 水分浓度 [kg/kg]
    return t, temp, humid


# =============================================================================
# 4. 三对角方程组的追赶法
# =============================================================================

def thomas(lower, diag, upper, rhs):
    """追赶法（Thomas 算法）解三对角方程组 A x = rhs，复杂度 O(n)。

    参数
    ----
    lower : ndarray, shape (n,)
        次对角线。约定 lower[0] 不使用（第一个方程没有左邻居）。
    diag : ndarray, shape (n,)
        主对角线。
    upper : ndarray, shape (n,)
        超对角线。约定 upper[-1] 不使用。
    rhs : ndarray, shape (n,)
        右端项。

    返回
    ----
    x : ndarray, shape (n,)
        解向量。

    算法
    ----
    分两趟：
      前向消元（第 1 个循环）：把方程组化成上三角。cp[i]、dp[i] 分别是
          消元后第 i 行的"上对角元素/主对角元素"与"右端项/主对角元素"。
      回代（第 2 个循环）：从最后一个未知量往前逐一求出 x[i]。

    说明
    ----
    本题每个时间步要解两次三对角系统（温度一次、水分一次，水分还要迭代
    若干次），规模 N=400、步数 1800，用 O(N) 的追赶法比通用矩阵求逆
    快几个数量级，且没有引入额外误差。
    """
    n = len(diag)
    cp = np.empty(n)      # 消元后的上对角（存的是比值，不是原值）
    dp = np.empty(n)      # 消元后的右端（同上）

    # ---- 前向消元 ----
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i] * cp[i - 1]      # 消元后的主对角线元素
        cp[i] = upper[i] / m
        dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / m

    # ---- 回代 ----
    x = np.empty(n)
    x[-1] = dp[-1]                              # 最后一个未知量直接得到
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]         # 逐个往前代回去
    return x


# =============================================================================
# 5. 温度场推进一步（常系数线性问题，不需要迭代）
# =============================================================================

def step_temperature(temp, t_inf, dt, hx, r, rf):
    """用后向 Euler 把温度场从 t^n 推进到 t^{n+1}。

    参数
    ----
    temp : ndarray, shape (N+1,)
        当前时刻 t^n 的温度分布 [°C]。
    t_inf : float
        本时间步结束时刻的烘房温度 T_inf(t^{n+1}) [°C]。
        后向 Euler 的边界值取**新时刻**，这是格式的一部分，不能取旧时刻。
    dt, hx : float
        时间步长 [s]、空间步长 [m]。
    r : ndarray, shape (N+1,)
        节点坐标 r_i = i*hx。
    rf : ndarray, shape (N,)
        界面坐标 r_{i+1/2} = (i+0.5)*hx。

    返回
    ----
    temp_new : ndarray, shape (N+1,)
        t^{n+1} 时刻的温度分布。

    离散推导（以内部节点为例）
    --------------------------
    把方程 ρcp ∂T/∂t = (1/r) ∂/∂r(k r ∂T/∂r) 在第 i 个控制体上对 r·dr 积分：
        ρcp·w_i·dT_i/dt = k·r_{i+1/2}·(T_{i+1}-T_i)/hx
                          - k·r_{i-1/2}·(T_i-T_{i-1})/hx
    其中 w_i = ∫r dr 就是 cell_weights 给的权重。用后向 Euler 离散时间、
    再两边同除以 (ρcp·hx)，并记 a_t = k·dt/(ρcp·hx²)，得
        r_i·T_i^{n+1} + a_t·r_{i-1/2}(T_i^{n+1}-T_{i-1}^{n+1})
                      + a_t·r_{i+1/2}(T_i^{n+1}-T_{i+1}^{n+1}) = r_i·T_i^n
    这就是代码里 diag[i]=r_i+a_t(r_{i-1/2}+r_{i+1/2})、lower/upper=-a_t·r 的来历。

    中心节点（i=0）
    ---------------
    r=0 处方程有 1/r 奇点，用洛必达法则取极限：(1/r)∂_r(k r ∂_rT) → 2k∂_r²T。
    离散上更简单：控制体 [0,hx/2] 只有一个外界面 r_{1/2}=hx/2，
        ρcp·(hx²/8)·ΔT/dt = k·(hx/2)·(T_1-T_0)/hx
    →   ΔT = 4a_t(T_1-T_0)  →  (1+4a_t)T_0 - 4a_t·T_1 = T_0^n
    这正是 diag[0]=1+4a_t、upper[0]=-4a_t、rhs[0]=temp[0]。
    注意：中心是**对称条件**（∂T/∂r=0），不能写 Dirichlet 固定值；
    上式就是对对称条件的正确离散。

    表面节点（i=N）
    ---------------
    控制体 [R-hx/2, R]，外界面在 r=R，通过它的热量按第三类边界给出：
        ρcp·w_N·ΔT/dt = -k·r_{N-1/2}(T_N-T_{N-1})/hx + R·h·(T_inf - T_N)
    记 g_t = dt/(ρcp·hx·(R-hx/4))（注意 w_N = hx(R-hx/4)/2，所以
    dt/(ρcp·w_N) = 2g_t），并记
        p_t = 2g_t·R·h          （对流项系数）
        q_t = 2g_t·(R-hx/2)·k/hx（内部导热项系数）
    整理得 (1+p_t+q_t)T_N - q_t·T_{N-1} = T_N^n + p_t·T_inf，
    即代码中的 lower[n]=-q_t、diag[n]=1+p_t+q_t、rhs[n]=temp[n]+p_t*t_inf。
    """
    n = len(r) - 1                       # 控制体个数 N
    a_t = K_COND * dt / (RHO * CP * hx * hx)          # 内部节点无量纲系数
    g_t = dt / (RHO * CP * hx * (RADIUS - hx / 4.0))  # 表面节点公共因子
    p_t = 2.0 * g_t * RADIUS * H_HEAT                 # 对流项系数
    q_t = 2.0 * g_t * (RADIUS - hx / 2.0) * K_COND / hx   # 表面导热项系数

    lower = np.zeros(n + 1)              # 次对角线
    diag = np.zeros(n + 1)               # 主对角线
    upper = np.zeros(n + 1)              # 超对角线
    rhs = np.zeros(n + 1)                # 右端项

    # ---- 中心节点 i=0：对称边界，见 docstring ----
    diag[0] = 1.0 + 4.0 * a_t
    upper[0] = -4.0 * a_t
    rhs[0] = temp[0]

    # ---- 内部节点 i=1..N-1：标准三点格式 ----
    for i in range(1, n):
        lower[i] = -a_t * rf[i - 1]                        # 左邻居系数
        upper[i] = -a_t * rf[i]                            # 右邻居系数
        diag[i] = r[i] + a_t * (rf[i - 1] + rf[i])         # 自身系数
        rhs[i] = r[i] * temp[i]                            # 只含旧时刻的值

    # ---- 表面节点 i=N：第三类边界，见 docstring ----
    lower[n] = -q_t
    diag[n] = 1.0 + p_t + q_t
    rhs[n] = temp[n] + p_t * t_inf

    return thomas(lower, diag, upper, rhs)


# =============================================================================
# 6. 水分场推进一步（D 依赖 C，需要 Picard 迭代）
# =============================================================================

def step_moisture(conc, c_inf, dt, hx, r, rf, picard_max, picard_tol):
    """用"后向 Euler + Picard 迭代"把水分场从 t^n 推进到 t^{n+1}。

    参数
    ----
    conc : ndarray, shape (N+1,)
        当前时刻 t^n 的含水率分布 [kg/kg]。
    c_inf : float
        本时间步结束时刻的烘房水分浓度 C_inf(t^{n+1}) [kg/kg]。
    picard_max : int
        Picard 迭代的最大次数（默认 8）。
    picard_tol : float
        收敛判据（默认 1e-13）。

    返回
    ----
    c_new : ndarray, shape (N+1,)
        t^{n+1} 时刻的含水率分布。

    为什么要迭代
    ------------
    方程 ∂C/∂t = (1/r)∂_r(D(C)·r·∂_rC) 里，D 本身是 C 的函数，所以离散后
    得到的是一个**非线性**方程组。Picard（不动点）迭代的做法是：
        用上一轮解出的 C^(k) 算出 D^(k)，把方程"冻"成线性三对角系统，
        解出 C^(k+1)；比较 C^(k+1) 与 C^(k)，足够接近就停。
    由于 D 关于 C 变化平缓（指数函数），通常 2～4 次迭代就收敛到机器精度。

    与温度步的三个区别
    ------------------
    1. k/(ρcp) 换成了 D，所以系数 a_t 换成 dtb = dt·D_face/hx²；
    2. 因为方程里没有 ρcp，节点自身系数直接用 r_i（而不是 r_i 乘某个物性）；
    3. 界面上用的扩散系数取相邻两节点的算术平均（d_face），而不是节点值。
       这一点很重要：用节点值会造成离散格式不自洽，粗网格下表面含水率
       会有约 1.6% 的系统偏差（改用面值后降到 0.12%）。

    边界条件的处理与温度完全同构，只是把 h 换成 hm、把 T 换成 C。
    """
    n = len(r) - 1
    g_c = dt / (hx * (RADIUS - hx / 4.0))          # 表面节点公共因子（无 ρcp）
    p_c = 2.0 * g_c * RADIUS * H_MASS              # 对流传质项系数
    q_base = 2.0 * g_c * (RADIUS - hx / 2.0) / hx  # 表面扩散项系数（再乘 D_face）

    c_old = conc.copy()          # 本时间步**开始时刻**的值：右端项只能用这个
    c_new = conc.copy()          # 迭代初值：先用旧值起步
    for _ in range(picard_max):
        # ---- 用当前迭代解算出界面上的扩散系数 ----
        d_node = diffusion_coefficient(c_new)          # 各节点的 D
        d_face = 0.5 * (d_node[:-1] + d_node[1:])      # 各界面上的 D（算术平均）
        dtb = dt * d_face / (hx * hx)                  # 内部节点的无量纲系数
        b0 = dt * d_face[0] / (hx * hx)                # 中心节点用到的那一个

        lower = np.zeros(n + 1)
        diag = np.zeros(n + 1)
        upper = np.zeros(n + 1)
        rhs = np.zeros(n + 1)

        # ---- 中心节点：对称边界 ∂C/∂r=0 ----
        diag[0] = 1.0 + 4.0 * b0
        upper[0] = -4.0 * b0
        rhs[0] = c_old[0]                     # 注意 rhs 用 c_old，不是 c_new！

        # ---- 内部节点 ----
        for i in range(1, n):
            bm = dtb[i - 1] * rf[i - 1]       # 左界面系数（含界面 D 与 r_{i-1/2}）
            bp = dtb[i] * rf[i]               # 右界面系数
            lower[i] = -bm
            upper[i] = -bp
            diag[i] = r[i] + bm + bp
            rhs[i] = r[i] * c_old[i]

        # ---- 表面节点：第三类边界 -D ∂C/∂r = hm(C_s - C_inf) ----
        q_c = q_base * d_face[n - 1]          # 用最外层界面的 D
        lower[n] = -q_c
        diag[n] = 1.0 + p_c + q_c
        rhs[n] = c_old[n] + p_c * c_inf

        solved = thomas(lower, diag, upper, rhs)      # 解一次线性系统

        if np.max(np.abs(solved - c_new)) < picard_tol:
            c_new = solved                            # 收敛：接受并跳出
            break
        c_new = solved                                # 未收敛：继续迭代
    return c_new


# =============================================================================
# 7. 主求解流程
# =============================================================================

def solve(n_fine=N_FINE, dt=DT, tend=T_END, picard_max=8, picard_tol=1e-13):
    """求解问题一，返回包含全部结果的字典。

    参数
    ----
    n_fine : int      计算网格单元数 N（默认 400）
    dt : float        时间步长 [s]（默认 1）
    tend : float      求解终止时刻 [s]（默认 1800）
    picard_max : int  Picard 最大迭代次数
    picard_tol : float Picard 收敛判据

    返回字典的键
    ------------
    times       : (n_steps,)      时间序列 1,2,...,tend
    r_out_cm    : (N_OUT+1,)      输出位置 0,0.1,...,2.0 cm
    temp_out    : (n_steps+1,21)  温度场历史（第 0 行是 t=0 的初值）
    conc_out    : (n_steps+1,21)  水分场历史
    surf_flux   : (n_steps+1,)    表面瞬时通量 hm(C_s-C_inf)
    c_volume    : (n_steps+1,)    ∫C dV（守恒检验量）
    phys_volume : (n_steps+1,)    ∫rho_d·C dV（物理水量，用于对照）
    hx, dt      : float           实际使用的步长
    final_temp, final_conc : 细网格上的末态场
    """
    # ---- 7.1 建网格 ----
    hx = RADIUS / n_fine                                  # 空间步长
    r = np.arange(n_fine + 1) * hx                        # 节点坐标 (N+1,)
    rf = (np.arange(n_fine) + 0.5) * hx                   # 界面坐标 (N,)

    stride = n_fine // N_OUT
    if stride * N_OUT != n_fine:
        raise ValueError("n_fine 必须是 N_OUT=%d 的整数倍" % N_OUT)
    # 输出节点必须是细网格的子集，否则要插值。整数倍约束保证这一点。

    # ---- 7.2 读边界条件 ----
    t_air, temp_air, humid_air = load_air_conditions()

    n_steps = int(round(tend / dt))
    temp = np.full(n_fine + 1, T_INIT)     # 初始温度场：处处 28 °C
    conc = np.full(n_fine + 1, C_INIT)     # 初始水分场：处处 2.55 kg/kg

    times = np.arange(1, n_steps + 1) * dt                 # 1,2,...,1800
    temp_out = np.empty((n_steps + 1, N_OUT + 1))          # 预留历史输出
    conc_out = np.empty((n_steps + 1, N_OUT + 1))
    surf_flux = np.zeros(n_steps + 1)
    temp_out[0] = temp[::stride]           # [::stride] 每 stride 个取一个
    conc_out[0] = conc[::stride]           # 第 0 行存 t=0 的初值

    # ---- 7.3 守恒检验用的累积量（在细网格上积分）----
    vol_factor = 2.0 * np.pi * LENGTH       # 体积公式里的 2πL
    weights = cell_weights(n_fine, hx)      # 各控制体的径向权重
    c_volume = np.zeros(n_steps + 1)        # Q = ∫ C dV
    phys_volume = np.zeros(n_steps + 1)     # W = ∫ rho_d C dV
    c_volume[0] = vol_factor * float(np.sum(weights * conc))
    phys_volume[0] = vol_factor * RHO * float(
        np.sum(weights * conc / (1.0 + conc))
    )
    # 注意 rho_d = rho/(1+C)，所以 rho_d·C = rho·C/(1+C)。
    # Q 是"简单扩散形式"下真正守恒的量；W 是物理水量，用来对照说明
    # 两种密度口径的差别（W 不守恒是口径近似的后果，不是代码错误）。

    # ---- 7.4 时间推进 ----
    for step in range(1, n_steps + 1):
        t_now = step * dt
        # 附件 1 只给 60 s 间隔的离散点，任一时刻的边界值用线性插值取得。
        t_inf = float(np.interp(t_now, t_air, temp_air))
        c_inf = float(np.interp(t_now, t_air, humid_air))

        # 问题一里两个方程解耦，所以先后各推进一次即可，互不影响。
        # （问题二起 D 依赖 T、物性依赖 C，必须联立迭代，见 solve_p234.py）
        conc = step_moisture(
            conc, c_inf, dt, hx, r, rf, picard_max, picard_tol
        )
        temp = step_temperature(temp, t_inf, dt, hx, r, rf)

        # 抽稀保存：只记录题目要求的 21 个径向位置
        temp_out[step] = temp[::stride]
        conc_out[step] = conc[::stride]

        # 表面瞬时通量（用于后面做累积守恒检验）
        surf_flux[step] = H_MASS * (conc[-1] - c_inf)

        # 累积守恒量
        c_volume[step] = vol_factor * float(np.sum(weights * conc))
        phys_volume[step] = vol_factor * RHO * float(
            np.sum(weights * conc / (1.0 + conc))
        )

    return {
        "times": times,
        "r_out_cm": np.arange(N_OUT + 1) * (RADIUS / N_OUT) * 100.0,
        "temp_out": temp_out,
        "conc_out": conc_out,
        "surf_flux": surf_flux,
        "c_volume": c_volume,
        "phys_volume": phys_volume,
        "hx": hx,
        "dt": dt,
        "final_temp": temp,
        "final_conc": conc,
    }


# =============================================================================
# 8. 质量守恒检验
# =============================================================================

def conservation_report(res):
    """做水分守恒检验，返回各项数值。

    检验原理
    --------
    在有限体积离散下，"内部存量的减少"必须严格等于"表面累计流出的量"：
        Q(0) - Q(t) = A · ∫ hm·(C_s - C_inf) dt ，  A = 2πRL
    其中 Q = ∫C dV 是模型守恒量。因为格式本身守恒，
    这个残差应当只有舍入误差量级（1e-14 左右）。

    两个容易踩的坑
    --------------
    1. 累积通量必须用**左矩形和**（通量与后向 Euler 的新时刻一致），
       改成梯形法反而会引入 O(dt) 的假残差，看起来像"格式不守恒"。
    2. Q 的积分必须用 cell_weights 的正确权重（中心 hx²/8、表面
       hx(R-hx/4)/2），否则会凭空多出约 1% 的残差。

    另外报告物理水量 W = ∫rho_d·C dV。W 不守恒是"密度口径近似"的
    直接后果，论文中作为建模选择的不确定性如实披露。
    """
    area = 2.0 * np.pi * RADIUS * LENGTH      # 侧面积 A = 2πRL
    q = res["c_volume"]
    removed = float(q[0] - q[-1])             # 内部存量的总减少
    cumulative = area * res["dt"] * float(np.sum(res["surf_flux"][1:]))
    # 注意 [1:]：surf_flux[0]=0（初始时刻没有记录通量），
    # 且离散恒等式对应的是"通量取新时刻"的左矩形和。
    phys = res["phys_volume"]
    return {
        "q_start": float(q[0]),
        "q_end": float(q[-1]),
        "removed": removed,
        "cumulative_flux": cumulative,
        "relative_residual": (removed - cumulative) / removed,
        "phys_start": float(phys[0]),
        "phys_end": float(phys[-1]),
    }


# =============================================================================
# 9. 结果输出
# =============================================================================

def print_report(res):
    """把表 1（温度）和表 2（水分浓度）打印到终端。"""
    times = res["times"]
    pos_idx = [int(round(p / 0.1)) for p in REPORT_POS_CM]      # 位置 -> 列号
    time_idx = [int(round(x / res["dt"])) for x in REPORT_TIMES]  # 时间 -> 行号
    blocks = (
        ("表1  30 分钟内药材的温度", res["temp_out"], "°C"),
        ("表2  30 分钟内药材的水分浓度", res["conc_out"], "kg/kg"),
    )
    for title, hist, unit in blocks:
        print("\n=== %s  (单位: %s) ===" % (title, unit))
        header = "时间/s |" + "".join(
            "%10s" % ("%gcm" % p) for p in REPORT_POS_CM
        )
        print(header)
        print("-" * len(header))
        for ti in time_idx:
            row = "%6d |" % times[ti - 1]      # 第 ti 步对应 t=ti*dt，索引为 ti-1
            row += "".join("%10.4f" % hist[ti, pj] for pj in pos_idx)
            print(row)


def markdown_tables(res):
    """把表 1、表 2 渲染成 Markdown 文本，便于直接粘进论文。"""
    pos_idx = [int(round(p / 0.1)) for p in REPORT_POS_CM]
    time_idx = [int(round(x / res["dt"])) for x in REPORT_TIMES]
    out = [
        "# 问题一结果表",
        "",
        "由 `代码/solve_p1.py` 计算，细网格 $N=400$、$\\Delta t=1$ s，保留四位小数。",
        "",
    ]
    blocks = (
        ("表 1　30 分钟内药材的温度（单位：°C）", res["temp_out"]),
        ("表 2　30 分钟内药材的水分浓度（单位：kg/kg）", res["conc_out"]),
    )
    for title, hist in blocks:
        out.append("## " + title)
        out.append("")
        out.append(
            "| 时间/s | "
            + " | ".join("%g" % p for p in REPORT_POS_CM)
            + " |"
        )
        out.append("| --- |" + " --- |" * len(REPORT_POS_CM))
        for ti in time_idx:
            out.append(
                "| %d | " % res["times"][ti - 1]
                + " | ".join("%.4f" % hist[ti, pj] for pj in pos_idx)
                + " |"
            )
        out.append("")
    out.append("> 到药材中心的距离，单位 cm。完整结果见 `result1.xlsx`。")
    out.append("")
    return "\n".join(out)


# =============================================================================
# 10. 命令行入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="A 题问题一求解")
    parser.add_argument("--n", type=int, default=N_FINE, help="细网格单元数")
    parser.add_argument("--dt", type=float, default=DT, help="时间步长 s")
    parser.add_argument("--save", default=None, help="保存解的 .npz 路径")
    parser.add_argument("--tables", default=None, help="导出表 1/表 2 的 Markdown 路径")
    args = parser.parse_args()

    print("求解中: N=%d, dt=%g s, 时长 %g s" % (args.n, args.dt, T_END))
    res = solve(n_fine=args.n, dt=args.dt)
    print_report(res)

    # ---- 守恒检验 ----
    rep = conservation_report(res)
    print("\n=== 水分守恒检验（细网格精确积分）===")
    print("  模型守恒量 Q = ∫C dV   [m^3*kg/kg]")
    print("    初始        : %.8e" % rep["q_start"])
    print("    末态        : %.8e" % rep["q_end"])
    print("    减少        : %.8e" % rep["removed"])
    print("    表面累计流出: %.8e" % rep["cumulative_flux"])
    print("    相对残差    : %.3e" % rep["relative_residual"])
    print("  物理水量 W = ∫rho_d*C dV")
    print("    初始        : %.6f kg  (干料 %.6f kg)"
          % (rep["phys_start"], rep["phys_start"] / C_INIT))
    print("    末态        : %.6f kg" % rep["phys_end"])
    print("    脱除        : %.6f kg" % (rep["phys_start"] - rep["phys_end"]))

    # ---- 打印几个特征数，便于和论文对照 ----
    print(
        "\n度量: alpha=%.4e m^2/s, D(C0)=%.4e m^2/s, Bi=%.3f, Bi_m=%.3f"
        % (
            ALPHA,
            diffusion_coefficient(C_INIT),
            H_HEAT * RADIUS / K_COND,
            H_MASS * RADIUS / diffusion_coefficient(C_INIT),
        )
    )

    # ---- 存盘 ----
    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)   # 目录不存在就建
        np.savez_compressed(
            path,
            times=res["times"],
            r_out_cm=res["r_out_cm"],
            temp_out=res["temp_out"],
            conc_out=res["conc_out"],
        )
        print("\n已保存: %s" % path)

        # 同时导出 JSON，供 代码/build_result1.mjs 生成 result1.xlsx
        import json

        json_path = path.with_suffix(".json")           # 同名换扩展名
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "radius_cm": [round(float(v), 6) for v in res["r_out_cm"]],
                    "times": [int(v) for v in res["times"]],
                    # 去掉 t=0 的初始行，使第 i 行严格对应 times[i]
                    "temperature": np.round(res["temp_out"][1:], 4).tolist(),
                    "moisture": np.round(res["conc_out"][1:], 4).tolist(),
                },
                fh,
            )
        print("已保存: %s" % json_path)

    # ---- 导出论文表格 ----
    if args.tables:
        path = Path(args.tables)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown_tables(res), encoding="utf-8")
        print("已保存: %s" % path)


if __name__ == "__main__":
    # 只有"直接运行本文件"时才执行 main()；被 import 时不会执行，
    # 这样可以安全地在别的脚本里复用 solve()。
    main()
