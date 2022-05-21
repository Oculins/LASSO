#  Optimization Methods Homework 3
#  Steihaug-Toint Conjugate Gradient
#  Author: Oculins
#  Reference: https://bicmr.pku.edu.cn/~wenzw/optbook/pages/newton/tCG.html

import numpy as np
from LR_util import *

class opts_tCG_struct:
    def __init__(self):
        self.kappa = 0
        self.theta = 0
        self.maxit = 0
        self.minit = 0

# 输入信息：迭代点x，梯度grad，信赖域半径Delta，包含算法参数的结构体opts
# 输出信息：x处下降方向eta，Heta为海森矩阵作用在eta上的结果，即& \nabla^{2} f(x) \eta &
def tCG(x, grad, A, b, mu, Delta, opts):
    if opts.kappa == -1:
        opts.kappa = 0.1   # 牛顿方程求解精度的参数
    if opts.theta == -1:
        opts.theta = 1     # 牛顿方程求精精度的参数
    if opts.maxit == -1:
        opts.maxit = len(x) # 最大迭代次数
    if opts.minit == -1:
        opts.minit = 5     # 最小迭代次数

    # 初始化
    theta = opts.theta
    kappa = opts.kappa
    eta = np.zeros([x.size, 1])
    Heta = eta
    r = grad
    e_Pe = 0
    r_r = np.dot(r.T, r)
    norm_r = np.sqrt(r_r)
    norm_r0 = norm_r

    mdelta = r
    d_Pd = r_r
    e_Pd = 0

    # 目标函数
    model_fun = np.dot(eta.T, grad) + 0.5 * np.dot(eta.T, Heta)
    model_value = 0

    # 迭代达到最大步数标记
    stop_tCG = 5

    j = 0
    for j in range(opts.maxit):
        Hmdelta = lr_hess(x, mdelta, A, b, mu)
        d_Hd = np.dot(mdelta.T, Hmdelta)
        alpha = r_r/d_Hd
        e_Pe_new = e_Pe + 2*alpha*e_Pd + alpha*alpha*d_Pd
        if d_Hd <= 0 or e_Pe_new >= Delta**2:
            tau = (-e_Pd + np.sqrt(e_Pd*e_Pd + d_Pd*(Delta**2 - e_Pe))) / d_Pd
            eta = eta - tau*mdelta
            Heta = Heta - tau*Hmdelta

            if d_Hd <= 0:
                stop_tCG = 1  # 以非正曲率退出
            else:
                stop_tCG = 2  # 以超出边界退出
            break

        # 更新迭代参数
        e_Pe = e_Pe_new
        new_eta = eta - alpha*mdelta
        new_Heta = Heta - alpha*Hmdelta

        new_model_value = np.dot(new_eta.T, grad) + 0.5 * np.dot(new_eta.T, new_Heta)
        if new_model_value >= model_value:
            stop_tCG = 6   # 目标函数值以非降终止
            break

        eta = new_eta
        Heta = new_Heta
        model_value = new_model_value

        r = r - alpha*Hmdelta
        r_rold = r_r
        r_r = np.dot(r.T, r)
        norm_r = np.sqrt(r_r)

        if j >= opts.minit and norm_r <= norm_r0*min(norm_r0**theta, kappa):
            if kappa < norm_r0**theta:
                stop_tCG = 3     # 线性收敛终止
            else:
                stop_tCG = 4     # 超线性收敛终止
            break

        beta = r_r/r_rold
        mdelta = r + beta*mdelta

        e_Pd = beta*(e_Pd + alpha*d_Pd)
        d_Pd = r_r + beta*beta*d_Pd

    iter = j

    return eta, Heta, iter, stop_tCG
