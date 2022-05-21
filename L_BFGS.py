#  Optimization Methods Homework 3
#  Problem 4 L-BFRS
#  Author: Oculins
#  Reference: https://bicmr.pku.edu.cn/~wenzw/optbook/pages/lbfgs/fminLBFGS_Loop.html

import numpy as np
from LR_util import *
from tCG import *
from LineSearch import *

# 输出信息
class out_struct:
    def __init__(self):
        self.msg = '0'      # 标记是否收敛
        self.nrmG = []      # 迭代过程的梯度范数
        self.iter = 0       # 迭代步数
        self.f = []         # 函数值
        self.nfe = 0        # 调用目标函数次数
        self.xitr = []      # 优化变量x
        self.nge = 0        # 调用梯度次数
        self.nrmg = 0       # 迭代终止时的梯度范数

class opts_struct:
    def __init__(self):
        self.xtol = 0       # 针对优化变量的停机机制
        self.gtol = 0       # 针对梯度范数的停机机制
        self.ftol = 0       # 针对函数值的停机机制
        self.rho1 = 0       # 线搜索参数1
        self.rho2 = 0       # 线搜索参数2
        self.m = 0          # L-BFGS内存对数
        self.maxit = 0      # 最大迭代次数
        self.storeitr = 0   # 是否记录每一步迭代的x
        self.record = 0     # 是否输出迭代信息
        self.itPrint = 0    # 每隔几步输出迭代信息

# 双循环递归，返回下一个搜索方向-Hg
def LBFGS_Hg_Loop(dv, status, pos, perm, rho, SK, YK, ygk):
    q = dv
    alpha = np.zeros([status, 1])

    for di in range(status-1, -1, -1):
        k = perm[di]
        alpha[di] = np.dot(q.T, SK[:, k])*rho[k]
        YKa = YK[:, k]
        q = q - alpha[di] * YKa.reshape((len(YKa), 1))

    y = q / (rho[pos] * np.dot(ygk.T, ygk))

    for di in range(status):
        k = perm[di]
        beta = rho[k] * np.dot(y.T, YK[:, k])
        SKa = SK[:, k]
        y = y + SKa.reshape((len(SKa), 1)) * (alpha[di] - beta)

    return y

def fminLBFGS_Loop(x, A, b, mu, opts):

    if opts.gtol == -1:
        opts.gtol = 1e-6
    if opts.xtol == -1:
        opts.xtol = 1e-6
    if opts.ftol == -1:
        opts.ftol = 1e-16
    if opts.rho1 == -1:
        opts.rho1 = 1e-4
    if opts.rho2 == -1:
        opts.rho2 = 0.9
    if opts.m == -1:
        opts.m = 5
    if opts.maxit == -1:
        opts.maxit = 1000
    if opts.storeitr == -1:
        opts.storeitr = 0
    if opts.record == -1:
        opts.record = 0
    if opts.itPrint == -1:
        opts.itPrint = 1

    # 线搜索信息
    parsls = option()
    parsls.maxiter = -1
    parsls.display = 'N'
    parsls.ftol = opts.rho1
    parsls.gtol = opts.rho2
    parsls.xtol = -1
    parsls.stpmin = -1
    parsls.stpmax = -1

    xtol = opts.xtol
    ftol = opts.ftol
    gtol = opts.gtol
    maxit = opts.maxit
    storeitr = opts.storeitr
    m = opts.m
    record = opts.record
    itPrint = opts.itPrint

    f, g = lr_loss(x, mu, A, b)
    nrmx = np.linalg.norm(x)

    # 输出信息
    out = out_struct()
    out.nfe = 1
    out.nrmG = []
    out.f.append(f)

    if storeitr:
        out.xitr.append(x)

    n = len(x)
    # 最近m步的s(x的变化量)
    SK = np.zeros([n, m])
    # 最近m步的y(g的变化量)
    YK = np.zeros([n, m])
    istore = -1
    pos = 0
    status = 0
    perm = []
    rho = np.zeros(m)
    ygk = g

    out.msg = 'MaxIter'

    for iter in range(maxit):
        xp = x
        nrmxp = nrmx
        fp = f
        gp = g

        # 下降方向：第一次负梯度，之后用双循环法确定
        if istore == -1:
            d = -g
        else:
            d = LBFGS_Hg_Loop(-g, status, pos, perm, rho, SK, YK, ygk)

        workls = work_struct()
        workls.task = 1

        deriv = np.dot(d.T, g)
        normd = np.linalg.norm(d)

        # 线搜索
        stp = 1
        while 1:
            stp, f, deriv, parsls, workls = ls_csrch(stp, f, deriv, parsls, workls)
            if workls.task == 2:
                x = xp + stp * d
                f, g = lr_loss(x, mu, A, b)
                out.nfe = out.nfe + 1
                deriv = np.dot(g.T, d)
            else:
                break

        nrms = stp * normd
        diffX = nrms / max(nrmxp, 1)

        nrmG = np.linalg.norm(g)
        out.nrmg = nrmG
        out.f.append(f)
        out.nrmG.append(nrmG)
        if storeitr:
            out.xitr.append(x)
        nrmx = np.linalg.norm(x)

        # 判断是否收敛
        cstop = ((diffX < xtol) and (nrmG < gtol)) or (abs(fp - f) / (abs(fp) + 1)) < ftol

        if (record == 1) and (cstop or iter == 1 or iter == maxit or np.mod(iter, itPrint) == 0):
            if iter == 1 or np.mod(iter, 20 * itPrint) == 0 and iter != maxit and cstop == 0:
                print('%5s   %6s   %6s   %10s   %6s   %6s\n', iter, stp, f, diffX, nrmG, workls.task)
            print('%4d  %+2.1e  %+2.1e  %+2.1e  %+2.1e  %2d\n', iter, stp, f, diffX, nrmG, workls.task)

        if cstop:
            out.msg = 'Converge'
            break

        # 更新双循环法的参数
        ygk = g - gp
        s = x - xp
        if np.dot(ygk.T, ygk) > 1e-20:
            istore = istore + 1
            pos = np.mod(istore, m)
            YK[:, pos] = ygk.reshape(len(ygk))
            SK[:, pos] = s.reshape(len(s))
            rho[pos] = 1 / np.dot(ygk.T, s)
            if istore < m:
                status = istore + 1
                perm.append(pos)
            else:
                status = m
                perm_m = perm[1: m]
                perm_m.append(perm[0])
                perm = perm_m

    out.iter = iter
    out.nge = out.nfe

    return x, out


