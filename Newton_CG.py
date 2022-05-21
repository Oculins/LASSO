#  Optimization Methods Homework 3
#  Problem 1 Newton CG
#  Author: Oculins
#  Reference: https://bicmr.pku.edu.cn/~wenzw/optbook/pages/newton/fminNewton.html

import numpy as np
from LR_util import *
from tCG import *
from LineSearch import *

# 输出信息
class out_struct:
    def __init__(self):
        self.msg = '0'    # 标记是否收敛
        self.nrmG = []    # 迭代退出时的梯度范数
        self.iter = 0     # 迭代退出时的迭代步数、函数值
        self.f = 0        # 迭代退出时的目标函数值
        self.nfe = 0      # 调用原函数的次数

# 迭代参数
class opts_struct:
    def __init__(self):
        self.xtol = 0     # 针对优化变量的停机机制
        self.gtol = 0     # 针对梯度范数的停机机制
        self.ftol = 0     # 针对函数值的停机机制
        self.rho1 = 0     # 线搜索参数1
        self.rho2 = 0     # 线搜索参数2
        self.maxit = 0    # 主循环最大迭代次数
        self.verbose = 0  # 是否输出迭代信息
        self.itPrint = 0  # 每隔几步输出迭代信息

def fminNewton(x, A, b, mu, opts):

    if opts.gtol == -1:
        opts.gtol = 1e-6
    if opts.xtol == -1:
        opts.xtol = 1e-6
    if opts.ftol == -1:
        opts.ftol = 1e-12
    if opts.rho1 == -1:
        opts.rho1 = 1e-4
    if opts.rho2 == -1:
        opts.rho2 = 0.9
    if opts.maxit == -1:
        opts.maxit = 200
    if opts.verbose == -1:
        opts.verbose = 0
    if opts.itPrint == -1:
        opts.itPrint = 1

    # 复制参数
    maxit = opts.maxit
    verbose = opts.verbose
    itPrint = opts.itPrint
    xtol = opts.xtol
    gtol = opts.gtol
    ftol = opts.ftol

    # 线搜索参数设置
    parsls = option()
    parsls.maxiter = -1
    parsls.display = 'N'
    parsls.ftol = opts.rho1
    parsls.gtol = opts.rho2
    parsls.xtol = -1
    parsls.stpmin = -1
    parsls.stpmax = -1

    f, g = lr_loss(x, mu, A, b)
    nrmg = np.linalg.norm(g, 2)
    nrmx = np.linalg.norm(x, 2)

    # 输出信息
    out = out_struct()
    out.msg = 'MaxIter'
    out.nfe = 1
    out.nrmG.append(nrmg)

    # 截断共轭梯度法参数设置
    opts_tCG = opts_tCG_struct()
    opts_tCG.kappa = -1
    opts_tCG.theta = -1
    opts_tCG.maxit = -1
    opts_tCG.minit = -1

    # 迭代主循环
    for iter in range(maxit):
        fp = f
        gp = g
        xp = x
        nrmxp = nrmx

        opts_tCG.kappa = 0.1
        opts_tCG.theta = 1
        d, out1, out2, out3 = tCG(x, gp, A, b, mu, float('inf'), opts_tCG)

        # 线搜索信息
        workls = work_struct()
        workls.task = 1

        deriv = np.dot(d.T, g)
        normd = np.linalg.norm(d)

        # 调用ls_csrch进行线搜索
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
        xdiff = nrms / max(nrmxp, 1)
        nrmg = np.linalg.norm(g, 2)
        out.nrmG.append(nrmg)
        nrmx = np.linalg.norm(x, 2)
        out.nfe = out.nfe + 1
        fdiff = abs(fp - f) / (abs(fp) + 1)

        cstop = nrmg <= gtol or (abs(fdiff) <= ftol and abs(xdiff) <= xtol)

        if verbose >= 1 and (cstop or iter == 1 or iter == maxit or np.mod(iter, itPrint) == 0):
            if np.mod(iter, 20 * itPrint) == 0 and iter != maxit and cstop == 0:
                print('%5s   %10s   %8s   %8s   %8s\n', iter, f, nrmg, fdiff, xdiff)
            print(' %4d  %+8.7e  %+2.1e  %+2.1e  %+2.1e\n', iter, f, nrmg, fdiff, xdiff)
        if cstop:
            out.msg = 'Optimal'
            break

    out.iter = iter + 1
    out.f = f
    out.nrmg = nrmg


    return x, out





