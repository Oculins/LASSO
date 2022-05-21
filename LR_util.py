#  Optimization Methods Homework 3
#  loss and Hessian
#  Author: Oculins

import numpy as np

# 逻辑回归的损失函数，作为优化问题的目标函数。输出函数值和梯度
def lr_loss(x, mu, A, b):
    m, n = A.shape
    Ax = np.dot(A, x)
    Atran = A.T
    expba = np.exp(- b * Ax)
    f = np.sum(np.log(1 + expba))/m + mu * np.linalg.norm(x, 2)**2
    g = np.dot(Atran, (b/(1+expba) - b))/m + 2*mu*x

    return f, g

# 目标函数的海森矩阵，要求提供当前优化变量x和方向u，返回海森矩阵在x处作用在方向u上的值
# 即 $ \nabla^{2} f(x) u  $
def lr_hess(x, u, A, b, mu):
    m, n = A.shape
    Ax = np.dot(A, x)
    Atran = A.T
    expba = np.exp(- b * Ax)
    p = 1/(1+expba)
    w = p*(1-p)
    H = np.dot(Atran, w * np.dot(A, u))/m + 2*mu*u

    return H