#  Optimization Methods Homework 3
#  Problem 4 Logistics Regression by L-BFRS
#  Author: Oculins
#  Reference: https://bicmr.pku.edu.cn/~wenzw/optbook/pages/lbfgs/demo_lr_lbfgs.html

import numpy as np
from libsvm.python.libsvm.svm import *
from libsvm.python.libsvm.svmutil import *
from libsvm.python.libsvm.svm import __all__ as svm_all
from LR_util import *
from tCG import *
from LineSearch import *
from L_BFGS import *
import matplotlib.pyplot as plt
import matplotlib as mpl

# a9a数据集测试
b, A = svm_read_problem('a9a_test.txt', True)
A = A.toarray()
b.resize((len(b), 1))
m, n = A.shape
mu = 1e-2 / m

opts = opts_struct()
opts.xtol = 1e-6
opts.gtol = 1e-6
opts.ftol = 1e-16
opts.rho1 = -1
opts.rho2 = -1
opts.maxit = 2000
opts.m = 5
opts.storeitr = -1
opts.record = 0
opts.itPrint = -1
x0 = np.zeros([n, 1])

x1, out1 = fminLBFGS_Loop(x0, A, b, mu, opts)

# a6a数据集测试
b, A = svm_read_problem('a6a_test.txt', True)
A = A.toarray()
b.resize((len(b), 1))
m, n = A.shape
mu = 1e-2 / m
x0 = np.zeros([n, 1])

x2, out2 = fminLBFGS_Loop(x0, A, b, mu, opts)

# ijcnn1数据集测试
b, A = svm_read_problem('ijcnn1_test.txt', True)
A = A.toarray()
b.resize((len(b), 1))
m, n = A.shape
mu = 1e-2 / m
x0 = np.zeros([n, 1])

x3, out3 = fminLBFGS_Loop(x0, A, b, mu, opts)

# 可视化
fig1 = plt.figure()
plt.yscale('log')
xx = np.arange(0, out1.iter, 10)
nrmG1 = []
for i in range(len(xx)):
    nrmG1.append(out1.nrmG[xx[i]])
plt.plot(xx, nrmG1, color='blue', label="a9a")
xx2 = np.arange(0, out2.iter, 10)
nrmG2 = []
for i in range(len(xx2)):
    nrmG2.append(out2.nrmG[xx2[i]])
plt.plot(xx2, nrmG2, color='red', label="a6a", linestyle='--')
xx3 = np.arange(0, out3.iter, 10)
nrmG3 = []
for i in range(len(xx3)):
    nrmG3.append(out3.nrmG[xx3[i]])
plt.plot(xx3, nrmG3, color='green', label="ijcnn1", linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel("iterations")
plt.ylabel('norm of gradient')
plt.show()
fig1.savefig('./L_BFGS_LR.png')
