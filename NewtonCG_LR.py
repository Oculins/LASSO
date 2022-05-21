#  Optimization Methods Homework 3
#  Problem 2 Logistics Regression by Newton CG
#  Author: Oculins
#  Reference: https://bicmr.pku.edu.cn/~wenzw/optbook/pages/newton/demo_lr_Newton.html

import numpy as np
from libsvm.python.libsvm.svm import *
from libsvm.python.libsvm.svmutil import *
from libsvm.python.libsvm.svm import __all__ as svm_all
from LR_util import *
from tCG import *
from LineSearch import *
from Newton_CG import *
import matplotlib.pyplot as plt
import matplotlib as mpl

# a9a数据集测试
b, A = svm_read_problem('a9a_test.txt', True)
A = A.toarray()
b.resize((len(b), 1))
m, n = A.shape
mu = 1e-2 / m

opts = opts_struct()
opts.xtol = 1e-8
opts.gtol = 1e-6
opts.ftol = 1e-16
opts.rho1 = -1
opts.rho2 = -1
opts.maxit = -1
opts.verbose = 0
opts.itPrint = -1
x0 = np.zeros([n, 1])

x1, out1 = fminNewton(x0, A, b, mu, opts)

# a6a数据集测试
b, A = svm_read_problem('a6a_test.txt', True)
A = A.toarray()
b.resize((len(b), 1))
m, n = A.shape
mu = 1e-2 / m
x0 = np.zeros([n, 1])

x2, out2 = fminNewton(x0, A, b, mu, opts)

# ijcnn1数据集测试
b, A = svm_read_problem('ijcnn1_test.txt', True)
A = A.toarray()
b.resize((len(b), 1))
m, n = A.shape
mu = 1e-2 / m
x0 = np.zeros([n, 1])

x3, out3 = fminNewton(x0, A, b, mu, opts)

# 可视化
fig1 = plt.figure()
plt.yscale('log')
xx = np.linspace(0, out1.iter, out1.iter+1)
plt.plot(xx, out1.nrmG, color='blue', label="a9a", marker='o')
xx2 = np.linspace(0, out2.iter, out2.iter+1)
plt.plot(xx2, out2.nrmG, color='red', label="a6a", linestyle='--', marker='x')
xx3 = np.linspace(0, out3.iter, out3.iter+1)
plt.plot(xx3, out3.nrmG, color='green', label="ijcnn1", linestyle='-.', marker='s')
plt.legend(loc='upper right')
plt.xlabel("iterations")
plt.ylabel("norm of gradient")
plt.show()
fig1.savefig('./Newton_CG.png')

