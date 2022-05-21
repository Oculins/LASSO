#  Optimization Methods Homework 3
#  Line Search
#  Author: Oculins
#  Reference: https://bicmr.pku.edu.cn/~wenzw/optbook/pages/newton/ls_csrch.html

import numpy as np
import sys

def ls_dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):

    zero = 0.0
    p66 = 0.66
    two = 2.0
    three = 3.0

    sgnd = dp*(dx/abs(dx))

    if fp > fx:
        theta = three*(fx-fp)/(stp-stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s*np.sqrt((theta/s)**2 - (dx/s)*(dp/s))
        if stp < stx:
            gamma = -gamma
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p/q
        stpc = stx + r*(stp - stx)
        stpq = stx + ((dx/((fx - fp)/(stp - stx) + dx))/two)*(stp - stx)
        if abs(stpc - stx) < abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc)/two

        brackt = True

    elif sgnd < zero:
        theta = three * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))

        if stp > stx:
            gamma = -gamma

        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)

        if abs(stpc - stp) > abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq

        brackt = True

    elif abs(dp) < abs(dx):
        theta = three * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))

        gamma = s * np.sqrt(max(zero, (theta / s) ** 2 - (dx / s) * (dp / s)))
        if stp > stx:
            gamma = -gamma

        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
        if r < zero and gamma != zero:
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + (dp / (dp - dx)) * (stx - stp)

        if brackt:
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            if stp > stx:
                stpf = min(stp + p66 * (sty - stp), stpf)
            else:
                stpf = max(stp + p66 * (sty - stp), stpf)
        else:
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = min(stpmax, stpf)
            stpf = max(stpmin, stpf)

    else:
        if brackt:
            theta = three * (fp - fy) / (sty - stp) + dy + dp
            s = max(abs(theta), abs(dx), abs(dp))
            gamma = s * np.sqrt((theta / s) ** 2 - (dy / s) * (dp / s))
            if stp > sty:
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p / q
            stpc = stp + r * (sty - stp)
            stpf = stpc
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin

    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < zero:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp

    stp = stpf

    return stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt

class option:
    def __init__(self):
        self.maxiter = 0
        self.display = '0'
        self.ftol = 0
        self.gtol = 0
        self.xtol = 0
        self.stpmin = 0
        self.stpmax = 0

class work_struct:
    def __init__(self):
        self.task = 0
        self.msg = '0'
        self.brackt = False
        self.stage = 0
        self.ginit = 0
        self.gtest = 0
        self.gx = 0
        self.gy = 0
        self.finit = 0
        self.fx = 0
        self.fy = 0
        self.stx = 0
        self.sty = 0
        self.stmin = 0
        self.stmax = 0
        self.width = 0
        self.width1 = 0
        self.bestfx = 0
        self.bestgx = 0
        self.beststp = 0
        self.iter = 0

def ls_csrch(stp, f, g, options, work):
    zero = 0.0
    p5 = 0.5
    p66 = 0.66
    xtrapl = 1.1
    xtrapu = 4.0

    if work.task == 1:
        if options.maxiter == -1:
            options.maxiter = 20
        if options.display == 'N':
            options.display = 'no'
        if options.ftol == -1:
            options.ftol = 1e-3
        if options.gtol == -1:
            options.gtol = 0.2
        if options.xtol == -1:
            options.xtol = 1e-30
        if options.stpmin == -1:
            options.stpmin = 1e-20
        if options.stpmax == -1:
            options.stpmax = 1e5

        if stp < options.stpmin:
            work.task = -5
            work.msg = 'ERROR: STP .LT. STPMIN'
        if stp > options.stpmax:
            work.task = -5
            work.msg = 'ERROR: STP .GT. STPMAX'
        if g > zero:
            work.task = -5
            work.msg = 'ERROR: INITIAL G .GE. ZERO'
        if options.ftol < zero:
            work.task = -5
            work.msg = 'ERROR: FTOL .LT. ZERO'
        if options.gtol < zero:
            work.task = -5
            work.msg = 'ERROR: GTOL .LT. ZERO'
        if options.xtol < zero:
            work.task = -5
            work.msg = 'ERROR: XTOL .LT. ZERO'
        if options.stpmin < zero:
            work.task = -5
            work.msg = 'ERROR: STPMIN .LT. ZERO'
        if options.stpmax < options.stpmin:
            work.task = -5
            work.msg = 'ERROR: STPMAX .LT. STPMIN'

        if work.task == -5:
            print(work.msg)
            sys.exit()

        work.brackt = False
        work.stage = 1
        work.ginit = g
        work.gtest = options.ftol * work.ginit
        work.gx = work.ginit
        work.gy = work.ginit
        work.finit = f
        work.fx = work.finit
        work.fy = work.finit
        work.stx = zero
        work.sty = zero
        work.stmin = zero
        work.stmax = stp + xtrapu * stp
        work.width = options.stpmax - options.stpmin
        work.width1 = work.width / p5

        work.bestfx = f
        work.bestgx = g
        work.beststp = stp

        work.iter = 0
        work.task = 2
        work.msg = 'FG'

        return stp, f, g, options, work

    if work.task == 2:
        if work.bestfx > f:
            work.bestfx = f
            work.bestgx = g
            work.beststp = stp

    if work.iter >= options.maxiter:
        work.task = 0
        work.msg = 'EXCEED MAX ITERATIONS'
        stp = work.beststp
        f = work.bestfx
        g = work.bestgx

        return stp, f, g, options, work

    work.iter = work.iter + 1

    ftest = work.finit + stp * work.gtest

    if work.stage == 1 and f <= ftest and g >= zero:
        work.stage = 2

    if work.brackt and (stp < work.stmin or stp >= work.stmax):
        work.task = -1
        work.msg = 'WARNING: ROUNDING ERRORS PREVENT PROGRESS'
    if work.brackt and work.stmax - work.stmin <= options.xtol * work.stmax:
        work.task = -1
        work.msg = 'WARNING: XTOL TEST SATISFIED'
    if stp >= options.stpmax and f <= ftest and g <= work.gtest:
        work.task = -1
        work.msg = 'WARNING: STP = STPMAX'
    if stp <= options.stpmin and (f > ftest or g >= work.gtest):
        work.task = -1
        work.msg = 'WARNING: STP = STPMIN'
    if options.display == 'iter':
        print('stpmin: %4.3e \t stpmax: %4.3e \t stage: %d, brackt: %d \n', \
              work.stmin, work.stmax, work.stage, work.brackt)

    if f <= ftest and abs(g) <= options.gtol * (-work.ginit):
        work.task = 0
        work.msg = 'CONVERGENCE'

    if work.task == -1 or work.task == 0:
        return stp, f, g, options, work

    if work.stage == 1 and f <= work.fx and f >= ftest:
        fm = f - stp * work.gtest
        fxm = work.fx - work.stx * work.gtest
        fym = work.fy - work.sty * work.gtest
        gm = g - work.gtest
        gxm = work.gx - work.gtest
        gym = work.gy - work.gtest

        [work.stx, fxm, gxm, work.sty, fym, gym, stp, fm, gm, work.brackt] = \
            ls_dcstep(work.stx, fxm, gxm, work.sty, fym, gym, stp, \
                      fm, gm, work.brackt, work.stmin, work.stmax)
        work.fx = fxm + work.stx * work.gtest
        work.fy = fym + work.sty * work.gtest
        work.gx = gxm + work.gtest
        work.gy = gym + work.gtest

    else:
        [work.stx, work.fx, work.gx, work.sty, work.fy, work.gy, stp, f, g, work.brackt] = \
            ls_dcstep(work.stx, work.fx, work.gx, work.sty, work.fy, work.gy, \
                      stp, f, g, work.brackt, work.stmin, work.stmax)

    if work.brackt:
        if abs(work.sty - work.stx) >= p66 * work.width1:
            stp = work.stx + p5 * (work.sty - work.stx)
        work.width1 = work.width
        work.width = abs(work.sty - work.stx)

    if work.brackt:
        work.stmin = min(work.stx, work.sty)
        work.stmax = max(work.stx, work.sty)
    else:
        work.stmin = stp + xtrapl * (stp - work.stx)
        work.stmax = stp + xtrapu * (stp - work.stx)

    stp = max(stp, options.stpmin)
    stp = min(stp, options.stpmax)

    if work.brackt and (stp <= work.stmin or stp >= work.stmax) or \
            (work.brackt and work.stmax - work.stmin <= options.xtol * work.stmax):
        stp = work.stx

    work.task = 2
    work.msg = 'FG'

    return stp, f, g, options, work