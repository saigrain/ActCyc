import numpy as np
import pylab as pl
from scipy import interpolate, signal

def mirrorEnds(x):
    L = len(x)
    midPoint = (L+1)/2
    h1, h2 = x[:midPoint][::-1], x[midPoint:][::-1]
    h1 += x[0] - h1[-1]
    h2 ++ x[-1] - h2[0]
    y = np.concatenate((h1, x, h2))
    yinds = len(h1), len(h1) + len(x)
    return y, yinds

def getMaxima(x):
    x1 = np.roll(x,-1)
    x2 = np.roll(x,1)
    loc = np.where(((x >= x1) & (x>x2)) | ((x>x1) & (x>=x2)))[0]
    return loc

def emd(x, Niter=10, ImfMaxNum=100):
    Ntps = 4
    L = len(x)
    # Mirror ends of array to avoid edge effects
    xr,xinds = mirrorEnds(x)
    LL = len(xr)

    t = np.arange(len(xr))
    xorig = xr.copy()
    xresid = xr.copy()

    yoffset = xr.max() - x.min()
    pl.figure(1)
    pl.clf()
    pl.plot(t, xr, 'k-')
    
    print 'Empirical mode decomposition'
    endFlag = False
    for ct in range(ImfMaxNum):
        print 'IMF %d' % (ct+1)
        for n in range(Niter):
            print 'IMF %d, iter %d' % (ct+1,n+1)
            pl.figure(2)
            pl.clf()
            pl.plot(t,xr,'k-')
            # Find upper turning points
            l = getMaxima(xr)
            # Not confident about turning points beyond original edges
            f = np.where((abs(l-xinds[0]) > 2) & (abs(l-xinds[1]) > 2))[0]
            l = l[f]
            pl.plot(t[l], xr[l], 'r.')
            print len(l)
            if (len(l) < Ntps): # Not enough turning points left
                print 'Not enough turning points left (top)!'
                endFlag = True
            else:
                # Fit spline to upper envelope
                tck = interpolate.splrep(t[l], xr[l], s=0)
                yu = interpolate.splev(t, tck, ext = 0)
                pl.plot(t,yu,'r-')
            # Find lower turning points
            l = getMaxima(-xr)
            # Not confident about turning points beyond original edges
            f = np.where((abs(l-xinds[0]) > 2) & (abs(l-xinds[1]) > 2))[0]
            l = l[f]
            pl.plot(t[l], xr[l],'b.')
            print len(l)
            if (len(l) < Ntps): # Not enough turning points left
                print 'Not enough turning points left (bottom)!'
                endFlag = True
                break
            else:
                # Fit spline to lower envelope
                tck = interpolate.splrep(t[l], xr[l], s=0)
                yl = interpolate.splev(t, tck, ext = 0)
                pl.plot(t,yl,'b-')
            # Subtract midpoint of envelope
            m = (yu+yl)/2
            pl.plot(t, m, 'k--')
            xr -= m
            pl.plot(t, xr + 1, '-', c = 'grey')
            pl.xlim(t[xinds[0]], t[xinds[1]])
            pl.ylim(-0.5,1.5)
            # raw_input('Next iteration?')
        if endFlag == True:
            break
        # Store current envelope midpoint, extent and IMF
        m_ = m[xinds[0]:xinds[1]]
        yu_ = yu[xinds[0]:xinds[1]]
        yl_ = yl[xinds[0]:xinds[1]]
        if ct == 0:
            mx = m_.reshape((L,1))
            rx = (yu_-yl_).reshape((L,1))
            imf = xr.reshape((LL,1))
        else:
            mx = np.concatenate((mx, m_.reshape((L,1))), axis = 1)
            rx = np.concatenate((rx, (yu_-yl_).reshape((L,1))), axis = 1)
            imf = np.concatenate((imf, xr.reshape((LL,1))), axis = 1)
        xresid -= xr
        xr = xresid.copy()
        pl.figure(1)
        pl.plot(t, imf[:,-1] + ct+1, 'r-')
        pl.plot(t, xr + ct+1.5, 'b-')
        pl.xlim(t[xinds[0]], t[xinds[1]])
        pl.ylim(-0.5, ct+2)
        # raw_input('Next IMF?')
    imf = imf[xinds[0]:xinds[1],:]
    xresid = xresid[xinds[0]:xinds[1]]
    return imf, xresid, mx, rx

def freqAmp(f, t):
    a = signal.hilbert(f)
    x, y = a.real, a.imag
    x1 = x[:-1]
    y1 = y[:-1]
    t1 = t[:-1]
    x2 = x[1:]
    y2 = y[1:]
    t2 = t[1:]
    r = (np.sqrt(x1**2+y1**2)+np.sqrt(x2**2+y2**2))/2
    l = np.sqrt((x2-x1)**2+(y2-y1)**2)
    dtheta = l/r
    dt = t2-t1
    w = dtheta/dt
    freq = w / 2 / np.pi
    return freq, r

def test():
    t, f = np.genfromtxt('lightcurve_0012.txt').T
    f -= f.min()
    f /= f.max()
    f -= 0.5
    res = emd(f)
    imf1 = res[0][:,0].flatten()
    freq, amp = freqAmp(imf1, t)
    pl.figure(3)
    pl.clf()
    pl.scatter(t[:-1], freq, c = amp, s = 5, edgecolors='none')
    return


