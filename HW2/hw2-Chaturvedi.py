import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as sci
import sympy as sp
from sympy import *
import time
import tqdm as tqdm

dat = pd.read_csv("hw2_fitting.dat", sep=" ", header=None)
dat = pd.DataFrame(dat)

freq = dat[0]
phi = dat[1]
phi_err = dat[2]

fig, ax = plt.subplots()
ax.scatter(freq,phi,c="k", s=5)
ax.errorbar(freq, phi, yerr=phi_err, c="k", elinewidth=0.5, ls="none")
ax.set_xlabel("$\\nu$")
ax.set_ylabel("$\phi$")
ax.set_title("Frequencies and their respective line strengths")
fig.savefig("rawData.jpg", dpi=300)



########################################################################################################################

def Lor(frequency,alpha_l,freq_0):
    y = (1/np.pi)*(alpha_l)/((frequency-freq_0)**2+(alpha_l)**2)
    return y

x, y, z, p = symbols('x y z p')

lor = (1 / np.pi) * (x) / ((y - z) ** 2 + (x) ** 2)
a = diff(lor, x)
b = diff(lor, x, 2)
c = diff(lor, z)
d = diff(lor, z, 2)
e = diff(lor, x, z)

def lorPrimeAlphaL(frequency,alpha_l,freq_0):
    valList = []
    for i in range(len(frequency)):
        values = a.subs([(x, alpha_l), (y, frequency[i]), (z, freq_0)])
        valList.append(N(values))
    return np.array(valList)

def lorPrimePrimeAlphaL(frequency,alpha_l,freq_0):
    valList = []
    for i in range(len(frequency)):
        values = b.subs([(x, alpha_l), (y, frequency[i]), (z, freq_0)])
        valList.append(N(values))
    return np.array(valList)

def lorPrimeFreq(frequency,alpha_l,freq_0):
    valList = []
    for i in range(len(frequency)):
        values = c.subs([(x, alpha_l), (y, frequency[i]), (z, freq_0)])
        valList.append(N(values))
    return np.array(valList)

def lorPrimePrimeFreq(frequency,alpha_l,freq_0):
    valList = []
    for i in range(len(frequency)):
        values = d.subs([(x, alpha_l), (y, frequency[i]), (z, freq_0)])
        valList.append(N(values))
    return np.array(valList)

def lorPrimePrimeAlphaLFreq(frequency,alpha_l,freq_0):
    valList = []
    for i in range(len(frequency)):
        values = e.subs([(x, alpha_l), (y, frequency[i]), (z, freq_0)])
        valList.append(N(values))
    return np.array(valList)

def Gau(frequency, alpha_g, freq_0):
    frequency = np.array(frequency).astype(float)
    alpha_g = float(alpha_g)
    freq_0 = float(freq_0)
    y = (1/alpha_g)*np.sqrt((np.log(2)/np.pi))*np.exp((-np.log(2)*(frequency-freq_0)**2)/alpha_g**2)
    return y

gau = (1 / x) * sp.sqrt((sp.log(2) / sp.pi)) * sp.exp((-sp.log(2) * (y - z) ** 2) / x ** 2)

f = diff(gau, x)
g = diff(gau,x,2)
h = diff(gau, z)
ii = diff(gau,z,2)
j = diff(gau, x, z)

def GauPrimeAlphaG(frequency, alpha_g, freq_0):
    valList = []
    for i in range(len(frequency)):
        values = f.subs([(x, alpha_g), (y, frequency[i]), (z, freq_0)])
        valList.append(N(values))
    return np.array(valList)

def GauPrimePrimeAlphaG(frequency, alpha_g, freq_0):
    valList = []
    for i in range(len(frequency)):
        values = g.subs([(x,alpha_g),(y,frequency[i]),(z,freq_0)])
        valList.append(N(values))
    return np.array(valList)

def GauPrimeFreq(frequency, alpha_g, freq_0):
    valList = []
    for i in range(len(frequency)):
        values = h.subs([(x, alpha_g), (y, frequency[i]), (z, freq_0)])
        valList.append(N(values))
    return np.array(valList)

def GauPrimePrimeFreq(frequency, alpha_g, freq_0):
    valList = []
    for i in range(len(frequency)):
        values = ii.subs([(x,alpha_g),(y,frequency[i]),(z,freq_0)])
        valList.append(N(values))
    return np.array(valList)

def GauPrimePrimeAlphaGFreq(frequency, alpha_g, freq_0):
    valList = []
    for i in range(len(frequency)):
        values = j.subs([(x, alpha_g), (y, frequency[i]), (z, freq_0)])
        valList.append(N(values))
    return np.array(valList)

########################################################################################################################

Lor_fit = sci.curve_fit(Lor, freq, phi, sigma=phi_err, method='lm')
Lor_freq_0, Lor_alpha = Lor_fit[0]
Lor_freq_0_err, Lor_alpha_err = np.sqrt(np.diag(Lor_fit[1]))

Gau_fit = sci.curve_fit(Gau, freq, phi, sigma=phi_err, method='lm')
Gau_freq_0, Gau_alpha = Gau_fit[0]
Gau_freq_0_err, G_alpha_err = np.sqrt(np.diag(Gau_fit[1]))

########################################################################################################################

def chiSq(func, y, x, y_error, param1, param2):
    chiSquared = np.sum(((y-func(x,param1,param2))/y_error)**2)
    return chiSquared

def LevenbergMarquardt(func, dfuncdparam1, dfuncdparam2, d2funcd2param1, d2funcd2param2, d2funcdparam1dparam2,
                       x, y, y_error, param1, param2):
    initChiSq = chiSq(func, y, x, y_error, param1, param2)
    LambdaVal = 0.001

    chiSqPrimeParam1 = -2*np.sum(((y - func(x, param1, param2))/(y_error**2.))*dfuncdparam1(x, param1, param2))
    chiSqPrimeParam2 = -2*np.sum(((y - func(x, param1, param2))/(y_error**2.))*dfuncdparam2(x, param1, param2))

    chiSqPrimePrimeParam1 = 2* np.sum((1/ y_error ** 2) * (dfuncdparam1(x, param1, param2) ** 2 - (y - func(x, param1, param2)) * d2funcd2param1(x, param1, param2)))
    chiSqPrimePrimeParam2 = 2* np.sum((1/ y_error ** 2) * (dfuncdparam2(x, param1, param2) ** 2 - (y - func(x, param1, param2)) * d2funcd2param2(x, param1, param2)))
    chiSqPrimePrimeParam1Param2 = 2* np.sum((1/ y_error ** 2) * (dfuncdparam1(x, param1, param2) * dfuncdparam2(x, param1, param2) - (y - func(x, param1, param2)) * d2funcdparam1dparam2(x, param1, param2)))


    Beta = -0.5*np.array([chiSqPrimeParam1, chiSqPrimeParam2])
    Alpha_LM = 0.5*np.array([[chiSqPrimePrimeParam1 * (1 + LambdaVal), chiSqPrimePrimeParam1Param2],
                             [chiSqPrimePrimeParam1Param2, chiSqPrimePrimeParam2 * (1 + LambdaVal)]])


    Beta = Matrix(Beta)
    Alpha_LM = Matrix(Alpha_LM)

    Sol = np.array(Alpha_LM.LUsolve(Beta))


    delParam1, delParam2 = Sol[0],Sol[1]
    param1_new, param2_new = param1 + delParam1[0], param2 + delParam2[0]

    chiSqUpdated = chiSq(func, y, x, y_error, param1_new, param2_new)

    #breakpoint()
    while abs(chiSqUpdated-initChiSq)/initChiSq > 10**(-6):
        if chiSqUpdated >= initChiSq:
            LambdaVal = LambdaVal*10

            Alpha_LM = 0.5*np.array([[chiSqPrimePrimeParam1*(1+LambdaVal),  chiSqPrimePrimeParam1Param2],
                                     [chiSqPrimePrimeParam1Param2, chiSqPrimePrimeParam2*(1+LambdaVal)]])


            Beta = Matrix(Beta)
            Alpha_LM = Matrix(Alpha_LM)


            Sol = np.array(Alpha_LM.LUsolve(Beta))
            delParam1, delParam2 = Sol[0],Sol[1]
            param1_new, param2_new = param1 + delParam1[0], param2 + delParam2[0]

            chiSqUpdated = chiSq(func, y, x, y_error, param1_new, param2_new)
            #chiSqFinal = chiSqUpdated


            #print("b")



        elif chiSqUpdated < initChiSq:
            LambdaVal = LambdaVal/10
            param1, param2 = param1+delParam1[0], param2+delParam2[0]

            chiSqPrimeParam1 = -2 * np.sum(((y - func(x, param1, param2)) / (y_error ** 2.)) * dfuncdparam1(x, param1, param2))
            chiSqPrimeParam2 = -2 * np.sum(((y - func(x, param1, param2)) / (y_error ** 2.)) * dfuncdparam2(x, param1, param2))

            chiSqPrimePrimeParam1 = 2 * np.sum((1 / y_error ** 2) * (dfuncdparam1(x, param1, param2) ** 2 - (y - func(x, param1, param2)) * d2funcd2param1(x, param1,param2)))
            chiSqPrimePrimeParam2 = 2 * np.sum((1 / y_error ** 2) * (dfuncdparam2(x, param1, param2) ** 2 - (y - func(x, param1, param2)) * d2funcd2param2(x, param1,param2)))
            chiSqPrimePrimeParam1Param2 = 2 * np.sum((1 / y_error ** 2) * (dfuncdparam1(x, param1, param2) * dfuncdparam2(x, param1, param2) - (y - func(x, param1, param2)) * d2funcdparam1dparam2(x, param1, param2)))

            Beta = -0.5 * np.array([chiSqPrimeParam1, chiSqPrimeParam2])
            Alpha_LM = 0.5 * np.array([[chiSqPrimePrimeParam1 * (1 + LambdaVal), chiSqPrimePrimeParam1Param2],
                                       [chiSqPrimePrimeParam1Param2, chiSqPrimePrimeParam2 * (1 + LambdaVal)]])


            Beta = Matrix(Beta)
            Alpha_LM = Matrix(Alpha_LM)

            Sol = np.array(Alpha_LM.LUsolve(Beta))

            delParam1, delParam2 = Sol[0],Sol[1]
            param1_new, param2_new = param1 + delParam1[0], param2 + delParam2[0]

            initChiSq = chiSqUpdated
            chiSqUpdated = chiSq(func, y, x, y_error, param1_new, param2_new)
            #chiSqFinal = chiSqUpdated

            #print("a")

    Alpha_LM = 0.5 * np.array([[chiSqPrimePrimeParam1, chiSqPrimePrimeParam1Param2],[chiSqPrimePrimeParam1Param2,
                                                                                     chiSqPrimePrimeParam2]])
    Alpha_LM = Matrix(Alpha_LM)
    Cov = Alpha_LM.inv()
    return (np.array([param1, param2]), Cov, chiSqUpdated)

########################################################################################################################
freq_0_estimate, alpha_L_estimate = 45, 10
startTime = time.time()
L_fit_LM = LevenbergMarquardt(Lor, lorPrimeAlphaL, lorPrimeFreq, lorPrimePrimeAlphaL, lorPrimePrimeFreq,
                              lorPrimePrimeAlphaLFreq, freq, phi, phi_err, freq_0_estimate, alpha_L_estimate)
L_alphaL, L_freq_0 = L_fit_LM[0] #Lorentzian fit parameters for L_freq_0 and L_alphaL
CovMatrix = L_fit_LM[1]
chiSquared = L_fit_LM[2]
P,D = CovMatrix.diagonalize()
D = np.array(D).astype(float)
D = abs(D)
L_FitErrors = np.sqrt(D)
L_alphaL_err, L_freq_0_err = L_FitErrors[0][0], L_FitErrors[1][1]
fig, ax = plt.subplots()
ax.scatter(freq, phi, c="k", s=5)
ax.plot(freq, Lor(freq, L_alphaL, L_freq_0), c="blue", label="Lorentzian Fit")
ax.errorbar(freq, phi, yerr=phi_err, c="k", elinewidth=0.5,ls="none")
ax.set_xlabel("$\\nu$")
ax.set_ylabel("$\phi$")
ax.set_title("Lorentizan Fit")
ax.text(80,0.025,'$\chi^2 = %.2f$'%(chiSquared))
ax.legend(loc="upper right")
#plt.show()
fig.savefig("LorentzianFit.jpg", dpi=300)
endTime = time.time()
print('Lorentzian fit using my LevenbergMarquardt function: \n L_freq_0 = %s +/- %s, \n L_alpha_L = %s +/- %s' % (L_freq_0,
                                                                                                              L_freq_0_err,
                                                                                                              L_alphaL,
                                                                                                              L_alphaL_err))
print("Compilation time for Lorentzian fit: %.3f s\n"%(endTime-startTime))

########################################################################################################################

freq_0_estimate, alpha_G_estimate = 40, 10
startTime = time.time()
G_fit_LM = LevenbergMarquardt(Gau, GauPrimeAlphaG, GauPrimeFreq, GauPrimePrimeAlphaG, GauPrimePrimeFreq,
                              GauPrimePrimeAlphaGFreq, freq, phi, phi_err, freq_0_estimate, alpha_G_estimate)
G_alphaG, G_freq_0 = G_fit_LM[0] #Lorentzian fit parameters for G_freq_0 and G_alphaG
CovMatrix = G_fit_LM[1]
chiSquared = G_fit_LM[2]
P,D = CovMatrix.diagonalize()
D = np.array(D).astype(float)
D = abs(D)
G_FitErrors = np.sqrt(D)
G_alphaG_err, G_freq_0_err = G_FitErrors[0][0], G_FitErrors[1][1]
fig, ax = plt.subplots()
ax.scatter(freq, phi,c="k",s=5)
ax.plot(freq, Gau(freq, G_alphaG, G_freq_0), c="blue",label="Gaussian Fit")
ax.errorbar(freq, phi, yerr=phi_err,c="k", elinewidth=0.5,ls="none")
ax.set_xlabel("$\\nu$")
ax.set_ylabel("$\phi$")
ax.set_title("Gaussian Fit")
ax.legend(loc="upper right")
ax.text(80,0.025,'$\chi^2 = %.2f$'%(chiSquared))
#plt.show()
fig.savefig("GaussianFit.jpg", dpi=300)
endTime = time.time()
print('Gaussian fit using my LevenbergMarquardt function: \n G_freq_0 = %s +/- %s, \n G_alpha_G = %s +/- %s' % (G_freq_0,
                                                                                                            G_freq_0_err,
                                                                                                            G_alphaG,
                                                                                                            G_alphaG_err))
print("Compilation time for Gaussian fit: %.3f s"%(endTime-startTime))