import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from tqdm import tqdm

dat = pd.read_csv("midterm.dat", sep=" ", header=None)
dat = pd.DataFrame(dat)

freq = dat[0]
phi = dat[1]
phi_err = dat[2]

initAlphaD = random.uniform(5, 25)
initMu = random.uniform(30, 60)
initA = random.uniform(0, 5)

def Markov(mode, par1, par2, par3, xData, yData, errorData):
    if mode == 1:
        par = par1
        parProp = random.uniform(par1 - 5, par1 + 5)
        m1 = (par3/par)*np.sqrt(np.log(2)/np.pi)*np.exp(-(np.log(2)*(xData-par2)**2)/(par**2))
        m2 = (par3/parProp)*np.sqrt(np.log(2)/np.pi)*np.exp(-(np.log(2)*(xData-par2)**2)/(parProp**2))
    elif mode == 2:
        par = par2
        parProp = random.uniform(par2 - 5, par2 + 5)
        m1 = (par3 / par1) * np.sqrt(np.log(2) / np.pi) * np.exp(-(np.log(2) * (xData - par) ** 2) / (par1 ** 2))
        m2 = (par3 / par1) * np.sqrt(np.log(2) / np.pi) * np.exp(-(np.log(2) * (xData - parProp) ** 2) / (par1 ** 2))
    elif mode == 3:
        par = par3
        parProp = random.uniform(par3 - 0.5, par3 + 0.5)
        m1 = (par / par1) * np.sqrt(np.log(2) / np.pi) * np.exp(-(np.log(2) * (xData - par2) ** 2) / (par1 ** 2))
        m2 = (parProp / par1) * np.sqrt(np.log(2) / np.pi) * np.exp(-(np.log(2) * (xData - par2) ** 2) / (par1 ** 2))

    p1 = (-1 / 2) * np.sum(((m1 - yData) ** 2) / (errorData ** 2))
    p2 = (-1 / 2) * np.sum(((m2 - yData) ** 2) / (errorData ** 2))

    if abs(p2/p1) < 1:
        probability = p2/p1
    else:
        probability = 1

    a = random.uniform(0, 1)

    if probability > a:
        param = par

    else:
        param = parProp

    return param


def MCMCMain(n):
    alphaDArray = [initAlphaD]
    muArray = [initMu]
    AArray = [initA]

    for i in tqdm(range(n)):
        alphaDVal = Markov(1, alphaDArray[i], muArray[i], AArray[i], freq, phi, phi_err)
        muVal = Markov(2, alphaDArray[i], muArray[i], AArray[i], freq, phi, phi_err)
        AVal = Markov(3, alphaDArray[i], muArray[i], AArray[i], freq, phi, phi_err)

        alphaDArray.append(alphaDVal)
        muArray.append(muVal)
        AArray.append(AVal)

    fig, ax = plt.subplots()
    ax.plot(alphaDArray)
    ax.set_ylim(min(alphaDArray) - 3, max(alphaDArray) + 3)
    ax.set_ylabel("$\\alpha_D$")
    ax.set_xlabel("Iterations")
    ax.set_title("$\\alpha_D$ as a function of iterations")
    fig1, ax1 = plt.subplots()
    ax1.plot(muArray)
    ax1.set_ylabel("$\mu$")
    ax1.set_xlabel("Iterations")
    ax1.set_title("$\mu$ as a function of iterations")
    ax1.set_ylim(min(muArray) - 3, max(muArray) + 3)
    fig2, ax2 = plt.subplots()
    ax2.plot(AArray)
    ax2.set_ylabel("$A$")
    ax2.set_xlabel("Iterations")
    ax2.set_title("A as a function of iterations")
    ax2.set_ylim(min(AArray) - 3, max(AArray) + 3)

    fig.savefig("alphaDArray.jpg", dpi=300)
    fig1.savefig("muArray.jpg", dpi=300)
    fig2.savefig("aArray.jpg", dpi=300)

    print("After %i iterations: alphaD = %.4f, mu = %.4f, A = %.4f"%(n, alphaDArray[-1], muArray[-1], AArray[-1]))
    fig3, ax3 = plt.subplots()
    x = np.linspace(min(freq)-5, max(freq)+5, 10000)
    ax3.scatter(freq, phi, s=5, label="Data")
    ax3.errorbar(freq, phi, yerr = phi_err, c="k", elinewidth=0.5,ls="none")
    y = (AArray[-1]/alphaDArray[-1])*np.sqrt(np.log(2)/np.pi)*np.exp(-(np.log(2)*(x-muArray[-1])**2)/(alphaDArray[-1]**2))
    ax3.plot(x,y, ls = "--", color="k", label = "Metropolis-Hastings MCMC Fit")
    ax3.set_xlabel("$\\nu$")
    ax3.set_ylabel("$\phi$")
    ax3.set_title("Metropolis-Hastings MCMC")
    ax3.legend(loc="upper right", fontsize="small")
    ax3.text(75,0.025, "$\\alpha_D$ = %.4f\n $\mu$ = %.4f\n A = %.4f"%(alphaDArray[0], muArray[0], AArray[0]), fontsize="small")
    fig3.savefig("MCMCFit2.jpg", dpi=300)

    print("After %i iterations: alphaD = %.4f, mu = %.4f, A = %.4f"%(n, np.mean(alphaDArray), np.mean(muArray), np.mean(AArray)))

MCMCMain(25000)