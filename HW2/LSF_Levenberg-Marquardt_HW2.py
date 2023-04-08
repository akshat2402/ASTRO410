
#To import required modules:
import numpy as np
import time
import matplotlib
import matplotlib.cm as cm #for color maps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec #for specifying plot attributes
import scipy.optimize as opt #for scipy optimization/fitting routines



##### (1), (2)

#To write a program for fitting the data provided in 'hw2_fitting.dat'
#Will first use scipy.optimize.curve_fit to fit the data to a Lorentzian and then a Gaussian

#To load the data:
data = np.loadtxt('hw2_fitting.dat') #columns are: frequency (nu), line strength data (phi), error in phi (err)
nu = data[:,0]
phi = data[:,1]
err = data[:,2]

#To define our fitting functions, a Lorentzian and a Gaussian:
def Lorentzian(x, nu0, alpha_L):
    #Takes in x values of nu, and two fit parameters nu0 and alpha_L
    y = (1./np.pi)*(alpha_L/((x - nu0)**2. + alpha_L**2.))
    return y

def Gaussian(x, nu0, alpha_D):
    #Takes in x values of nu, and two fit parameters nu0 and alpha_D
    y = (1./alpha_D)*np.sqrt(np.log(2.)/np.pi)*np.exp((-np.log(2.)*(x - nu0)**2.)/(alpha_D**2.))
    return y

#Now use scipy.optimize.curve_fit to fit our data to both functions:
L_fit = opt.curve_fit(Lorentzian, nu, phi, sigma=err, method='lm') #Lorentzian fit
L_nu0, L_alpha = L_fit[0] #Lorentzian fit parameters for nu0 and alpha_L
L_nu0_err, L_alpha_err = np.sqrt(np.diag(L_fit[1])) #Lorentzian fit errors for nu0 and alpha_L
print('Lorentzian fit using the scipy function: \n nu_0 = %s +/- %s$, \n alpha_L = %s +/- %s' % (L_nu0, L_nu0_err, L_alpha, L_alpha_err))

G_fit = opt.curve_fit(Gaussian, nu, phi, sigma=err, method='lm') #Gaussian fit
G_nu0, G_alpha = G_fit[0] #Gaussian fit parameters for nu0 and alpha_L
G_nu0_err, G_alpha_err = np.sqrt(np.diag(G_fit[1])) #Gaussian fit errors for nu0 and alpha_D
print('Gaussian fit using the scipy function: \n nu_0 = %s +/- %s$, \n alpha_D = %s +/- %s' % (G_nu0, G_nu0_err, G_alpha, G_alpha_err))



#####

##### Now we will attempt to write our own Levenberg Marquardt fit function:

#Define a chisquared function:
def Chisquared(f, data_f, data_x, data_err, a1, a2):
    #Evaluates the chi square given a fit function, the data and its errors, and the fit function parameters
    return np.sum(((data_f - f(data_x, a1, a2))/data_err)**2.)

#Now define the LMfit function:
def LMfit(f, dfda1, dfda2, ddfda1da1, ddfda2da2, ddfda1da2, data_f, data_x, data_err, a1, a2):
    #This performs the LM fit for any given function with 2 fit parameters
    #Must take in an initial guess for the 2 fit parameters, a1 and a2
    
    #To initiate LM algorithm:
    chisq = Chisquared(f, data_f, data_x, data_err, a1, a2) #initial chisq
    lamb = 0.001 #initial value of lambda

    dchisq_da1 = -2.*np.sum(((data_f - f(data_x, a1, a2))/(data_err**2.))*dfda1(data_x, a1, a2))
    dchisq_da2 = -2.*np.sum(((data_f - f(data_x, a1, a2))/(data_err**2.))*dfda2(data_x, a1, a2))

    ddchisq_da1da1 = 2.*np.sum((1./data_err**2.)*(dfda1(data_x, a1, a2)**2. - (data_f - f(data_x, a1, a2))*ddfda1da1(data_x, a1, a2)))
    ddchisq_da2da2 = 2.*np.sum((1./data_err**2.)*(dfda2(data_x, a1, a2)**2. - (data_f - f(data_x, a1, a2))*ddfda2da2(data_x, a1, a2)))
    ddchisq_da1da2 = 2.*np.sum((1./data_err**2.)*(dfda1(data_x, a1, a2)*dfda2(data_x, a1, a2) - (data_f - f(data_x, a1, a2))*ddfda1da2(data_x, a1, a2)))

    Beta = -0.5*np.array([dchisq_da1, dchisq_da2]) #Beta vector
    Alpha_LM = 0.5*np.array([[ddchisq_da1da1*(1.+lamb), ddchisq_da1da2], [ddchisq_da1da2, ddchisq_da2da2*(1.+lamb)]]) #Alpha matrix with lambda factor on diagonal

    da1, da2 = np.linalg.solve(Alpha_LM, Beta) #solve the system of equations for the increments to the fit parameters, da1 and da2
    a1_new, a2_new = a1+da1, a2+da2 #temporary incremented fit p    arameters a1 and a2 (only update a1 and a2 if chisq_new < chisq)

    chisq_new = Chisquared(f, data_f, data_x, data_err, a1_new, a2_new)
    
    #Main iterative loop:
    while abs(chisq_new - chisq)/chisq > 10.**(-6.):
        if chisq_new >= chisq:
            lamb = lamb*10.
            
            Alpha_LM = 0.5*np.array([[ddchisq_da1da1*(1.+lamb), ddchisq_da1da2], [ddchisq_da1da2, ddchisq_da2da2*(1.+lamb)]]) #Alpha matrix with lambda factor on diagonal
            
            da1, da2 = np.linalg.solve(Alpha_LM, Beta) #solve the system of equations for the increments to the fit parameters, da1 and da2
            a1_new, a2_new = a1+da1, a2+da2 #incremented fit parameters a1 and a2

            chisq_new = Chisquared(f, data_f, data_x, data_err, a1_new, a2_new)

            print(chisq_new)

        elif chisq_new < chisq:
            lamb = lamb/10.
            a1, a2 = a1+da1, a2+da2 #update the fit parameters
            
            dchisq_da1 = -2.*np.sum(((data_f - f(data_x, a1, a2))/(data_err**2.))*dfda1(data_x, a1, a2))
            dchisq_da2 = -2.*np.sum(((data_f - f(data_x, a1, a2))/(data_err**2.))*dfda2(data_x, a1, a2))
            
            ddchisq_da1da1 = 2.*np.sum((1./data_err**2.)*(dfda1(data_x, a1, a2)**2. - (data_f - f(data_x, a1, a2))*ddfda1da1(data_x, a1, a2)))
            ddchisq_da2da2 = 2.*np.sum((1./data_err**2.)*(dfda2(data_x, a1, a2)**2. - (data_f - f(data_x, a1, a2))*ddfda2da2(data_x, a1, a2)))
            ddchisq_da1da2 = 2.*np.sum((1./data_err**2.)*(dfda1(data_x, a1, a2)*dfda2(data_x, a1, a2) - (data_f - f(data_x, a1, a2))*ddfda1da2(data_x, a1, a2)))
            
            Beta = -0.5*np.array([dchisq_da1, dchisq_da2]) #Beta vector
            Alpha_LM = 0.5*np.array([[ddchisq_da1da1*(1.+lamb), ddchisq_da1da2], [ddchisq_da1da2, ddchisq_da2da2*(1.+lamb)]]) #Alpha matrix with lambda factor on diagonal
            
            da1, da2 = np.linalg.solve(Alpha_LM, Beta) #solve the system of equations for the increments to the fit parameters, da1 and da2
            a1_new, a2_new = a1+da1, a2+da2 #incremented fit parameters a1 and a2
            
            chisq = chisq_new
            chisq_new = Chisquared(f, data_f, data_x, data_err, a1_new, a2_new)
            print(chisq_new)
            print("a")

    #To calculate the covariance matrix of errors now that the fit is found:
    Alpha_LM = 0.5*np.array([[ddchisq_da1da1, ddchisq_da1da2], [ddchisq_da1da2, ddchisq_da2da2]]) #Alpha matrix with lambda = 0
    Cov = np.linalg.inv(Alpha_LM) #Covariance matrix of errors; the inverse of the Alpha_LM matrix with lambda = 0

    return (np.array([a1, a2]), Cov) #return the 2 fit parameters a1 and a2, and a matrix of variances Cov



#For the Lorentzian fit:
def dLda1(x, nu0, alpha_L):
    #Evaluates the derivative dL/dnu0
    y = (2.*(x - nu0)*alpha_L)/(np.pi*((x - nu0)**2. + alpha_L**2.)**2.)
    return y

def dLda2(x, nu0, alpha_L):
    #Evaluates the derivative dL/dalpha_L
    y = (1./np.pi)*(((x - nu0)**2. - alpha_L**2.)/(((x - nu0)**2. + alpha_L**2.)**2.))
    return y

def ddLda1da1(x, nu0, alpha_L):
    #Evaluates the second derivative (d/dnu0)(dL/dnu0)
    y = ((2.*alpha_L)/np.pi)*((3.*(x - nu0)**2. - alpha_L**2.)/(((x - nu0)**2. + alpha_L**2.)**3.))
    return y

def ddLda2da2(x, nu0, alpha_L):
    #Evaluates the second derivative (d/dalpha_L)(dL/dalpha_L)
    y = ((2.*alpha_L)/np.pi)*((alpha_L**2. - 3.*(x - nu0)**2.)/(((x - nu0)**2. + alpha_L**2.)**3.))
    return y

def ddLda1da2(x, nu0, alpha_L):
    #Evaluates the mixed second derivative (d/dalpha_L)(dL/dnu0)
    y = ((2.*(x - nu0))/np.pi)*(((x - nu0)**2. - 3.*alpha_L**2.)/(((x - nu0)**2. + alpha_L**2.)**3.))
    return y

nu0_guess, alpha_L_guess = 40., 10.
L_fit_own = LMfit(Lorentzian, dLda1, dLda2, ddLda1da1, ddLda2da2, ddLda1da2, phi, nu, err, nu0_guess, alpha_L_guess)
L_nu0, L_alpha = L_fit_own[0] #Lorentzian fit parameters for nu0 and alpha_L
L_nu0_err, L_alpha_err = np.sqrt(np.diag(L_fit_own[1])) #Lorentzian fit errors for nu0 and alpha_L
print('Lorentzian fit using our LMfit function: \n nu_0 = %s +/- %s, \n alpha_L = %s +/- %s' % (L_nu0, L_nu0_err, L_alpha, L_alpha_err))



#For the Gaussian fit:
def dGda1(x, nu0, alpha_D):
    #Evaluates the derivative dG/dnu0
    y = ((2.*(np.log(2.)**(3./2.))*(x - nu0))/((alpha_D**3.)*np.sqrt(np.pi)))*np.exp((-np.log(2.)*(x - nu0)**2.)/(alpha_D**2.))
    return y

def dGda2(x, nu0, alpha_D):
    #Evaluates the derivative dG/dalpha_D
    y = (1./(alpha_D**4.))*np.sqrt(np.log(2.)/np.pi)*np.exp((-np.log(2.)*(x - nu0)**2.)/(alpha_D**2.))*(2.*np.log(2.)*(x - nu0)**2. - alpha_D**2.)
    return y

def ddGda1da1(x, nu0, alpha_D):
    #Evaluates the second derivative (d/dnu0)(dG/dnu0)
    y = ((2.*(np.log(2.)**(3./2.)))/((alpha_D**4.)*np.sqrt(np.pi)))*np.exp((-np.log(2.)*(x - nu0)**2.)/(alpha_D**2.))*(2.*np.log(2.)*(x - nu0)**2. - alpha_D**2.)
    return y

def ddGda2da2(x, nu0, alpha_D):
    #Evaluates the second derivative (d/dalpha_D)(dG/dalpha_D)
    y = (2./(alpha_D**7.))*np.sqrt(np.log(2.)/np.pi)*np.exp((-np.log(2.)*(x - nu0)**2.)/(alpha_D**2.))*(alpha_D**4. - 5.*np.log(2.)*((x - nu0)**2.)*(alpha_D**2.) + 2.*(np.log(2.)**2.)*(x - nu0)**4.)
    return y

def ddGda1da2(x, nu0, alpha_D):
    #Evaluates the mixed second derivative (d/dalpha_D)(dG/dnu0)
    y = ((2.*(np.log(2.)**(3./2.))*(x - nu0))/(alpha_D**6.))*np.exp((-np.log(2.)*(x - nu0)**2.)/(alpha_D**2.))*(2.*np.log(2.)*(x - nu0)**2. - 3.*alpha_D**2.)
    return y

nu0_guess, alpha_D_guess = 40., 10.
G_fit_own = LMfit(Gaussian, dGda1, dGda2, ddGda1da1, ddGda2da2, ddGda1da2, phi, nu, err, nu0_guess, alpha_D_guess)
G_nu0, G_alpha = G_fit_own[0] #Gaussian fit parameters for nu0 and alpha_L
G_nu0_err, G_alpha_err = np.sqrt(np.diag(G_fit_own[1])) #Gaussian fit errors for nu0 and alpha_D
print('Gaussian fit using our LMfit function: \n nu_0 = %s +/- %s, \n alpha_D = %s +/- %s' % (G_nu0, G_nu0_err, G_alpha, G_alpha_err))



##### (3)

#To plot the data and the fits:
fig = plt.figure(figsize=(10,5))
plot = GridSpec(1,1,left=0.125,bottom=0.15,right=0.95,top=0.925,wspace=0.1,hspace=0)
plt.subplot(plot[:,:]) #Lorentzian fit plot
plt.errorbar(nu, phi, yerr=err, fmt='.')
plt.plot(nu, Lorentzian(nu, L_nu0, L_alpha), 'g-')
plt.xlabel(r'$\nu$', fontsize=20)
plt.ylabel(r'$\phi(\nu)$', fontsize=20)
#plt.show()
plt.savefig('lorentz1.jpg', bbox_inches="tight", dpi=300)

fit = plt.figure(figsize=(10,5))
plot = GridSpec(1,1,left=0.125,bottom=0.15,right=0.95,top=0.925,wspace=0.1,hspace=0)
plt.subplot(plot[:,:]) #Gaussian fit plot
plt.errorbar(nu, phi, yerr=err, fmt='.')
plt.plot(nu, Gaussian(nu, G_nu0, G_alpha), 'g-')
plt.xlabel(r'$\nu$', fontsize=20)
plt.ylabel(r'$\phi(\nu)$', fontsize=20)
#plt.show()
plt.savefig('gauss1.jpg', bbox_inches="tight", dpi=300)

