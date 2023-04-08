import numpy as np


def cubicspline(n, f): #Cubic natural spline interpolation
	x = np.linspace(-1,1,n+1) #n intervals for cubic spline
	y = f(x) #values at interval boundaries from 0 to n
	h = np.zeros(n) #For interval lengths
	for i in range(n):
		h[i] = x[i+1]-x[i] #Intervalic lengths
	#A = np.zeros((n+1,n+1)) #For finding p'' values (Ax=b equation)
	#A[0,0] = 1 #Values known for natural spline for the matrix
	#A[n,n] = 1
	p2prime = np.zeros(n+1) #For filling p'' values--the first and last will remain zero for natural spline
	a, b, c, r = np.zeros(n-1), np.zeros(n-1), np.zeros(n-1), np.zeros(n-1) #For finding p'' values (Ax=r equation)--tridiagonal system with diagonals a, b, c--only considering the non-trivial rows and columns (since p''_0 and p''_n = 0)
	for i in range(n-1): #Set up the matrix A (with diagonals a,b,c) and the column vector r
		r[i] = 6*(1/h[i+1]*(y[i+2]-y[i+1]) - 1/h[i]*(y[i+1]-y[i]))
		b[i] = 2*(h[i]+h[i+1])
		if (i<n-2): c[i] = h[i+1] #c[n-2] = 0
		if (i>0): a[i] = h[i] #a[0] = 0
		

	#Now we need to do Gaussian elimination (special case of tridiagonal matrix) using algorithm from earlier lecture
	beta = b #This is the beta from the algorithm
	rho = r #This is the rho from the algorithm
	for j in range(1,n-1):
		beta[j]=b[j]-a[j]/beta[j-1] * c[j-1]
		rho[j] = r[j] - a[j]/beta[j-1] * rho[j-1]
	p2prime[n-1] = rho[n-2]/beta[n-2] #Back substitution--the weird mixes of indices is because of different sized arrays
	for j in range(1,n-1):
		p2prime[n-j-1] = 1/beta[n-2-j] * (rho[n-2-j] - c[n-2-j]*p2prime[n-j])
	#Now we have finally calculated the p'' vector!
	x0 = np.linspace(-1,1,100)
	p = np.zeros(len(x0)) #This is the array of values we will fill with polynomial values
	for j in range(n):
		cond = np.logical_and(x0>=x[j],x0<=x[j+1]) #Fitting cubic spline between x_j and x_j+1
		p[cond] = (p2prime[j+1]-p2prime[j])/(6*h[j]) * (x0[cond]-x[j])**3 + p2prime[j]/2 * (x0[cond]-x[j])**2 + ((y[j+1]-y[j])/h[j] - h[j]*p2prime[j+1]/6 - h[j]*p2prime[j]/3)*(x0[cond]-x[j]) + y[j] #Pretty nasty formula--this gives the function p(x) between x_j and x_j+1. By splicing together these cubic splines, we get the entire interpolation (which I call p)!

	return p

def f(x):
    return 1/(25*x**2+1)

print(len(cubicspline(10,f)))



