import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x = sp.symbols('x')

########################################################################################################################

# Q1
def LagrangeWeights(n):
    xArray = np.linspace(-1, 1, n)
    denominatorArray = []
    weightArray =[]
    for i in range(len(xArray)):
        weightArray1 = []
        for j in range(len(xArray)):
            if xArray[i] != xArray[j]:
                weightArray1 += [1/(xArray[i]-xArray[j])]
                if weightArray1 not in denominatorArray:
                    denominatorArray.append(weightArray1)
    #print(denominatorArray)
    for i in range(len(denominatorArray)):
        weight = 1
        for j in range(len(denominatorArray[i])):
            weight *= denominatorArray[i][j]

        weightArray.append(weight)

    return np.array(weightArray)

def LagrangeBasisPolynomials(weightArray, n):
    weightArray = weightArray.round(decimals=4)
    basisPolyArray = []
    numeratorArray = []
    x = sp.symbols('x')
    xArray = np.linspace(-1,1,n)
    xArray = np.ma.array(xArray, mask=False)
    for i in range(len(xArray)):
        xArray.mask[i] = True
        numeratorArray.append(x-xArray)
        xArray.mask[i] = False
    numeratorArray = np.array(numeratorArray)
    np.fill_diagonal(numeratorArray, 1)
    for j in range(len(numeratorArray)):
        basisPolyArray += [sp.expand(np.prod(numeratorArray[j]))]
    basisPolyArray = basisPolyArray*weightArray
    return basisPolyArray

def func(x):
    return 1/(25*x**2+1)

def LagrangeInterPoly(basisPoly, func, n):
    xArray = np.linspace(-1,1,n)
    yVals = func(xArray)
    interPoly = []
    for i in range(len(xArray)):
        interPoly += [yVals[i]*basisPoly[i]]
    interPoly = np.array(interPoly)
    LagrangeInterPolynomial = interPoly.sum()
    return LagrangeInterPolynomial

def LagrangeMain():
    #fig, ax = plt.subplots()
    xVals = np.linspace(-1, 1, 100)
    y = 1/(25*xVals**2+1)
    plt.plot(xVals, y, label="Runge function")
    y1 = LagrangeInterPoly(LagrangeBasisPolynomials(LagrangeWeights(6), 6), func, 6)
    y2 = LagrangeInterPoly(LagrangeBasisPolynomials(LagrangeWeights(8), 8), func, 8)
    y3 = LagrangeInterPoly(LagrangeBasisPolynomials(LagrangeWeights(10), 10), func, 10)
    y1ValArray = []
    y2ValArray = []
    y3ValArray = []

    for i in range(len(xVals)):
        yVals1 = y1.subs(x, xVals[i])
        yVals2 = y2.subs(x, xVals[i])
        yVals3 = y3.subs(x, xVals[i])
        y1ValArray.append((sp.N(yVals1)))
        y2ValArray.append((sp.N(yVals2)))
        y3ValArray.append((sp.N(yVals3)))

    plt.plot(xVals, y1ValArray, linestyle="--", label='$n = %i$'%(6))
    plt.plot(xVals, y2ValArray, linestyle="--", label='$n = %i$' % (8))
    plt.plot(xVals, y3ValArray, linestyle="--", label='$n = %i$' % (10))

    plt.title("Interpolated data")
    plt.legend(loc="upper right")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-0.3, 1.05)
    plt.savefig("LagrangeInterpolation.jpg", dpi=300)

def CubicSpline(func, n):
    xArray = np.linspace(-1,1,n+1)
    yArray = func(xArray)

    hArray = []

    for i in range(n):
        hArray += [xArray[i+1] - xArray[i]]

    alphaArray = []

    for i in range(n):
        alphaArray += [(3/hArray[i])*(yArray[i+1]-yArray[i])-(3/hArray[i-1])*(yArray[i]-yArray[i-1])]

    lArray = [1]
    muArray = [0]
    zArray = [0]

    for i in range(1, n):
        lArray += [2*(xArray[i+1]-xArray[i-1])-hArray[i-1]*muArray[i-1]]
        muArray += [hArray[i]/lArray[i]]
        zArray += [(alphaArray[i]-hArray[i-1]*zArray[i-1])/lArray[i]]

    lArray.append(1)
    zArray.append(0)

    cArray = np.zeros(n+1)
    cArray = cArray.astype(int)
    bArray = []
    dArray = []

    for j in range(n-1, -1, -1):
        cArray[j] = zArray[j]-muArray[j]*cArray[j+1]
        bArray = [(yArray[j+1]-yArray[j])/hArray[j]-(hArray[j]*(cArray[j+1]+2*cArray[j]))/3]+bArray
        dArray = [(cArray[j+1]-cArray[j])/(3*hArray[j])] + dArray

    splines = []
    for i in range(n):
        splines += [[yArray[i], bArray[i], cArray[i], dArray[i], xArray[i]]]

    return splines

def CubicSplineMain():
    a = []
    b = []
    c = []
    d = []
    x_j = []

    for i in range(len(CubicSpline(func, 10))):
        a += [CubicSpline(func, 10)[i][0]]
        b += [CubicSpline(func, 10)[i][1]]
        c += [CubicSpline(func, 10)[i][2]]
        d += [CubicSpline(func, 10)[i][3]]
        x_j += [CubicSpline(func, 10)[i][4]]

    xArray1 = np.linspace(-1, 1, 11)
    gap1 = np.linspace(xArray1[0], xArray1[1], 100)
    gap2 = np.linspace(xArray1[1], xArray1[2], 100)
    gap3 = np.linspace(xArray1[2], xArray1[3], 100)
    gap4 = np.linspace(xArray1[3], xArray1[4], 100)
    gap5 = np.linspace(xArray1[4], xArray1[5], 100)
    gap6 = np.linspace(xArray1[5], xArray1[6], 100)
    gap7 = np.linspace(xArray1[6], xArray1[7], 100)
    gap8 = np.linspace(xArray1[7], xArray1[8], 100)
    gap9 = np.linspace(xArray1[8], xArray1[9], 100)
    gap10 = np.linspace(xArray1[9], xArray1[10], 100)

    y0 = a[0] + b[0] * (gap1 - x_j[0]) + c[0] * (gap1 - x_j[0]) ** 2 + d[0] * (gap1 - x_j[0]) ** 3
    y1 = a[1] + b[1] * (gap2 - x_j[1]) + c[1] * (gap2 - x_j[1]) ** 2 + d[1] * (gap2 - x_j[1]) ** 3
    y2 = a[2] + b[2] * (gap3 - x_j[2]) + c[2] * (gap3 - x_j[2]) ** 2 + d[2] * (gap3 - x_j[2]) ** 3
    y3 = a[3] + b[3] * (gap4 - x_j[3]) + c[3] * (gap4 - x_j[3]) ** 2 + d[3] * (gap4 - x_j[3]) ** 3
    y4 = a[4] + b[4] * (gap5 - x_j[4]) + c[4] * (gap5 - x_j[4]) ** 2 + d[4] * (gap5 - x_j[4]) ** 3
    y5 = a[5] + b[5] * (gap6 - x_j[5]) + c[5] * (gap6 - x_j[5]) ** 2 + d[5] * (gap6 - x_j[5]) ** 3
    y6 = a[6] + b[6] * (gap7 - x_j[6]) + c[6] * (gap7 - x_j[6]) ** 2 + d[6] * (gap7 - x_j[6]) ** 3
    y7 = a[7] + b[7] * (gap8 - x_j[7]) + c[7] * (gap8 - x_j[7]) ** 2 + d[7] * (gap8 - x_j[7]) ** 3
    y8 = a[8] + b[8] * (gap9 - x_j[8]) + c[8] * (gap9 - x_j[8]) ** 2 + d[8] * (gap9 - x_j[8]) ** 3
    y9 = a[9] + b[9] * (gap10 - x_j[9]) + c[9] * (gap10 - x_j[9]) ** 2 + d[9] * (gap10 - x_j[9]) ** 3

    #fig, ax = plt.subplots()
    plt.plot(gap1, y0, label="S0")
    plt.plot(gap2, y1, label="S1")
    plt.plot(gap3, y2, label="S2")
    plt.plot(gap4, y3, label="S3")
    plt.plot(gap5, y4, label="S4")
    plt.plot(gap6, y5, label="S5")
    plt.plot(gap7, y6, label="S6")
    plt.plot(gap8, y7, label="S7")
    plt.plot(gap9, y8, label="S8")
    plt.plot(gap10, y9, label="S9")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Cubic Spline Interpolation")
    plt.legend(loc="upper right", fontsize='x-small')

    plt.savefig("cubicSpline.jpg", dpi=300)

########################################################################################################################
# Q2

def CompositeTrapezoidal(func, a, b, n):
    xArray = np.linspace(a, b, n+1)
    height = (abs(a)+abs(b))/n
    firstHalf = (height/2)*(func(a)+func(b))
    secondHalfArray = []
    for i in range(n):
        secondHalfArray += [func(xArray[i])]
    secondHalfArray = np.array(secondHalfArray)
    secondHalf = height*np.sum(secondHalfArray)
    integralValue = firstHalf + secondHalf
    return integralValue

def SimpsonsComposite(func, a, b, n):
    xArray = np.linspace(a, b, n+1)
    height = (abs(a) + abs(b)) / n
    firstThird = (height/3)*(func(a)+func(b))
    secondThirdArray = []
    m = int(n/2)
    for i in range(m):
        secondThirdArray += [func(xArray[2*i-1])]
    secondThirdArray = np.array(secondThirdArray)
    secondThird = (4/3)*height*np.sum(secondThirdArray)
    thirdThirdArray = []
    o = int(n/2-1)
    for j in range(o):
        thirdThirdArray += [func(xArray[2*j])]
    thirdThirdArray = np.array(thirdThirdArray)
    thirdThird = (2/3)*height*np.sum(thirdThirdArray)
    integralValue = firstThird + secondThird + thirdThird
    return integralValue

def func1(x):
    y = 1/(np.sqrt(2*np.pi))*np.exp((-(x-1)**2)/2)
    return y

def integrationMain(n):
    analyticalValue = 1

    trapValue = []
    simpValue = []
    counter = []

    for i in range(1,n,10):
        counter += [i]
        trapValue += [CompositeTrapezoidal(func1, -100, 100, i)]
        simpValue += [SimpsonsComposite(func1, -100, 100, i)]

    trapDiff = []
    simpDiff = []

    for j in range(100):
        trapDiff += [abs(trapValue[j] - analyticalValue) / analyticalValue]
        simpDiff += [abs(simpValue[j] - analyticalValue) / analyticalValue]

    fig, ax = plt.subplots()

    ax.plot(counter, trapValue, label="Trapezoidal Integration")
    ax.plot(counter, simpValue, label="Simspon's Integration", linestyle="--")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Integration Value")
    ax.legend(loc="upper right")
    ax.set_title("Integration Value as function of steps")

    fig.savefig("stepValuePlot.jpg", dpi=300)

    fig1, ax1 = plt.subplots()
    ax1.plot(counter, trapDiff, label="Trapezoidal Integration")
    ax1.plot(counter, simpDiff, label="Simpson's Integration", linestyle="--")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Integration Value Error")
    ax1.set_title("Integration Value Error as function of steps")

    fig1.savefig("stepErrorPlot.jpg", dpi=300)

def main(n): #n is the number of the iterations/steps for the integration functions
    LagrangeMain()
    CubicSplineMain()
    integrationMain(n)

main(1000)