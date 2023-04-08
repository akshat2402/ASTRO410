import numpy as np
import matplotlib.pyplot as plt

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

    for i in range(1,n):
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

def func(x):
    return 1/(25*x**2+1)



print(CubicSpline(func,10)[9])

xArray= np.linspace(-1,1,100)

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

print(a)
print(b)
print(c)
print(d)
print(x_j)

xArray1 = np.linspace(-1,1,11)
gap1 = np.linspace(xArray1[0],xArray1[1], 100)
gap2 = np.linspace(xArray1[1],xArray1[2], 100)
gap3 = np.linspace(xArray1[2],xArray1[3], 100)
gap4 = np.linspace(xArray1[3],xArray1[4], 100)
gap5 = np.linspace(xArray1[4],xArray1[5], 100)
gap6 = np.linspace(xArray1[5],xArray1[6], 100)
gap7 = np.linspace(xArray1[6],xArray1[7], 100)
gap8 = np.linspace(xArray1[7],xArray1[8], 100)
gap9 = np.linspace(xArray1[8],xArray1[9], 100)
gap10 = np.linspace(xArray1[9],xArray1[10], 100)

y0 = a[0] + b[0]*(gap1-x_j[0]) + c[0]*(gap1-x_j[0])**2 + d[0]*(gap1-x_j[0])**3
y1 = a[1] + b[1]*(gap2-x_j[1]) + c[1]*(gap2-x_j[1])**2 + d[1]*(gap2-x_j[1])**3
y2 = a[2] + b[2]*(gap3-x_j[2]) + c[2]*(gap3-x_j[2])**2 + d[2]*(gap3-x_j[2])**3
y3 = a[3] + b[3]*(gap4-x_j[3]) + c[3]*(gap4-x_j[3])**2 + d[3]*(gap4-x_j[3])**3
y4 = a[4] + b[4]*(gap5-x_j[4]) + c[4]*(gap5-x_j[4])**2 + d[4]*(gap5-x_j[4])**3
y5 = a[5] + b[5]*(gap6-x_j[5]) + c[5]*(gap6-x_j[5])**2 + d[5]*(gap6-x_j[5])**3
y6 = a[6] + b[6]*(gap7-x_j[6]) + c[6]*(gap7-x_j[6])**2 + d[6]*(gap7-x_j[6])**3
y7 = a[7] + b[7]*(gap8-x_j[7]) + c[7]*(gap8-x_j[7])**2 + d[7]*(gap8-x_j[7])**3
y8 = a[8] + b[8]*(gap9-x_j[8]) + c[8]*(gap9-x_j[8])**2 + d[8]*(gap9-x_j[8])**3
y9 = a[9] + b[9]*(gap10-x_j[9]) + c[9]*(gap10-x_j[9])**2 + d[9]*(gap10-x_j[9])**3

plt.plot(gap1, y0)
plt.plot(gap2, y1)
plt.plot(gap3, y2)
plt.plot(gap4, y3)
plt.plot(gap5, y4)
plt.plot(gap6, y5)
plt.plot(gap7, y6)
plt.plot(gap8, y7)
plt.plot(gap9, y8)
plt.plot(gap10, y9)
plt.show()



#y0 = a + b*(xArray-x_j) + c*(xArray-x_j)**2 + d*(xArray-x_j)**3

# plt.plot(xArray,y0)
# plt.plot(xArray, func(xArray))

