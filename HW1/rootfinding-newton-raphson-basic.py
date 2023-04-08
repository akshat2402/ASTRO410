
def f(x):
    return 6*x**5-5*x**4-4*x**3+3*x**2
 
def df(x):
    return 30*x**4-20*x**3-12*x**2+6*x

 
def newtons_raphson(f, df, x0, e):
    df0=df(x0)
    if df0+1 == 1: 
        if f(x0)+1 == 1:
            print('Root is at: ', x0)
            print('f(x) at root is: ', f(x0))
        else:
            print('invalid df0')
        return
    
    dx=f(x0)/df0
   
    while abs(dx) > e:
        x0 = x0 - dx
        dx=f(x0)/df(x0)
        print('Root is at: ', x0)
        print('f(x) at root is: ', f(x0))


x0s = [0,0.5, 1]
for x0 in x0s:
    newtons_raphson(f, df, x0, 1e-5)
 
# Root is at:  0
# f(x) at root is:  0
# Root is at:  0.628668078167
# f(x) at root is:  -1.37853879978e-06
# Root is at:  1
# f(x) at root is:  0
