def func(x):
    return x ** 6- 1


def secant_method(f, x0, x1, tol=1e-5, n=0):
    # increment counter
    n += 1

    # calculate function values at endpoints
    y0, y1 = f(x0), f(x1)

    # calculate next root approximation
    xn = x1 - y1 * ((x1 - x0) / (y1 - y0))

    # check tolerance condition
    if -tol < y1 < tol:
        return xn, n

    # recursive call with updated interval
    return secant_method(f, x1, xn, n=n)

print(secant_method(func, 1, 3, tol=1e-5, n=0))
