import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.optimize import minimize

''' 
TODO: Fix the Rosenbrock equation for all examples...
TODO: Move functions to appropriate py files
'''


def demo1(): # 2-5-2018

    A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    b = np.array([[1], [0], [1]])

    def func1(x):
        return (x[0]**2-x[1])**2 + (1-x[1])**2

    def func2(x):
        return 0.5*(np.matmul(x.T, np.matmul(A, x))) - np.matmul(b.T, x)


    print(LA.eigvals(A))
    print()
    print(LA.solve(A.T.dot(A), A.T.dot(b)))  # A\b matrix left divide
    print()
    # find matrix row sum

    x0 = [0, 0]
    sol1 = minimize(func1, x0, method='Nelder-Mead')
    print(sol1)

    print()

    x1 = [0, 0, 0]
    sol2 = minimize(func2, x1, method='SLSQP')
    print(sol2)


def rosenPlot(): # 2-5-2018

    fig = plt.figure()

    def func(x):
        return (x[0]**2-x[1])**2 + (1-x[1])**2

    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)

    X, Y = np.meshgrid(x, y)

    Z = func([X, Y])

    ax = plt.axes(projection='3d')

    cont = ax.contour(X, Y, Z, 1000, cmap='hsv', alpha=0.6)

    ax.view_init(60, -80)

    plt.show()


def demo2():  # line search 2-12-2018

    def func(x):
        return 2*(x**2) - 8*x + 12

    a0 = 0
    b0 = 3
    rho = 0.382

    for x in range(0, 10):
        a1 = a0 + rho*(b0 - a0)
        b1 = b0 - rho*(b0 - a0)

        f1 = func(a1)
        f2 = func(b1)

        if f1 > f2:
            a0 = a1
        else:
            b0 = b1


        print('iteration #', x)
        print('[', a0, ',', b0, ']')
        print()


def demo2challenge():  # challenge problem at end of class 2-12-2018
    def func(x):
        return x**4 - 14*(x**3) + 60*(x**2) - 70*x

    a0 = 0
    b0 = 2
    uncertainty = 0.0001
    rho = 0.382  # 1 - golden ratio
    print('iteration # 0', [a0, b0])

    iteration = 0
    while uncertainty < abs(b0 - a0):
        a1 = a0 + rho*(b0 - a0)
        b1 = b0 - rho*(b0 - a0)

        f1 = func(a1)
        f2 = func(b1)

        if f1 > f2:
            a0 = a1
        else:
            b0 = b1

        iteration += 1
        print('iteration #', iteration, [a0, b0])


def gradDescentDemo1():  # 2-14-2018    # constant steps
    fig = plt.figure()

    def func(x):
        return 4*(x[0]**2) + x[1]**2

    def gradFunc(x):
        return [8*x[0], 2*x[1]]

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)

    X, Y = np.meshgrid(x, y)

    Z = func([X, Y])

    # Plot 3D contour
    # ax = plt.axes(projection='3d')
    # cont3D = ax.contour(X, Y, Z, 100, alpha=0.9, cmap='magma')

    # Gradient descent - with constant steps
    a = 0.05  # step size
    x0 = [3, 5]
    xP = x0

    for x in range(0, 25):
        xP = xP - a*np.array(gradFunc(xP))
        print(xP)
        plt.scatter(xP[0], xP[1])

    # Plot 2D contour
    plt.contour(X, Y, Z, 20)
    plt.show()


def gradDescentDemo2():  # line search    2-14-2018
    def func(x):
        return 4*(x[0]**2) + x[1]**2

    def gradFunc(x):
        return [8*x[0], 2*x[1]]

    def lineFunc(x, grad, t):
        return np.array(x) - t*np.array(grad)

    def goldSearch(x, grad):
        a0 = -10
        b0 = 10
        rho = 0.382  # 1 - golden ratio
        while abs(a0 - b0) > 0.001:  # maybe change from 0.01??
            a1 = a0 + rho * (b0 - a0)
            b1 = b0 - rho * (b0 - a0)

            at = lineFunc(x, grad, a1)
            bt = lineFunc(x, grad, b1)

            f1 = func(at)
            f2 = func(bt)

            if f1 > f2:
                a0 = a1
            else:
                b0 = b1

        return lineFunc(x, grad, a0 + (abs(a0-b0)/2))

    def plotPath(points):  # maybe learn how to animate the plot???
        fig = plt.figure()

        x = np.linspace(-6, 6, 100)
        y = np.linspace(-6, 6, 100)
        X, Y = np.meshgrid(x, y)

        Z = func([X, Y])

        plt.plot([p[0] for p in points], [p[1] for p in points], '-o')
        plt.contour(X, Y, Z, 20)
        plt.show()

    x0 = [3, 5]  # starting point
    points = [x0]
    xP = x0

    for x in range(0, 8):
        xP = goldSearch(xP, gradFunc(xP))
        points.append(xP)

    print(xP)
    plotPath(points)


def gradDescentDemo3(testPoint):  # line search - Rosenbrock    2-14-2018
    def func(x):
        return (x[0]**2-x[1])**2 + (1-x[1])**2

    def gradFunc(x):
        return [4*(x[0]**3) - 4*x[0]*x[1], -2*(x[0]**2) + 4*x[1] - 2]

    def lineFunc(x, grad, t):
        return np.array(x) - t*np.array(grad)

    def goldSearch(x, grad):
        a0 = -1
        b0 = 5
        rho = 0.382  # 1 - golden ratio
        while abs(a0 - b0) > 0.0001:  # maybe change from 0.01??
            a1 = a0 + rho * (b0 - a0)
            b1 = b0 - rho * (b0 - a0)

            at = lineFunc(x, grad, a1)
            bt = lineFunc(x, grad, b1)

            f1 = func(at)
            f2 = func(bt)

            if f1 > f2:
                a0 = a1
            else:
                b0 = b1

        return lineFunc(x, grad, a0 + (abs(a0-b0)/2))

    def plotPath(points):  # maybe learn how to animate the plot???
        fig = plt.figure()

        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)

        Z = func([X, Y])

        plt.plot([p[0] for p in points], [p[1] for p in points], '-o', color='red')
        plt.contour(X, Y, Z, 60)
        plt.show()

    # starting point
    points = [testPoint]
    xP = testPoint

    for x in range(0, 1000):
        xP = goldSearch(xP, gradFunc(xP))
        points.append(xP)

    plotPath(points)
    print(xP)
    print(func(xP))
    print()


def fibSearchExample():         # 2-16-2018         Not impressed

    def func(x):
        return x**4 - 14*(x**3) + 60*(x**2) - 70*x

    def fibNum(x):

        if x == 1 or x == 0:
            return x

        return fibNum(x-1) + fibNum(x-2)


    a = 0
    b = 2
    e = 0.001         # some small number
    N = 1

    print('iteration # 0', [a, b])

    finalRange = 0.0001    # change this to get more accurate answers - similar to demo2Challenge
    initialRange = b - a

    while fibNum(N) < (1 + 2*e)/(finalRange/initialRange):
        N += 1

    N -= 2    # correct for different Fibonacci sequences (with and without 0)

    iteration = 0
    while N != 0:
        rho = 1 - (fibNum(N+1)/fibNum(N+2))

        a1 = a + rho*(b-a)
        b1 = a + (1 - rho)*(b - a)

        f1 = func(a1)
        f2 = func(b1)

        if f1 > f2:
            a = a1
        else:
            b = b1

        N -= 1
        iteration += 1
        print('Iteration #', iteration, [a, b])

    return [a, b]


def newtonsMethodExample():         #2-17-2018

    def func(x):
        return 0.5*(x**2) - math.sin(x)

    def dfunc1(x):
        return x - math.cos(x)

    def dfunc2(x):
        return 1 + math.sin(x)

    def nextX(x):
        return x - (dfunc1(x)/dfunc2(x))

    x = 0.5             # set initial value here

    e = .000001         # accuracy  abs(x1 - x0)

    count = 0
    while True:
        xNext = nextX(x)

        count += 1

        if abs(xNext - x) < e:
            x = xNext
            break

        x = xNext

    print('Total # of iterations:', count)
    print('min @', [x, func(x)])

    return [x, func(x)]


def secantMethodExample1():      # 2-17-2018     Minimizes function

    def func(x):
        return 0.5*(x**2) - math.sin(x)

    def dfunc1(x):
        return x - math.cos(x)

    def nextX(xk, xk1):         # xk = x^k          xk1 = x^(k-1)
        return xk - dfunc1(xk)*((xk - xk1)/(dfunc1(xk) - dfunc1(xk1)))

    x1 = 13
    x2 = 12
    e = 0.000001        # accuracy  abs(x2-x1)

    count = 0
    while abs(x2-x1) > e:

        xNext = nextX(x2, x1)

        x1 = x2
        x2 = xNext

        count += 1
    print('Total # of iterations:', count)
    print('min @', [x2, func(x2)])

    return [x2, func(x2)]


def secantMethodExample2():     # 2-17-2018 finds nearest root of function

    def func(x):
        return x**3 - 12.2*(x**2) + 7.45*x + 42

    def nextX(xk, xk1):          # xk = x^k          xk1 = x^(k-1)
        return xk - func(xk)*((xk - xk1)/(func(xk)-func(xk1)))

    x1 = 20
    x2 = 19
    e = 0.0001          # accuracy abs(x2-x1)

    count = 0
    while abs(x2 - x1) > e:

        xNext = nextX(x2, x1)

        x1 = x2
        x2 = xNext

        count += 1

    print('Total # of iterations:', count)
    print('root @', x2)


def sincPlot3D():

    def func(x):
        return np.sinc(x[0])*np.sinc(x[1])

    def plotSinc():
        fig = plt.figure()

        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)

        Z = func([X, Y])

        ax = plt.axes(projection='3d')

        ax.plot_surface(X, Y, Z, cmap='gist_ncar', lw=.5, rstride=1, cstride=1)

        plt.show()

    plotSinc()


def simple1():  # just a quick thing tossed up in class

    def func(x):
        return np.cos(x)

    sol = minimize(func, 0)

    return sol
#######################################################

