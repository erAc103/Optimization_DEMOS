import matplotlib
matplotlib.use('TkAgg')
from scipy import optimize as opt
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


''' Chapter 9 Newton's Method Examples '''


def example1(x0=[3,-1,0,1], iterations=3, printiter=True):

    def func(x):
        return (x[0] + 10*x[1])**2 + 5*(x[2] - x[3])**2 + (x[1] - 2*x[2])**4 + 10*(x[0] - x[3])**4

    def grad(x):
        return np.array([[2*(x[0] + 10*x[1]) + 40*(x[0] - x[3])**3],
                         [20*(x[0] + 10*x[1]) + 4*(x[1] - 2*x[2])**3],
                         [10*(x[2] - x[3]) - 8*(x[1] - 2*x[2])**3],
                         [-10*(x[2] - x[3]) - 40*(x[0] - x[3])**3]])

    def hess(x):
        return np.array([[2 + 120*(x[0]-x[3])**2, 20, 0, -120*(x[0]-x[3])**2],
                         [20, 200+12*(x[1] - 2*x[2])**2, -24*(x[1]-2*x[2])**2, 0],
                         [0, -24*(x[1]-2*x[2])**2, 10 + 48*(x[1]-2*x[2])**2, -10],
                         [-120*(x[0]-x[3])**2, 0, -10, 10+120*(x[0]-x[3])**2]])

    count = 0

    xs = [x0]
    x = x0

    while count < iterations:
        F = hess(x)
        invF = np.linalg.inv(F)
        g = grad(x)

        nextX = np.array([x]).T - np.matmul(invF, g) # numpy transpose not working

        xs.append(nextX.ravel())
        x = nextX.ravel()

        count +=1

    def printiterations():
        i = 0
        for x in xs:
            print('Iteration #',i,x)
            i += 1

    if printiter:
        printiterations()

    print(func(x))


def linRegression1(degree= '1'):

    def deg1():

        beta = np.random.randn(1, 2)

        def func(x):  # y
            return .5 + 0.5 * x

        def funcData(x):  # produces y data with some error
            return np.random.normal(x, .25)

        def fit(x, b):  # yHat
            return b[0] + b[1] * x

        x = np.linspace(0, 4, 30)
        y = funcData(func(x))
        ystuff = func(x)

        def cost(b):
            J = 0.5 * np.sum((y - fit(x, b)) ** 2)
            return J

        sol = opt.minimize(cost, beta)

        print(sol)
        beta = np.array([sol.x[0], sol.x[1]])

        fig = plt.figure()
        plt.xlim(0, 4)

        xspace = np.linspace(0, 4)
        plt.plot(xspace, fit(xspace, beta))
        plt.plot(x, y, 'o')
        plt.show()


    def deg2():

        beta = np.random.randn(1, 3)

        def func(x):  # y
            return .5 + 0.5 * x - 2 * x ** 2

        def funcData(x):  # produces y data with some error
            return np.random.normal(x, 4)

        def fit(x, b):  # yHat
            return b[0] + b[1] * x + b[2] * x ** 2

        x = np.linspace(0, 4, 30)
        y = funcData(func(x))
        ystuff = func(x)

        def cost(b):
            J = 0.5 * np.sum((y - fit(x, b)) ** 2)
            return J

        sol = opt.minimize(cost, beta)

        print(sol)
        beta = np.array([sol.x[0], sol.x[1], sol.x[2]])

        fig = plt.figure()
        plt.xlim(0,4)

        xspace = np.linspace(0, 4)
        plt.plot(xspace, fit(xspace, beta))
        plt.plot(x, y, 'o')
        plt.show()


    def deg3():

        beta = np.random.randn(1, 4)
        print(beta)

        def func(x): # y
            return .5 + 0.5*x - 2*x**2 + 1.1*x**3

        def funcData(x): # produces y data with some error
            return np.random.normal(x, 4)

        def fit(x, b): # yHat
            return b[0] + b[1]*x + b[2]*x**2 + b[3]*x**3

        x = np.linspace(0, 4, 30)
        y = funcData(func(x))
        ystuff = func(x)

        def cost(b):
            J = 0.5 * np.sum((y - fit(x, b)) ** 2)
            return J

        sol = opt.minimize(cost, beta)

        print(sol)
        beta = np.array([sol.x[0], sol.x[1], sol.x[2], sol.x[3]])

        fig = plt.figure()
        plt.xlim(0, 4)

        xspace = np.linspace(0, 4)
        plt.plot(xspace, fit(xspace, beta))
        plt.plot(x, y, 'o')
        plt.show()

    if degree == '1':
        deg1()
    elif degree == '2':
        deg2()
    elif degree == '3':
        deg3()
    else:
        print('Wrong degree number')


def nonLinearRegression():

    x = np.array([0.9, 1.5, 13.8, 19.8, 24.1, 28.2, 35.2, 60.3, 74.6, 81.3])
    x = 0.1*x
    y = np.array([455.2, 428.6, 124.1, 67.3, 43.2, 28.1, 13.1, -.4, -1.3, -1.5])

    beta = np.random.randn(1, 2)

    def fit(x, b):
        return b[0]*np.exp(x * b[1])

    def cost(b):
        J = 0.5 * np.sum((y - fit(x, b)) ** 2)
        return J

    sol = opt.minimize(cost, beta)

    print(sol)
    beta = np.array([sol.x[0], sol.x[1]])

    xspace = np.linspace(0, 10)
    plt.plot(xspace, fit(xspace, beta))
    plt.plot(x, y, 'o')
    plt.show()


def LMexample():

    x = np.array([0.9, 1.5, 13.8, 19.8, 24.1, 28.2, 35.2, 60.3, 74.6, 81.3])
    x = 0.1 * x
    y = np.array([455.2, 428.6, 124.1, 67.3, 43.2, 28.1, 13.1, -.4, -1.3, -1.5])

    beta = np.random.randn(1, 2)
    beta = np.array([[1, 2]])

    u = 0 # lambda value

    def fit(x, b):
        b = np.ravel(b)
        return b[0] * np.exp(x * b[1])

    def res(b):
        return y - fit(x, b)

    def jacRow(x, b):  # creates one row of jacobian
        return [np.exp(x*b[1]), b[0]*x*np.exp(x*b[1])]

    def jac(x, b):
        return np.array(jacRow(x, np.ravel(b))).T

    '''
    for i in range(0,1):

        J = jac(x, beta)

        r = np.array([res(beta)]).T

        # calculate metric
        JTJ = np.matmul(J.T, J)
        metric = np.add(JTJ, u * np.identity(2))
        metricInv = np.linalg.inv(metric)

        # calculate gradient
        grad = np.matmul(J.T, r)

        bOld = beta
        bNew = np.subtract(bOld.T, np.matmul(metricInv, grad))

        rNew = np.array([res(bNew)]).T

        cOld = 0.5*np.matmul(r.T, r)
        cNew = 0.5*np.matmul(rNew.T, rNew)
    '''

    beta = np.array([[20, .5]])
    #x = np.array([1, 2, 3, 4])
    #y = np.array([2, 4, 9, 12])

    for i in range(0, 20):

        try:
            print(beta, 'beta\n')
            r = np.array([res(beta)]).T

            print(r,'r\n')

            print(jac(x,beta), 'J\n')
            print(jac(x,beta).T, 'JT\n')
            JTJ = np.matmul(jac(x, beta).T, jac(x, beta))

            evals = np.linalg.eigvals(JTJ)

            #if min(evals) < 0:    # maybe???
            #    u = abs(min(evals))

            u = abs(min(evals))

            print(JTJ,'JTJ          eigvals', evals, '\n')

            uI = np.add(JTJ, u*np.identity(2))

            print(uI,'UI \n')

            inv = np.linalg.inv(uI)

            print(inv, 'INV \n')

            Jr = np.matmul(jac(x, beta).T, r)

            print(Jr,'Jr\n')

            fin = np.matmul(inv, Jr)

            print(fin, 'inv*Jr\n')

            beta = np.add(beta.T, fin).T    # why add?

            print(beta, '(beta-fin).T\n')

            print(i,'##########################################\n\n')

        except OverflowError:
            print('Overflow at iteration ', i)
            break

    print(beta)
    print(u)

