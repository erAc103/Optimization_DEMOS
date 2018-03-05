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


def linRegression(points=[[1,2], [2,1], [3,3], [4,3], [5,6], [4,6], [2,0.75]]):

    beta = np.random.randn(1,2)

    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])

    def func(x, b):
        b = b.ravel()
        yHat = b[0] + x*b[1]
        return yHat

    def cost(b):
        J = 0.5 * np.sum((y-func(x, b))**2)
        return J

    def fit(x):
        return beta[0] + x*beta[1]

    sol = opt.minimize(cost, beta)

    beta = np.array([sol.x[0], sol.x[1]])
    print(beta)

    fig = plt.figure()
    plt.ylim(0,7)
    plt.xlim(0,7)

    xspace = np.linspace(0,7)
    print(xspace)
    plt.plot(xspace, fit(xspace))
    plt.plot(x,y, 'o')
    plt.show()

