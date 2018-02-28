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



