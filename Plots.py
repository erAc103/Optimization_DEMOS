import matplotlib
matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np


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


def rastriginGraph():

    def func(x):
        return 20 + x[0]**2 + x[1]**2 - 10*(np.cos(np.pi * 2 * x[0]) + np.cos(np.pi * 2 * x[1]))

    def plotRastrigin():
        fig = plt.figure()

        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)

        Z = func([X, Y])

        ax = plt.axes(projection='3d')

        ax.plot_surface(X, Y, Z, cmap='terrain', lw=.5, rstride=1, cstride=1)

        plt.show()

    plotRastrigin()


def easomGraph():

    def func(x):
        return -1*np.cos(x[0])*np.cos(x[1])*np.exp(-1*(x[0]-np.pi)**2 - (x[1] - np.pi)**2)

    def plotEasom():
        fig = plt.figure()

        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)

        Z = func([X, Y])

        ax = plt.axes(projection='3d')

        ax.plot_surface(X, Y, Z, cmap='winter', lw=.5, rstride=1, cstride=1)

        plt.show()

    plotEasom()


def bealeGraph():

    def func(x):
        return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*(x[1]**2))**2 + (2.625 - x[0] + x[0]*(x[1]**3))**2

    def plotBeale():
        fig = plt.figure()

        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)

        Z = func([X, Y])

        ax = plt.axes(projection='3d')

        ax.plot_surface(X, Y, Z, cmap='gist_ncar', lw=.5, rstride=1, cstride=1)

        plt.show()

    plotBeale()
