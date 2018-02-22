import matplotlib
matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation
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
