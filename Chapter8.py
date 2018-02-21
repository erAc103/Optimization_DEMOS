import matplotlib
matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


''' Various gradient methods discussed in class, in chapter 8 '''


def gradDescentConstantStep(initialPoint=[4.5, 5.5], accuracy=.6, stepsize=0.5, printIter=False, graph=True):
    """ Gradient Descent with constant step size - An example done in class
    :param initialPoint: [x, y] starting point
    :param accuracy: Stops when distance between current and previous point is less than accuracy
    :param stepsize: points shift by stepsize*gradient
    :param printIter: do you want each iteration printed?
    :param graph: do you want it graphed?
    :return: [a, b, func(x,y)] final minimum point
    """

    if accuracy < stepsize:
        return None

    def func(x):
        return 4*(x[0]**2) + x[1]**2

    def gradFunc(x):
        return [8*x[0], 2*x[1]]


    iterationOutput = [initialPoint]
    xP = initialPoint


    while np.linalg.norm(np.array(gradFunc(xP))) > accuracy:  # when magnitude of gradient is < accuracy... stop
        gradNorm = np.array(gradFunc(xP))/np.linalg.norm(np.array(gradFunc(xP)))

        xP = xP - stepsize * gradNorm

        iterationOutput.append(xP)


    def printIterations():
        count = 0
        for x in iterationOutput:
            print('Iteration #', count, x)
            count += 1

    def graphFunc():
        fig = plt.figure()
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)

        pts = np.array(iterationOutput).ravel().tolist()

        x = pts[0::2]
        y = pts[1::2]

        graph, = plt.plot([], [], '-o')

        def animate(i):
            graph.set_data(x[:i + 1], y[:i + 1]) # use this to keep points on the plot
            return graph

        t = np.linspace(-6, 6, 100)
        X, Y = np.meshgrid(t, t)
        Z = func([X, Y])

        plt.contour(X, Y, Z, 25, cmap='gist_ncar')

        ani = FuncAnimation(fig, animate, frames=len(x)-1, interval=300)

        # you need imagemagick for this - it saves as gif
        # ani.save('gradDescentConstantStep.gif', dpi=80, writer='imagemagick')

        plt.show()


    if printIter:
        printIterations()
    if graph:
        graphFunc()

    return [xP[0], xP[1], func(xP)]


def gradDescent1(initialPoint=[4.5, 5], alpha=.05, accuracy=0.4, printiter=False, graph=True):
    """
    :param initialPoint: what it sounds like
    :param alpha: sets constant alpha value
    :param accuracy: stops when gradient magnitude gets below this number
    :param printiter: do you want each iteration printed
    :param graph: do you want it graphed
    :return: [x, y, func(x,y)]
    """

    def func(x):
        return 4 * (x[0] ** 2) + x[1] ** 2

    def gradFunc(x):
        return [8 * x[0], 2 * x[1]]

    iterationOutput = [initialPoint]
    xP = initialPoint

    while np.linalg.norm(np.array(gradFunc(xP))) > accuracy:  # when magnitude of gradient is < accuracy... stop

        xP = xP - alpha * np.array(gradFunc(xP))

        iterationOutput.append(xP)

    def printIterations():
        count = 0
        for x in iterationOutput:
            print('Iteration #', count, x)
            count += 1

    def graphFunc():
        fig = plt.figure()
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)

        pts = np.array(iterationOutput).ravel().tolist()

        x = pts[0::2]
        y = pts[1::2]

        graph, = plt.plot([], [], '-o')

        def animate(i):
            graph.set_data(x[:i + 1], y[:i + 1])  # use this to keep points on the plot
            return graph

        t = np.linspace(-6, 6, 100)
        X, Y = np.meshgrid(t, t)
        Z = func([X, Y])

        plt.contour(X, Y, Z, 25, cmap='gist_ncar')

        ani = FuncAnimation(fig, animate, frames=len(x) - 1, interval=300)

        # you need imagemagick for this - it saves as gif
        # ani.save('gradDescentConstantStep.gif', dpi=80, writer='imagemagick')

        plt.show()

    if printiter:
        printIterations()
    if graph:
        graphFunc()

    return [xP[0], xP[1], func(xP)]

