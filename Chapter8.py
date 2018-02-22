import matplotlib
matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


''' Various gradient methods discussed in class, in chapter 8 '''


def gradDescentConstantStep(initialPoint=[4.5, 5.5], accuracy=.6, stepsize=0.5, printIter=False, graph=True):
    """ Gradient Descent with constant step size - An example done in class
    :param initialPoint: [x, y] starting point
    :param accuracy: Stops when magnitude of gradient is less than this number
    :param stepsize: points shift by stepsize*gradient
    :param printIter: do you want each iteration printed?
    :param graph: do you want it graphed?
    :return: [a, b, func(x,y)] final minimum point
    """

    #if accuracy < stepsize:
     #   return None

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


def gradDescent1(initialPoint=[4.5, 5], alpha=.05, accuracy=0.001, printiter=False, graph=True):
    """ Gradient step with varying step sizes, depends on magnitude of gradient*alpha
    :param initialPoint: what it sounds like
    :param alpha: sets constant alpha value
    :param accuracy: stops when distance between prev and current point is less than this number
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

    while True:

        xNext = xP - alpha * np.array(gradFunc(xP))

        if abs(np.linalg.norm(np.array(xNext) - np.array(xP))) < accuracy:
            xP = xNext
            iterationOutput.append(xP)
            break

        xP = xNext
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


def gradDescent2(initialPoint=[4.5, 5], accuracy=0.01, printiter=False, graph=True):
    """ Example from class - Gradient descent with using golden search
    :param initialPoint: [x,y] starting point
    :param accuracy: stops when distance between two points is less than this number
    :param printiter: prints each iteration
    :param graph: graphs function
    :return: [x, y, f(x,y)]
    """

    def func(x):
        return 4*(x[0]**2) + x[1]**2

    def gradFunc(x):
        return [8*x[0], 2*x[1]]

    def lineFunc(x, grad, t):
        return np.array(x) - t*np.array(grad)

    def goldSearch(x, grad):
        a0 = -.0001
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


    iterationOutput = [initialPoint]
    xP = initialPoint

    while True:

        xNext = goldSearch(xP, gradFunc(xP))

        if abs(np.linalg.norm(np.array(xNext) - np.array(xP))) < accuracy:
            xP = xNext
            iterationOutput.append(xP)
            break

        xP = xNext
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


def gradDescent3(initialPoint=[4.5, 5], accuracy=0.001, printiter=False, graph=True):
    """ Rosenbrock function - Gradient descent with using golden search
    :param initialPoint: [x,y] starting point
    :param accuracy: stops when distance between two points is less than this number
    :param printiter: prints each iteration
    :param graph: graphs function
    :return: [x, y, f(x,y)]
    """

    def func(x):
        return (1 - x[0])**2 + (x[1]-x[0]**2)**2

    def gradFunc(x):
        return [4*x[0]**3 - 4*x[0]*x[1] + 2*x[0] - 2, 2*(x[1] - x[0]**2)]

    def lineFunc(x, grad, t):
        return np.array(x) - t*np.array(grad)

    def goldSearch(x, grad):
        a0 = -.0001
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


    iterationOutput = [initialPoint]
    xP = initialPoint

    while True:

        xNext = goldSearch(xP, gradFunc(xP))

        if abs(np.linalg.norm(np.array(xNext) - np.array(xP))) < accuracy:
            xP = xNext
            iterationOutput.append(xP)
            break

        xP = xNext
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
