import matplotlib
matplotlib.use('TkAgg')

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


''' Various line search methods discussed in chapter 7 '''


def goldSearchExample(initialRange, accuracy, printIter=False, graph=False):
    """ Golden Search - Example 7.1, page 94
    :param initialRange: [min, max]
    :param accuracy: accuracy of final interval
    :param printIter: do you want to print each iteration to the console?
    :param graph: do you want to graph the function?
    :return: [a, b] a minimum value from the initial range lies within this range
    """
    def func(x):
        return x**4 - 14*(x**3) + 60*(x**2) - 70*x

    if initialRange[0] >= initialRange[1]:
        print('Fix initial range input')
        return

    # used for plotting at the end
    start1 = initialRange[0]
    start2 = initialRange[1]

    # changes each iteration
    a = initialRange[0]
    b = initialRange[1]

    # 1 - golden ratio
    rho = 0.382

    iterationOutput = [[a, b]]

    while accuracy < abs(b - a):
        a1 = a + rho * (b - a)
        b1 = a + ((1 - rho) * (b - a))

        f1 = func(a1)
        f2 = func(b1)

        if f1 > f2:
            a = a1
        else:
            b = b1

        iterationOutput.append([a, b])

    def printIterations():
        count = 0
        for x in iterationOutput:
            print('Iteration #', count, x)
            count +=1

    def graphFunc():

        fig = plt.figure()
        plt.xlim(-0.5, 7)
        plt.ylim(-25, 50)

        x = np.array(iterationOutput).ravel().tolist()

        x1 = x[0::2]
        x2 = x[1::2]

        graph, = plt.plot([], [], 'o', color='red')

        def animate(i):
            graph.set_data([x1[i], x2[i]], [func(np.array(x1[i])), func(np.array(x2[i]))])
            return graph

        t1 = np.linspace(-0.5, 7)
        plt.plot(t1, func(t1))

        ani = FuncAnimation(fig, animate, frames=len(x1)-1, interval=500)
        plt.show()

    if printIter:
        printIterations()
    if graph:
        graphFunc()

    return [a, b]

def fibSearchExample(initialRange, accuracy, printIter=False, graph=False):         # 2-16-2018         Not impressed
    """ Fibonacci Search - Example 7.2, page 98
    :param initialRange: [min, max]
    :param accuracy: accuracy of final interval
    :param printIter: do you want each iteration output printed to the console?
    :param graph: do you want the function graphed?
    :return: [a, b] a minimum value from the initial range lies within this range
    """

    # objective function
    def func(x):
        return x**4 - 14*(x**3) + 60*(x**2) - 70*x

    # retrieves Fibonacci number at a given index
    def fibNum(x):

        if x == 1 or x == 0:
            return x

        return fibNum(x-1) + fibNum(x-2)

    if initialRange[0] >= initialRange[1]:
        print('Fix initial range input')
        return

    # used for plotting at the end
    start1 = initialRange[0]
    start2 = initialRange[1]

    # changes each iteration
    a = initialRange[0]
    b = initialRange[1]

    e = 0.0001         # any small number
    N = 1

    iterationOutput = [[a, b]]

    finalRange = accuracy
    initialRange = b - a

    while fibNum(N) < (1 + 2*e)/(finalRange/initialRange):
        N += 1

    N -= 2    # correct for different Fibonacci sequences (with and without 0)

    while N != 0:
        rho = 1 - (fibNum(N+1)/fibNum(N+2))

        a1 = a + rho*(b-a)
        b1 = a + (1 - rho) *(b - a)

        f1 = func(a1)
        f2 = func(b1)

        if f1 > f2:
            a = a1
        else:
            b = b1

        N -= 1
        iterationOutput.append([a, b])

    def printIterations(iterationOutput):
        count = 0
        for x in iterationOutput:
            print('Iteration #', count, x)
            count += 1

    def graphFunc():
        fig = plt.figure()
        plt.xlim(-0.5, 7)
        plt.ylim(-25, 50)

        x = np.array(iterationOutput).ravel().tolist()

        x1 = x[0::2]
        x2 = x[1::2]

        graph, = plt.plot([], [], 'o', color='red')

        def animate(i):
            graph.set_data([x1[i], x2[i]], [func(np.array(x1[i])), func(np.array(x2[i]))])
            return graph

        t1 = np.linspace(-0.5, 7)
        plt.plot(t1, func(t1))

        ani = FuncAnimation(fig, animate, frames=len(x1) - 1, interval=500)
        plt.show()

    if printIter:
        printIterations(iterationOutput)
    if graph:
        graphFunc()

    return [a, b]


def newtonsMethodExample(x0, accuracy, printIter=False, graph=False):
    """ Secant Method - example from class - Example 7.3, Page 103
    :param x0: starting point
    :param accuracy: accuracy of minimum point
    :param printIter: do you want each iteration output printed to the console?
    :param graph: do you want the function graphed?
    :return: [x, func(x)] minimum point
    """

    # objective function
    def func(x):
        return 0.5*(x**2) - np.sin(x)

    # objective function 1st derivative
    def dfunc1(x):
        return x - np.cos(x)

    # objective function 2nd derivative
    def dfunc2(x):
        return 1 + np.sin(x)

    # retrieves next potential x value
    def nextX(x):
        return x - (dfunc1(x)/dfunc2(x))

    x = x0

    iterationOutput = [[x0, func(x0)]]

    while abs(nextX(x) - x) > accuracy:
        x = nextX(x)
        iterationOutput.append([x, func(x)])


    def printIterations(iterationOutput):
        count = 0
        for x in iterationOutput:
            print('Iteration #', count, x)
            count += 1

    def graphFunc():
        fig = plt.figure()
        plt.xlim(-10, 10)
        plt.ylim(-1, 20)


        x = [i[0] for i in iterationOutput]
        y = [i[1] for i in iterationOutput]

        graph, = plt.plot([], [], 'o', color='red')

        def animate(i):
            graph.set_data(x[i], y[i])
            return graph

        t1 = np.linspace(-10, 10)
        plt.plot(t1, func(t1))

        ani = FuncAnimation(fig, animate, frames=len(x) - 1, interval=500, repeat=False)
        plt.show()

    if printIter:
        printIterations(iterationOutput)
    if graph:
        graphFunc()

    return [x, func(x)]


def secantMethodExample1(a, b, accuracy, printIter=False, graph=False):
    """ Secant Method - Example 7.4, page 103 - Minimizing
    :param a: first x value
    :param b: second x value - keep them close
    :param printIter: do you want each iteration printed to the console
    :param graph: do you want the function graphed?
    :return: [x, func(x0]
    """
    def func(x):
        return 0.5*(x**2) - np.sin(x)

    def dfunc1(x):
        return x - np.cos(x)

    def nextX(xk, xk1):         # xk = x^k          xk1 = x^(k-1)
        return xk - dfunc1(xk)*((xk - xk1)/(dfunc1(xk) - dfunc1(xk1)))

    x1 = a
    x2 = b

    iterationOutput = [[x1, func(x1)], [x2, func(x2)]]

    while abs(x2-x1) > accuracy:

        xNext = nextX(x2, x1)

        x1 = x2
        x2 = xNext

        iterationOutput.append([x2, func(x2)])

    def printIterations(iterationOutput):
        count = 0
        for x in iterationOutput:
            print('Iteration #', count, x)
            count += 1

    def graphFunc():
        fig = plt.figure()
        plt.xlim(-10, 10)
        plt.ylim(-1,20)

        x = np.array(iterationOutput).ravel().tolist()
        x = x[0::2]
        y = x[1::2]

        graph, = plt.plot([], [], 'o', color = 'red')

        def animate(i):
            graph.set_data(x[i], y[i])
            return graph

        t1 = np.linspace(-10, 10)
        plt.plot(t1, func(t1))

        ani = FuncAnimation(fig, animate, frames = len(x) - 1, interval=500, repeat = False)
        plt.show()

    if printIter:
        printIterations(iterationOutput)
    if graph:
        graphFunc()

    return [x2, func(x2)]


def secantMethodExample2(a, b, accuracy, printIter=False, graph=False):
    """ Secant Method - Example 7.4, page 103 - Root Finding
    :param a: first x value
    :param b: second x value - keep them close
    :param printIter: do you want each iteration printed to the console
    :param graph: do you want the function graphed?
    :return: [x, func(x0]
    """

    def func(x):
        return x**3 - 12.2*(x**2) + 7.45*x + 42

    def nextX(xk, xk1):          # xk = x^k          xk1 = x^(k-1)
        return xk - func(xk)*((xk - xk1)/(func(xk)-func(xk1)))

    x1 = a
    x2 = b

    iterationOutput = [[x1, func(x2)], [x2, func(x2)]]

    while abs(x2-x1) > accuracy:

        xNext = nextX(x2, x1)

        x1 = x2
        x2 = xNext

        iterationOutput.append([x2, func(x2)])

    def printIterations(iterationOutput):
        count = 0
        for x in iterationOutput:
            print('Iteration #', count, x)
            count += 1

    def graphFunc():
        plt.figure()
        t1 = np.linspace(-5, 20.5)

        # plot the objective function
        plt.plot(t1, func(t1))
        plt.plot(t1, t1*0)      # x-axis

        plt.scatter(a, func(a), color='darkgreen')  # start points
        plt.scatter(b, func(b), color='g')          # start points
        plt.scatter(x2, func(x2), color='red')      # end point


        plt.show()

    if printIter:
        printIterations(iterationOutput)
    if graph:
        graphFunc()

    return [x2, func(x2)]
########################################################################################################################
''' Run code down here '''

secantMethodExample2(-10, 10, .00001, True, True)

