import matplotlib.pyplot as plt
import numpy as np

''' Various line search methods discussed in chapter 7 '''


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
        b1 = a + (1 - rho)*(b - a)

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
            count +=1

    def graphFunc():
        plt.figure()
        t1 = np.linspace(-0.5, 7.25)

        # plot the function
        plt.plot(t1, func(t1))

        # plot points at initial range
        plt.scatter(start1, func(start1), color='lime')
        plt.scatter(start2, func(start2), color='lime')

        # plot points at final range
        plt.scatter(a, func(a), color='red')
        plt.scatter(b, func(b), color='red')

        plt.show()

    if printIter:
        printIterations(iterationOutput)
    if graph:
        graphFunc()

    return [a, b]


########################################################################################################################
''' Run code down here '''


