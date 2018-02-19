import numpy as np

''' Various line search methods discussed in chapter 7 '''


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