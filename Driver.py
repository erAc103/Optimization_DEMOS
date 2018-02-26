import Chapter7 as ch7
import Chapter8 as ch8
import numpy as np
import Plots

''' Toss code in here to test instead of at the end of the scripts'''

if __name__ == '__main__':

    A = np.array([[3,2],[2,3]])
    b = np.array([[5],[5]])

    print(ch8.matLeftDivide(A, b))



