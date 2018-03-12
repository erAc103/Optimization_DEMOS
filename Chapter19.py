import matplotlib
matplotlib.use('TkAgg')
from scipy import optimize as opt
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

def rosenbrockMinCon():

    def func(x):
        return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

    def con2(x):
        return 0-x[0]-2*x[1]

    cons = [{'type':'ineq', 'fun':con2}]

    sol = opt.minimize(func, [-1,2], constraints=cons)

    print(sol)

