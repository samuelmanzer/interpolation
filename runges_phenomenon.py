#!/usr/bin/env python
###############################################################################
# FILE: runges_phenomenon.py 
# AUTHOR: Samuel F. Manzer
# URL: http://www.samuelmanzer.com/
###############################################################################

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from lagrange_poly import *
import math


parser = ArgumentParser("Produces plots demonstrating the decline in accuracy with higher polynomial approximation")
args = parser.parse_args()

def runge_function(val):
    return float(1)/(1+25*val*val)


start = -1
end = 1
n_eval_pts = 1000

x = np.linspace(start,end,n_eval_pts)
y_exact = [runge_function(val) for val in x]
f1,ax1= plt.subplots()
f2,ax2= plt.subplots()

n_points_range = range(6,13,3)
# Runge's points
interp_point_sets = [[ float(2*i)/(n-1) - 1 for i in range(0,n) ] for n in n_points_range]
polys = [get_lagrange_poly(interp_points,runge_function) for interp_points in interp_point_sets]
lines,mae_list,rmsd_list,maxe_list = plot_lagrange_polys(x,n_points_range,polys,y_exact,ax1)
texts_1 = plot_stats(mae_list,rmsd_list,maxe_list,n_points_range,ax2)

plt.show()
