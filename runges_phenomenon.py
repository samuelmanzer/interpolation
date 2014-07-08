#!/usr/bin/env python
###############################################################################
# Interpolation 
# Copyright (C) Samuel F. Manzer. All rights reserved.
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3.0 of the License, or (at your option) any later version.
# 
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with this library.
# 
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
f1.legend(lines,map(lambda x: str(x)+" Points",n_points_range)+["1/(1+25x^2)"],loc="upper right")

texts_1 = plot_stats(mae_list,rmsd_list,maxe_list,n_points_range,ax2)
ax2.set_title("Interpolation Errors for Various Numbers of Points")


plt.show()
