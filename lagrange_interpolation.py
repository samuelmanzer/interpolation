#!/usr/bin/env python
###############################################################################
# FILE: lagrange_interpolation.py
# AUTHOR: Samuel F. Manzer
# URL: http://www.samuelmanzer.com/
###############################################################################

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import math


###############################################################################
# OPTION PARSING
###############################################################################
parser = ArgumentParser("Produces plots of Lagrange interpolation of cos(x) for various numbers of Chebyshev and equally spaced points")
args = parser.parse_args()


###############################################################################
# END OPTION PARSING 
###############################################################################

start = 0
end = (5*math.pi)/2

def get_lagrange_poly(interp_points,fn_to_eval):
    def lagrange_poly(eval_point):
        val = 0
        for cur_interp_point in interp_points:
            weight = 1
            # Construct weight
            for other_interp_point in interp_points:
                if other_interp_point != cur_interp_point:
                    weight *= eval_point - other_interp_point
                    weight /= (cur_interp_point - other_interp_point)
            # Function eval
            val += weight*fn_to_eval(cur_interp_point)
        return val 
    return lagrange_poly 


n_eval_pts = 50
eval_step_size = float(end-start)/n_eval_pts
n_points_range = range(2,6,1)
interp_point_sets = [np.linspace(start,end,n_points) for n_points in n_points_range]
lagrange_polys = [get_lagrange_poly(interp_points,math.cos) for interp_points in interp_point_sets]

# Plot Lagrange interpolating polynomials
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.linspace(start,end,n_eval_pts)
for poly_idx in range(0,len(lagrange_polys)):
    y = [ lagrange_polys[poly_idx](pt) for pt in x ]
    plt.plot(x,y,label=str(n_points_range[poly_idx])+" Points")

# Plot exact result
line, = plt.plot(x,np.cos(x),label="cos(x)")

ax.legend(loc="lower right")
ax.set_ylim(ymin=-1.5,ymax=1.5)
fig.savefig("lagrange_evenly_spaced.svg")
