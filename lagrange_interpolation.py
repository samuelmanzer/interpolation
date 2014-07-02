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
end = 2*math.pi


# Compute the Lagrange primitive coefficient 
n_eval_pts = 50
eval_step_size = float(end-start)/n_eval_pts
interp_range = range(2,8,2)
ys = [[] for i in range(0,len(interp_range))]
for eval_point_idx in range(0,n_eval_pts):
    eval_point = start + eval_point_idx*eval_step_size
    interp_idx = 0
    for n_interp_points in interp_range:
        interp_step_size = float(end-start)/n_interp_points
        val = 0
        for point_idx in range(0,n_interp_points):
            point = start + point_idx*interp_step_size
            weight = 1
            # Construct weight
            for other_point_idx in range(0,n_interp_points):
                other_point = start+other_point_idx*interp_step_size
                if point_idx != other_point_idx:
                    weight *= eval_point - other_point
                    weight /= (point - other_point)

            # Function eval
            val += weight*math.cos(point)

        # Store the value of this interpolation polynomial at the evaluation point
        ys[interp_idx].append(val)
        interp_idx +=1 

# Plot Lagrange interpolating polynomials
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.linspace(start,end,n_eval_pts)
interp_idx = 0
for y in ys:
    plt.plot(x,y,label=str(interp_range[interp_idx])+" Points")
    interp_idx += 1

# Plot exact result
line, = plt.plot(x,np.cos(x),label="cos(x)")

ax.legend(loc="upper left")
fig.savefig("lagrange_evenly_spaced.svg")
