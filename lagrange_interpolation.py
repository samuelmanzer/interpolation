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

def plot_lagrange_polys(x,lagrange_polys,y_exact,ax):
    worst_points_x = []
    worst_points_y = []


    for poly_idx in range(0,len(lagrange_polys)):
        y = [ lagrange_polys[poly_idx](pt) for pt in x ]
        max_error = max(abs(y[i] - y_exact[i]) for i in range(0,len(x)))
        max_error_idx = filter(lambda i:  abs(y[i] - y_exact[i]) == max_error,range(0,len(x)))[0]
        worst_points_x.append(x[max_error_idx])
        worst_points_y.append(y[max_error_idx])
        ax.plot(x,y,label=str(n_points_range[poly_idx])+" Points")

    # Plot exact result
    ax.plot(x,y_exact,label="cos(x)")

    # Plot points where interpolation error is largest foreach degree 
    ax.scatter(worst_points_x,worst_points_y,c="r")

n_eval_pts = 1000
eval_step_size = float(end-start)/n_eval_pts
n_points_range = range(2,6,1)

x = np.linspace(start,end,n_eval_pts)
y_exact = np.cos(x)
f,(ax1,ax2) = plt.subplots(1,2,sharey=True)
ax1.set_ylim(ymin=-1.1,ymax=1.5)

# Equally spaced points
evenly_spaced_sets = [np.linspace(start,end,n_points) for n_points in n_points_range]
evenly_spaced_polys = [get_lagrange_poly(interp_points,math.cos) for interp_points in evenly_spaced_sets]
plot_lagrange_polys(x,evenly_spaced_polys,y_exact,ax1)



# Chebyshev points - we must transform them to our interval
cp_sets = [ [ math.cos((float(2*k - 1)/(2*n))*math.pi) for k in range(1,n+1)] for n in n_points_range ]
tcp_sets = [ [ 0.5*((end - start)*pt + start + end) for pt in point_set] for point_set in cp_sets]
chebyshev_point_polys = [get_lagrange_poly(interp_points,math.cos) for interp_points in tcp_sets]
plot_lagrange_polys(x,chebyshev_point_polys,y_exact,ax2)

#ax.legend(loc="lower right")
f.savefig("lagrange_evenly_spaced.svg")
