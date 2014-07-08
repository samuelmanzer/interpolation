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
# FILE: lagrange_poly.py
# AUTHOR: Samuel F. Manzer
# URL: http://www.samuelmanzer.com/
#
# Utility functions for producing and plotting lagrange polynomials
###############################################################################

import math
import numpy as np
import matplotlib.pyplot as plt

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

# Plot the curves of the benchmark and the interpolating polynomials 
# Returns MAE and RMSD of each poly
def plot_lagrange_polys(x,n_points_range,lagrange_polys,y_exact,ax):
    lines = []
    mae_list = []
    rmsd_list = []
    maxe_list = []

    for poly_idx in range(0,len(lagrange_polys)):
        # Plot
        y = [ lagrange_polys[poly_idx](pt) for pt in x ]
        line, = ax.plot(x,y,label=str(n_points_range[poly_idx])+" Points")
        lines.append(line)

        # Compute stats
        abs_errors = [abs(y[i] - y_exact[i]) for i in range(0,len(y))]
        max_error = max(abs_errors)
        mae = sum(abs_errors)/float(len(abs_errors))
        rmsd = math.sqrt(sum(val*val for val in abs_errors)/float(len(abs_errors)))
        mae_list.append(mae)
        rmsd_list.append(rmsd)
        maxe_list.append(max(abs_errors))

    # Plot exact result
    line, = ax.plot(x,y_exact,label="cos(x)")
    lines.append(line)


    return lines,mae_list,rmsd_list,maxe_list


# Sets up a bar graph comparing the MAE and RMSD for different numbers
# of interpolation points
def plot_stats(mae_list,rmsd_list,maxe_list,n_points_range,ax):
    ind = np.arange(1,3*len(mae_list)+1,3)
    width = 0.75
    mae_rects = ax.bar(ind,mae_list,width,align="center",color='r')
    rmsd_rects = ax.bar(ind+width,rmsd_list,width,align="center",color='g')
    maxe_rects = ax.bar(ind+width+width,maxe_list,width,align="center",color='b')
    ticks = [ rect.get_x() for rect in rmsd_rects ]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(val) for val in n_points_range])
    ax.set_xlabel("# of Interpolation Points")
    ax.set_ylabel("Error Magnitude")
    ax.legend(["MAE","RMSD","MAXE"],loc="upper right")


    # Need to track these objects so we can adjust the axes to account for their size once
    # figure is drawn
    texts = []
    rect_idx = 0
    for rect in mae_rects+rmsd_rects+maxe_rects:
        height = rect.get_height()
        t = ax.text(rect.get_x()+rect.get_width()/2,1.05*height,'%.2f' % height,ha='center')
        texts.append(t)
        rect_idx += 1
    return texts
