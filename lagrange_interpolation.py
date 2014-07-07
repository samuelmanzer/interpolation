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
import itertools
import tempfile
import pdb

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
def plot_lagrange_polys(x,lagrange_polys,y_exact,ax):
    mae_list = []
    rmsd_list = []
    maxe_list = []

    for poly_idx in range(0,len(lagrange_polys)):
        # Plot
        y = [ lagrange_polys[poly_idx](pt) for pt in x ]
        ax.plot(x,y,label=str(n_points_range[poly_idx])+" Points")

        # Compute stats
        abs_errors = [abs(y[i] - y_exact[i]) for i in range(0,len(y))]
        max_error = max(abs_errors)
        mae = sum(abs_errors)/float(len(abs_errors))
        rmsd = math.sqrt(sum(val*val for val in abs_errors)/float(len(abs_errors)))
        mae_list.append(mae)
        rmsd_list.append(rmsd)
        maxe_list.append(max(abs_errors))

    # Plot exact result
    ax.plot(x,y_exact,label="cos(x)")

    return mae_list,rmsd_list,maxe_list


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

parser = ArgumentParser("Produces plots of Lagrange interpolation of cos(x) for various numbers of Chebyshev and equally spaced points")
args = parser.parse_args()

start = 0
end = (5*math.pi)/2
n_eval_pts = 1000
eval_step_size = float(end-start)/n_eval_pts
n_points_range = range(2,6,1)

x = np.linspace(start,end,n_eval_pts)
y_exact = np.cos(x)
f1,(ax1,ax2) = plt.subplots(1,2,sharey=True)
ax1.set_ylim(ymin=-1.1,ymax=1.5)
ax1.set_title("Equally-Spaced")
ax2.set_title("Chebyshev")

f2,ax3 = plt.subplots()
f3,ax4 = plt.subplots()

# Equally spaced points
evenly_spaced_sets = [np.linspace(start,end,n_points) for n_points in n_points_range]
evenly_spaced_polys = [get_lagrange_poly(interp_points,math.cos) for interp_points in evenly_spaced_sets]
mae_list,rmsd_list,maxe_list = plot_lagrange_polys(x,evenly_spaced_polys,y_exact,ax1)
texts_1 = plot_stats(mae_list,rmsd_list,maxe_list,n_points_range,ax3)

# Chebyshev points - we must transform them to our interval
cp_sets = [ [ math.cos((float(2*k - 1)/(2*n))*math.pi) for k in range(1,n+1)] for n in n_points_range ]
tcp_sets = [ [ 0.5*((end - start)*pt + start + end) for pt in point_set] for point_set in cp_sets]
chebyshev_point_polys = [get_lagrange_poly(interp_points,math.cos) for interp_points in tcp_sets]
mae_list,rmsd_list,maxe_list = plot_lagrange_polys(x,chebyshev_point_polys,y_exact,ax2)
texts_2 = plot_stats(mae_list,rmsd_list,maxe_list,n_points_range,ax4)

ax3.set_title("Lagrange Interpolation with Equally-Spaced Points")
ax4.set_title("Lagrange Interpolation with Chebyshev Points")

# Awful haxx for text labels above bars to not get cut off by top of figure
tmp_file = tempfile.NamedTemporaryFile()
f2.savefig(tmp_file.name)
f3.savefig(tmp_file.name)
renderer_2 = f2.axes[0].get_renderer_cache()
renderer_3 = f3.axes[0].get_renderer_cache()
for (ax,renderer,texts) in [(ax3,renderer_2,texts_1),(ax4,renderer_3,texts_2)]:
    window_bbox_list = [t.get_window_extent(renderer) for t in texts]
    data_bbox_list = [b.transformed(ax.transData.inverted()) for b in window_bbox_list]
    data_coords_list = [b.extents for b in data_bbox_list] 
    heights = [ coords[-1] for coords in data_coords_list]
    #widths = [ coords[0] for coords in data_coords_list]
    ax.set_ylim(ymax=max(heights)*1.05)
    #ax.set_xlim(xmin=min(widths)*0.85)
plt.show()

#ax.legend(loc="lower right")
#f.savefig("lagrange_evenly_spaced.svg")
