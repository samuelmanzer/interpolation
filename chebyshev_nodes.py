#!/usr/bin/env python
###############################################################################
# FILE: lagrange_interpolation.py
# AUTHOR: Samuel F. Manzer
# URL: http://www.samuelmanzer.com/
###############################################################################

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tempfile
import pdb
from lagrange_poly import *

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
lines,mae_list,rmsd_list,maxe_list = plot_lagrange_polys(x,n_points_range,evenly_spaced_polys,y_exact,ax1)
texts_1 = plot_stats(mae_list,rmsd_list,maxe_list,n_points_range,ax3)

f1.legend(lines,map(lambda x: str(x)+" Points",n_points_range)+["cos(x)"],loc="upper right")

# Chebyshev points - we must transform them to our interval
cp_sets = [ [ math.cos((float(2*k - 1)/(2*n))*math.pi) for k in range(1,n+1)] for n in n_points_range ]
tcp_sets = [ [ 0.5*((end - start)*pt + start + end) for pt in point_set] for point_set in cp_sets]
chebyshev_point_polys = [get_lagrange_poly(interp_points,math.cos) for interp_points in tcp_sets]
lines,mae_list,rmsd_list,maxe_list = plot_lagrange_polys(x,n_points_range,chebyshev_point_polys,y_exact,ax2)
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
    ax.set_ylim(ymax=max(heights)*1.05)

plt.show()
