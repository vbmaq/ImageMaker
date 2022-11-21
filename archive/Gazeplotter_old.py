from typing import Dict, List

import matplotlib
from matplotlib.colors import ListedColormap

from util import *
from scipy.stats import kde

# Original copy right
# This file is part of PyGaze - the open-source toolbox for eye tracking
#
# PyGazeAnalyser is a Python module for easily analysing eye-tracking data
# Copyright (C) 2014  Edwin S. Dalmaijer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>

# Gaze Plotter
#
# Produces different kinds of plots that are generally used in eye movement
# research, e.g. heatmaps, scanpaths, and fixation locations as overlays of
# images.
#
# version 2 (02 Jul 2014)

__author__ = "Edwin Dalmaijer"

# This current code below is a modified version of PyGaze.
# It has been modified to fit the needs of the data analysis we needed

__modified_by__ = "Adam Peter Frederick Reynolds"

# # # # #
# LOOK

# COLOURS
# all colours are from the Tango colourmap, see:
# http://tango.freedesktop.org/Tango_Icon_Theme_Guidelines#Color_Palette
# COLS = {"butter":     ['#fce94f',
#                        '#edd400',
#                        '#c4a000'],
#         "orange":     ['#fcaf3e',
#                        '#f57900',
#                        '#ce5c00'],
#         "chocolate":  ['#e9b96e',
#                        '#c17d11',
#                        '#8f5902'],
#         "chameleon":  ['#8ae234',
#                        '#73d216',
#                        '#4e9a06'],
#         "skyblue":    ['#729fcf',
#                        '#3465a4',
#                        '#204a87'],
#         "plum":       ['#ad7fa8',
#                        '#75507b',
#                        '#5c3566'],
#         "scarletred": ['#ef2929',
#                        '#cc0000',
#                        '#a40000'],
#         "aluminium":  ['#eeeeec',
#                        '#d3d7cf',
#                        '#babdb6',
#                        '#888a85',
#                        '#555753',
#                        '#2e3436'],
#         }
# # SHAPE texas old v.
# SHAPE = {"veryShort": ['.', 'b', 0.50],
#          "short":     ['.', 'r', 0.50],
#          "medium":    ['*', 'm', 1],
#          "long":      ['p', 'y', 1.5],
#          "veryLong":  ['X', 'w', 2]
#          }

# SHAPE ROI multicolor
'''
'^' - trinangle shapes
https://matplotlib.org/stable/api/markers_api.html
'''
SHAPE_ROI = {"own":     ['^', 'blue', 1.75],
             "other":   ['D', 'lime', 1.75],
             "outside": ['.', 'fuchsia', 1],
             }

SHAPE_ROI_UNIF = {"own":     ['^', 'green', 1.75],
                  "other":   ['^', 'green', 1.75],
                  "outside": ['^', 'green', 1.75],
                  }

AOI_COL = {"own":   'dimgray',
           "other": 'darkgray'
           }

AOI_LOC = {"own":   np.array(np.meshgrid([402, 832, 1258],
                                         [350, 615, 880]
                                         )).T.reshape(-1, 2),
           "other": np.array(np.meshgrid([650, 1078, 1500],
                                         [200, 468, 735]
                                         )).T.reshape(-1, 2)
           }


def get_shape_config(aoi_key, is_uniform=False):
	if is_uniform:
		return SHAPE_ROI_UNIF.get(aoi_key)
	else:
		return SHAPE_ROI.get(aoi_key)


def offset_aoi(aoi_dict, offset_x=0, offset_y=0):
	_aois = {}
	for key, value in aoi_dict.items():
		_aois[key] = value + [offset_x, offset_y]

	return _aois


def locate_aoi(x, y, aoi_dict, radius, default="outside"):
	for k, v in aoi_dict.items():
		if np.any(np.apply_along_axis(lambda a: np.linalg.norm(a - [x, y]) < radius, 1, v)):
			return k

	return default


def get_aoi_concentration(aoi_x, aoi_y, radius, xs, ys, offset_x, offset_y):
	within_x = (aoi_x + offset_x - radius < xs + offset_x) & (xs + offset_x < aoi_x + offset_y + radius)
	within_y = (aoi_y + offset_y - radius < ys + offset_y) & (ys + offset_y < aoi_y + offset_y + radius)
	return sum(within_x & within_y)


# FONT not adam:
FONT = {'family': 'Ubuntu',
        'size':   12}
matplotlib.rc('font', **FONT)


# # # # #
# FUNCTIONS

#<editor-fold desc="legacy">
def draw_scanpath_color(saccades, fig, dispsize, originalSize, cmap_saccades='winter', linewidth=2, imagefile=None,
                        loop=None, alpha=0.5, savefilename=None):
	"""Draws a scanpath: a series of lines between fixations,
	optionally drawn over an image.

	arguments

	saccades - a list of saccades events from a single trial,
		as produced by edfreader.read_edf, e.g.
		edfdata[trialnr]['events']['Esac']

	dispsize  - tuple or list indicating the size of the final display size,
		i.e. what dimension to display on
		e.g. (1024,768)

	originalSize  - tuple or list indicating the size of the original display size
		i.e. where the eyetracking data was collected from,
		e.g. (1024,768)

	fig  - fig of plt.fig

	 dispsize - dispsize of image

	keyword arguments
	 cmap_saccades  - the color map used for the saccades

	 linewidth  - width of saccade lines

	 loop  - if prodcuing multiple image loop = 'True' will clear the figure
				after saving the images

	imagefile  - full path to an image file over which the heatmap_wimg
		is to be laid, or None for no image; NOTE: the image
		may be smaller than the display size, the function
		assumes that the image was presented at the centre of
		the display (default = None)

	alpha  - float between 0 and 1, indicating the transparancy of
		the heatmap_wimg, where 0 is completely transparant and 1
		is completely untransparant (default = 0.5)

	savefilename - full path to the file in which the heatmap_wimg should be
		saved, or None to not save the file (default = None)

	returns

	fig   - a matplotlib.plt Figure instance, containing the
		heatmap_wimg
	"""

	# IMAGE
	fig, ax = draw_display(fig=fig, dispsize=dispsize, imagefile=imagefile)
	# to ensure the final image is correct in aspect ratio
	extraY = (dispsize[1] - originalSize[1]) / 2
	extraX = (dispsize[0] - originalSize[0]) / 2

	# get color of saccades
	n = len(saccades)
	colors = matplotlib.cm.get_cmap(cmap_saccades, n)
	# SACCADES
	if saccades:
		# loop through all saccades
		i = 0
		for st, et, dur, sx, sy, ex, ey in saccades:
			# draw an line between every saccade start and ending
			ax.plot([sx + extraX, ex + extraX], [sy + extraY, ey + extraY], c=colors(range(n))[i], linewidth=linewidth)
			i += 1
	# invert the y axis, as (0,0) is top left on a display
	ax.invert_yaxis()
	# save the figure if a file name was provided
	if savefilename != None:
		fig.savefig(savefilename)
	if loop != None:
		fig.clf()
	return fig


def draw_scanpath_fixations_color(saccades, fixations, fig, dispsize, originalSize,
                                  cmap_saccades='winter', linewidth=2,
                                  preserveSaccadeTemporalInfo=True, defaultSaccadeCol="green",
                                  setUniformFixations=False,
                                  radius=108,
                                  imagefile=None, loop=None, alpha=0.5, savefilename=None):
	"""Draws a scanpath: a series of lines between fixations.
	   Fixaions drawn as as blue Triangle(own payoff), green Diamond(other paroff) or pink dots(everywhere else).
	   Determiend by an economic game that was played by participants.
	   optionally drawn over an image.
	   CAN MAKE MULTIPLE !!!!!!!! :)

	arguments

	fixations  - a list of fixation ending events from a single trial,
		as produced by edfreader.read_edf, e.g.
		edfdata[trialnr]['events']['Efix']

	saccades - a list of saccades events from a single trial,
		as produced by edfreader.read_edf, e.g.
		edfdata[trialnr]['events']['Esac']

	fig  - fig of plt.fig

	dispsize  - tuple or list indicating the size of the final display size,
		i.e. what dimension to display on
		e.g. (1024,1024)

	originalSize  - tuple or list indicating the size of the original display size
		i.e. where the eyetracking data was collected from,
		e.g. (1024,768)

	keyword arguments
	cmap_saccades  - the color map used for the saccades

	 linewidth  - width of saccade lines

	 radius  - radius around the region of interest

	 loop  - if prodcuing multiple image loop = 'True' will clear the figure
				after saving the images

	imagefile  - full path to an image file over which the heatmap_wimg
		is to be laid, or None for no image; NOTE: the image
		may be smaller than the display size, the function
		assumes that the image was presented at the centre of
		the display (default = None)

	alpha  - float between 0 and 1, indicating the transparancy of
		the heatmap_wimg, where 0 is completely transparant and 1
		is completely untransparant (default = 0.5)

	savefilename - full path to the file in which the heatmap_wimg should be
		saved, or None to not save the file (default = None)

	returns

	fig   - a matplotlib.plt Figure instance,
	"""

	# IMAGE
	fig, ax = draw_display(fig=fig, dispsize=dispsize, imagefile=imagefile)
	# to ensure the final image is correct in aspect ratio
	extraY = (dispsize[1] - originalSize[1]) / 2
	extraX = (dispsize[0] - originalSize[0]) / 2

	# draw circles
	n = len(saccades)
	colors = matplotlib.cm.get_cmap(cmap_saccades, n)

	# SACCADES
	if saccades:
		n = len(saccades)
		if preserveSaccadeTemporalInfo:
			colors = matplotlib.cm.get_cmap(cmap_saccades, n)
			colors = colors(range(n))
		else:
			colors = [defaultSaccadeCol for _ in range(n)]

		# loop through all saccades
		i = 0
		for st, et, dur, sx, sy, ex, ey in saccades:
			# draw an line between every saccade start and ending
			ax.plot([sx + extraX, ex + extraX], [sy + extraY, ey + extraY], c=colors[i], linewidth=linewidth)
			i += 1

	# FIXATIONS
	if fixations:
		fix = parse_fixations(fixations)
		for i in range(len(fixations)):
			# _aoi = locate_aoi()
			# own
			if (((402 + extraX - radius < fix['x'][i] < 402 + extraX + radius) or
			     (832 + extraX - radius < fix['x'][i] < 832 + extraX + radius) or
			     (1258 + extraX - radius < fix['x'][i] < 1258 + extraX + radius)) & (
					(350 + extraY - radius < fix['y'][i] + extraY < 350 + extraY + radius) or
					(615 + extraY - radius < fix['y'][i] + extraY < 615 + extraY + radius) or
					(880 + extraY - radius < fix['y'][i] + extraY < 880 + extraY + radius))):
				ax.scatter(fix['x'][i] + extraX, fix['y'][i] + extraY,
				           s=get_shape_config('own', setUniformFixations)[2] * 1000,
				           c=get_shape_config('own', setUniformFixations)[1],
				           marker=get_shape_config('own', setUniformFixations)[0], alpha=alpha, edgecolors='white',
				           zorder=50)

			# other
			elif (((650 + extraX - radius < fix['x'][i] < 650 + extraX + radius) or
			       (1078 + extraX - radius < fix['x'][i] < 1078 + extraX + radius) or
			       (1500 + extraX - radius < fix['x'][i] < 1500 + extraX + radius)) & (
					      (200 + extraY - radius < fix['y'][i] + extraY < 200 + extraY + radius) or
					      (468 + extraY - radius < fix['y'][i] + extraY < 468 + extraY + radius) or
					      (735 + extraY - radius < fix['y'][i] + extraY < 735 + extraY + radius))):
				ax.scatter(fix['x'][i] + extraX, fix['y'][i] + extraY,
				           s=get_shape_config('other', setUniformFixations)[2] * 1000,
				           c=get_shape_config('other', setUniformFixations)[1],
				           marker=get_shape_config('other', setUniformFixations)[0], alpha=alpha, edgecolors='white',
				           zorder=50)

			# outside
			else:
				ax.scatter(fix['x'][i] + extraX, fix['y'][i] + extraY,
				           s=get_shape_config('outside', setUniformFixations)[2] * 1000,
				           c=get_shape_config('outside', setUniformFixations)[1],
				           marker=get_shape_config('outside', setUniformFixations)[0], alpha=alpha, edgecolors='white',
				           zorder=50)

	# invert the y axis, as (0,0) is top left on a display
	ax.invert_yaxis()
	# save the figure if a file name was provided
	if savefilename != None:
		fig.savefig(savefilename)
	if loop != None:
		fig.clf()
	return fig


def draw_scanpath_fixations_AOI(saccades, fixations, fig, dispsize, originalSize, cmap_saccades='winter',
                                cmap_fixations='magma',
                                linewidth=2, radius=108, imagefile=None, loop=None, alpha=0.5, savefilename=None,

                                preserveSaccadeTemporalInfo=True, defaultSaccadeCol="green", drawAOI=True
                                ):
	"""Draws a scanpath: a series of lines between fixations.
	   Draws a single fixation in the centre of a Area of Interest(AOI) that was determiend by an economic game display
	   that was played by participants.  The color of the fixations determiens numebr fo times the AOI was visited.
	   AOIs are depicted by a grey background. Optionally drawn over an image.

	arguments

	fixations  - a list of fixation ending events from a single trial,
		as produced by edfreader.read_edf, e.g.
		edfdata[trialnr]['events']['Efix']

	saccades - a list of saccades events from a single trial,
		as produced by edfreader.read_edf, e.g.
		edfdata[trialnr]['events']['Esac']

	fig  - fig of plt.fig

	dispsize  - tuple or list indicating the size of the final display size,
		i.e. what dimension to display on
		e.g. (1024,1024)

	originalSize  - tuple or list indicating the size of the original display size
		i.e. where the eyetracking data was collected from,
		e.g. (1024,768)

	keyword arguments
	cmap_saccades  - the color map used for the saccades

	cmap_fixations  - the color map used for the fixations

	 linewidth  - width of saccade lines

	 radius  - radius around the region of interest

	 loop  - if prodcuing multiple image loop = 'True' will clear the figure
				after saving the images

	imagefile  - full path to an image file over which the heatmap_wimg
		is to be laid, or None for no image; NOTE: the image
		may be smaller than the display size, the function
		assumes that the image was presented at the centre of
		the display (default = None)

	alpha  - float between 0 and 1, indicating the transparancy of
		the heatmap_wimg, where 0 is completely transparant and 1
		is completely untransparant (default = 0.5)

	savefilename - full path to the file in which the heatmap_wimg should be
		saved, or None to not save the file (default = None)

	returns

	fig   - a matplotlib.plt Figure instance,
	"""

	# IMAGE
	fig, ax = draw_display(fig=fig, dispsize=dispsize, imagefile=imagefile)

	# to ensure the final image is correct in aspect ratio
	extraY = (dispsize[1] - originalSize[1]) / 2
	extraX = (dispsize[0] - originalSize[0]) / 2

	# SACCADES
	if saccades:
		n = len(saccades)

		if preserveSaccadeTemporalInfo:
			colors = matplotlib.cm.get_cmap(cmap_saccades, n)
			colors = colors(range(n))
		else:
			colors = [defaultSaccadeCol for _ in range(n)]

		# loop through all saccades
		i = 0
		for st, et, dur, sx, sy, ex, ey in saccades:
			# draw an line between every saccade start and ending
			# TO MAKE SACCADES ONE COLOUR CHANGES TO GET ONE COLUR SACADE ITS AN ARRAY SO WATCH OUT C = GREEN.
			ax.plot([sx + extraX, ex + extraX], [sy + extraY, ey + extraY], c=colors[i], linewidth=linewidth,
			        zorder=49, alpha=alpha)
			i += 1

	# Areas of Interest
	ownX = [402, 832, 1258]
	ownY = [350, 615, 880]
	otherX = [650, 1078, 1500]
	otherY = [200, 468, 735]

	# color map of fixations
	nlevelColors = 20  # CHANGE TO 1 GET 1 COLOUR FOR PAPER WAS 20
	colorsFix = matplotlib.cm.get_cmap(cmap_fixations, nlevelColors)(np.arange(nlevelColors))

	for coordX in range(len(ownX)):
		for coordY in range(len(ownY)):
			if drawAOI:
				# plot grey circles MAYBE REMOVE FOR PAPER !
				circle1 = plt.Circle((ownX[coordX] + extraX, ownY[coordY] + extraY), radius, color='dimgray', zorder=48)
				ax.add_patch(circle1)
				circle2 = plt.Circle((otherX[coordX] + extraX, otherY[coordY] + extraY), radius, color='darkgray',
				                     zorder=48)
				ax.add_patch(circle2)

	if fixations is not None:
		fix = parse_fixations(fixations)

		for coordX in range(len(ownX)):
			for coordY in range(len(ownY)):
				# if drawAOI:
				# 	# plot grey circles MAYBE REMOVE FOR PAPER !
				# 	circle1 = plt.Circle((ownX[coordX] + extraX, ownY[coordY] + extraY), radius, color='dimgray', zorder=48)
				# 	ax.add_patch(circle1)
				# 	circle2 = plt.Circle((otherX[coordX] + extraX, otherY[coordY] + extraY), radius, color='darkgray',
				# 	                     zorder=48)
				# 	ax.add_patch(circle2)

				# FIXATIONS
				# plot Own
				# TO MAKE SACCADES ONE COLOUR CHANGES TO GET ONE COLUR SACADE ITS AN ARRAY SO WATCH OUT C = GREEN. COLOR=SHAPE_ROI['own'][1]
				fixOccOwn = sum(((ownX[coordX] + extraX - radius < fix['x'] + extraX) & (
						fix['x'] + extraX < ownX[coordX] + extraY + radius)) &
				                ((ownY[coordY] + extraY - radius < fix['y'] + extraY) & (
						                fix['y'] + extraY < ownY[coordY] + extraY + radius)))
				if nlevelColors > fixOccOwn > 0:
					ax.scatter(ownX[coordX] + extraX, ownY[coordY] + extraY, s=SHAPE_ROI['own'][2] * 1000,
					           color=colorsFix[fixOccOwn],
					           marker=SHAPE_ROI['own'][0], alpha=alpha, edgecolors='white', zorder=50)
				elif fixOccOwn >= nlevelColors:
					ax.scatter(ownX[coordX] + extraX, ownY[coordY] + extraY, s=SHAPE_ROI['own'][2] * 1000,
					           color=colorsFix[nlevelColors - 1],
					           marker=SHAPE_ROI['own'][0], alpha=alpha, edgecolors='white', zorder=50)

				# plot Other
				# TO MAKE SACCADES ONE COLOUR CHANGES TO GET ONE COLUR SACADE ITS AN ARRAY SO WATCH OUT C = GREEN. COLOR=SHAPE_ROI['OTHER'][1]
				fixOccOther = sum(((otherX[coordX] + extraX - radius < fix['x'] + extraX) & (
						fix['x'] + extraX < otherX[coordX] + extraX + radius)) &
				                  ((otherY[coordY] + extraY - radius < fix['y'] + extraY) & (
						                  fix['y'] + extraY < otherY[coordY] + extraY + radius)))
				if nlevelColors > fixOccOther > 0:
					ax.scatter(otherX[coordX] + extraX, otherY[coordY] + extraY, s=SHAPE_ROI['other'][2] * 1000,
					           color=colorsFix[fixOccOther],
					           marker=SHAPE_ROI['other'][0], alpha=alpha, edgecolors='white', zorder=50)
				elif fixOccOther >= nlevelColors:
					ax.scatter(otherX[coordX] + extraX, otherY[coordY] + extraY, s=SHAPE_ROI['other'][2] * 1000,
					           color=colorsFix[nlevelColors - 1],
					           marker=SHAPE_ROI['other'][0], alpha=alpha, edgecolors='white', zorder=50)

		# plot other, a longwinded way but I cannot seem to index the outside
		for i in range(len(fixations)):
			# own
			if (((402 - radius < fix['x'][i] < 402 + radius) or
			     (832 - radius < fix['x'][i] < 832 + radius) or
			     (1258 - radius < fix['x'][i] < 1258 + radius)) & (
					(350 + extraY - radius < fix['y'][i] + extraY < 350 + extraY + radius) or
					(615 + extraY - radius < fix['y'][i] + extraY < 615 + extraY + radius) or
					(880 + extraY - radius < fix['y'][i] + extraY < 880 + extraY + radius))):
				continue
			# other
			elif (((650 - radius < fix['x'][i] < 650 + radius) or
			       (1078 - radius < fix['x'][i] < 1078 + radius) or
			       (1500 - radius < fix['x'][i] < 1500 + radius)) & (
					      (200 + extraY - radius < fix['y'][i] + extraY < 200 + extraY + radius) or
					      (468 + extraY - radius < fix['y'][i] + extraY < 468 + extraY + radius) or
					      (735 + extraY - radius < fix['y'][i] + extraY < 735 + extraY + radius))):
				continue

			# outside
			else:
				ax.scatter(fix['x'][i], fix['y'][i] + extraY, s=SHAPE_ROI['outside'][2] * 1000,
				           c=SHAPE_ROI['outside'][1],
				           marker=SHAPE_ROI['outside'][0], alpha=alpha, edgecolors='white', zorder=50)

	# invert the y axis, as (0,0) is top left on a display
	ax.invert_yaxis()
	# save the figure if a file name was provided
	if savefilename is not None:
		fig.savefig(savefilename)
	if loop is not None:
		fig.clf()
	return fig
#</editor-fold>


def draw_heatmap(fixations, ax, x_offset=0, y_offset=0, alpha=1, cmap='PuRd', nbins=300, sample_strength=100):
	# cmap alpha
	# cmap = plt.cm.get_cmap(cmap)
	# my_cmap = cmap(np.arange(cmap.N))
	# my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
	# my_cmap = ListedColormap(my_cmap)
	# cmap = my_cmap

	fix = parse_fixations(fixations)

	x = fix['x'].repeat(sample_strength) + x_offset
	y = fix['y'].repeat(sample_strength) + y_offset

	k = kde.gaussian_kde([x, y])
	xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
	zi = k(np.vstack([xi.flatten(), yi.flatten()]))

	ax.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=alpha, cmap=cmap)


def draw_fixations(fixations, ax, x_offset=0, y_offset=0,
                   radius=108, set_uniform_fixations=False, alpha=0.5,
                   ):
	aois_offset = offset_aoi(AOI_LOC, x_offset, y_offset)

	# FIXATIONS
	fix = parse_fixations(fixations)
	for i in range(len(fixations)):
		_aoi = locate_aoi(fix['x'][i] + x_offset, fix['y'][i] + y_offset, aoi_dict=aois_offset, radius=radius,
		                  default="outside")
		ax.scatter(fix['x'][i] + x_offset, fix['y'][i] + y_offset,
		           s=get_shape_config(_aoi, set_uniform_fixations)[2] * 1000,
		           c=get_shape_config(_aoi, set_uniform_fixations)[1],
		           marker=get_shape_config(_aoi, set_uniform_fixations)[0], alpha=alpha, edgecolors='white',
		           zorder=50)


def draw_fixations_aggregate(fixations, ax, x_offset=0, y_offset=0, radius=108, set_uniform_fixations=False,
                             alpha=0.5, cmap_fixations='magma', n_level_colors=20):
	# color map of fixations
	colorsFix = matplotlib.cm.get_cmap(cmap_fixations, n_level_colors)(np.arange(n_level_colors))

	if fixations is not None:
		fix = parse_fixations(fixations)

		for key, aois in AOI_LOC.items():
			for aoi in aois:
				concentration = get_aoi_concentration(aoi_x=aoi[0], aoi_y=aoi[1], radius=radius,
				                                      xs=fix['x'], ys=fix['y'], offset_x=x_offset, offset_y=y_offset)
				if concentration <= 0:
					continue
				elif n_level_colors > concentration > 0:
					c_ = colorsFix[concentration]
				else:  # concentration >= nlevelColors:
					c_ = colorsFix[n_level_colors - 1]

				ax.scatter(aoi[0] + x_offset, aoi[1] + y_offset, s=get_shape_config(key, set_uniform_fixations)[2] * 1000,
				           color=c_,
				           marker=get_shape_config(key, set_uniform_fixations)[0], alpha=alpha, edgecolors='white', zorder=50)

		# plot outside aoi
		aois_offset = offset_aoi(AOI_LOC, x_offset, y_offset)

		for i in range(len(fixations)):
			_aoi = locate_aoi(fix['x'][i] + x_offset, fix['y'][i] + y_offset, aoi_dict=aois_offset, radius=radius,
			                  default="outside")
			if _aoi not in list(AOI_LOC.keys()):
				ax.scatter(fix['x'][i] + x_offset, fix['y'][i] + y_offset,
				           s=get_shape_config(_aoi, set_uniform_fixations)[2] * 1000,
				           c=get_shape_config(_aoi, set_uniform_fixations)[1],
				           marker=get_shape_config(_aoi, set_uniform_fixations)[0], alpha=alpha, edgecolors='white',
				           zorder=50)


def draw_saccades(saccades, ax, x_offset=0, y_offset=0,
                  preserve_saccade_temporal_info=True, cmap_saccades='winter', default_color='green'
                  , alpha=0.5, linewidth=2):
	"""

	:param saccades: sequence of (start time, end time, duration, start x, start y, end x, end y)
	:param ax:
	:param x_offset:
	:param y_offset:
	:param preserve_saccade_temporal_info: uses cmap_saccades if set to True else uses a default color
	:param cmap_saccades: default='winter' choose a sequential cmap here otherwise there's no temporal info
	:param default_color: this value is ignored if preserve_saccade_temporal_info is True
	:param alpha:
	:param linewidth:
	:return: fig
	"""
	n = len(saccades)

	if preserve_saccade_temporal_info:
		colors = matplotlib.cm.get_cmap(cmap_saccades, n)
		colors = colors(range(n))
	else:
		colors = [default_color for _ in range(n)]

	for i, (_, _, _, sx, sy, ex, ey) in enumerate(saccades):
		ax.plot([sx + x_offset, ex + x_offset], [sy + y_offset, ey + y_offset], c=colors[i], linewidth=linewidth,
		        zorder=49, alpha=alpha)


def draw_AOI(ax, x_offset=0, y_offset=0, radius=108):
	for key, aois in AOI_LOC.items():
		for aoi in aois:
			circle = plt.Circle((aoi[0] + x_offset, aoi[1] + y_offset), radius, color=AOI_COL.get(key), zorder=48)
			ax.add_patch(circle)


def draw_gaze(ax, x, y, x_offset=0, y_offset=0, color='white', marker='o', size=1.75, alpha=0.5, linewidth=0):
	ax.scatter(x + x_offset, y + y_offset, c=color, marker=marker, s=size, alpha=alpha, linewidth=linewidth)


