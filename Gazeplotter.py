import matplotlib
from util import *

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
COLS = {"butter":     ['#fce94f',
                       '#edd400',
                       '#c4a000'],
        "orange":     ['#fcaf3e',
                       '#f57900',
                       '#ce5c00'],
        "chocolate":  ['#e9b96e',
                       '#c17d11',
                       '#8f5902'],
        "chameleon":  ['#8ae234',
                       '#73d216',
                       '#4e9a06'],
        "skyblue":    ['#729fcf',
                       '#3465a4',
                       '#204a87'],
        "plum":       ['#ad7fa8',
                       '#75507b',
                       '#5c3566'],
        "scarletred": ['#ef2929',
                       '#cc0000',
                       '#a40000'],
        "aluminium":  ['#eeeeec',
                       '#d3d7cf',
                       '#babdb6',
                       '#888a85',
                       '#555753',
                       '#2e3436'],
        }
# SHAPE texas old v.
SHAPE = {"veryShort": ['.', 'b', 0.50],
         "short":     ['.', 'r', 0.50],
         "medium":    ['*', 'm', 1],
         "long":      ['p', 'y', 1.5],
         "veryLong":  ['X', 'w', 2]
         }

# SHAPE ROI multicolor
'''
'^' - trinangle shapes
https://matplotlib.org/stable/api/markers_api.html
'''
SHAPE_ROI = {"own":     ['^', 'blue', 1.75],
             "other":   ['D', 'lime', 1.75],
             "outside": ['.', 'fuchsia', 1],
             }

# FONT not adam:
FONT = {'family': 'Ubuntu',
        'size':   12}
matplotlib.rc('font', **FONT)


# # # # #
# FUNCTIONS

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

	imagefile  - full path to an image file over which the heatmap
		is to be laid, or None for no image; NOTE: the image
		may be smaller than the display size, the function
		assumes that the image was presented at the centre of
		the display (default = None)

	alpha  - float between 0 and 1, indicating the transparancy of
		the heatmap, where 0 is completely transparant and 1
		is completely untransparant (default = 0.5)

	savefilename - full path to the file in which the heatmap should be
		saved, or None to not save the file (default = None)

	returns

	fig   - a matplotlib.plt Figure instance, containing the
		heatmap
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


def draw_scanpath_fixations_color(saccades, fixations, fig, dispsize, originalSize, cmap_saccades='winter', linewidth=2,
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

	imagefile  - full path to an image file over which the heatmap
		is to be laid, or None for no image; NOTE: the image
		may be smaller than the display size, the function
		assumes that the image was presented at the centre of
		the display (default = None)

	alpha  - float between 0 and 1, indicating the transparancy of
		the heatmap, where 0 is completely transparant and 1
		is completely untransparant (default = 0.5)

	savefilename - full path to the file in which the heatmap should be
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
		# loop through all saccades
		i = 0
		for st, et, dur, sx, sy, ex, ey in saccades:
			# draw an line between every saccade start and ending
			ax.plot([sx + extraX, ex + extraX], [sy + extraY, ey + extraY], c=colors(range(n))[i], linewidth=linewidth)
			i += 1

	# FIXATIONS
	if fixations:
		fix = parse_fixations(fixations)
		for i in range(len(fixations)):
			# own
			if (((402 + extraX - radius < fix['x'][i] < 402 + extraX + radius) or
			     (832 + extraX - radius < fix['x'][i] < 832 + extraX + radius) or
			     (1258 + extraX - radius < fix['x'][i] < 1258 + extraX + radius)) & (
					(350 + extraY - radius < fix['y'][i] + extraY < 350 + extraY + radius) or
					(615 + extraY - radius < fix['y'][i] + extraY < 615 + extraY + radius) or
					(880 + extraY - radius < fix['y'][i] + extraY < 880 + extraY + radius))):
				ax.scatter(fix['x'][i] + extraX, fix['y'][i] + extraY, s=SHAPE_ROI['own'][2] * 1000, c=SHAPE_ROI['own'][1],
				           marker=SHAPE_ROI['own'][0], alpha=alpha, edgecolors='white', zorder=50)

			# other
			elif (((650 + extraX - radius < fix['x'][i] < 650 + extraX + radius) or
			       (1078 + extraX - radius < fix['x'][i] < 1078 + extraX + radius) or
			       (1500 + extraX - radius < fix['x'][i] < 1500 + extraX + radius)) & (
					      (200 + extraY - radius < fix['y'][i] + extraY < 200 + extraY + radius) or
					      (468 + extraY - radius < fix['y'][i] + extraY < 468 + extraY + radius) or
					      (735 + extraY - radius < fix['y'][i] + extraY < 735 + extraY + radius))):
				ax.scatter(fix['x'][i] + extraX, fix['y'][i] + extraY, s=SHAPE_ROI['other'][2] * 1000,
				           c=SHAPE_ROI['other'][1],
				           marker=SHAPE_ROI['other'][0], alpha=alpha, edgecolors='white', zorder=50)

			# outside
			else:
				ax.scatter(fix['x'][i] + extraX, fix['y'][i] + extraY, s=SHAPE_ROI['outside'][2] * 1000,
				           c=SHAPE_ROI['outside'][1],
				           marker=SHAPE_ROI['outside'][0], alpha=alpha, edgecolors='white', zorder=50)

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

                                preserveSaccadeTemporalInfo=True, defaultSaccadeCol = "green", drawAOI = True
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

	imagefile  - full path to an image file over which the heatmap
		is to be laid, or None for no image; NOTE: the image
		may be smaller than the display size, the function
		assumes that the image was presented at the centre of
		the display (default = None)

	alpha  - float between 0 and 1, indicating the transparancy of
		the heatmap, where 0 is completely transparant and 1
		is completely untransparant (default = 0.5)

	savefilename - full path to the file in which the heatmap should be
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
			        zorder=49)
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
				ax.scatter(fix['x'][i], fix['y'][i] + extraY, s=SHAPE_ROI['outside'][2] * 1000, c=SHAPE_ROI['outside'][1],
				           marker=SHAPE_ROI['outside'][0], alpha=alpha, edgecolors='white', zorder=50)

			# invert the y axis, as (0,0) is top left on a display
	ax.invert_yaxis()
	# save the figure if a file name was provided
	if savefilename is not None:
		fig.savefig(savefilename)
	if loop is not None:
		fig.clf()
	return fig


