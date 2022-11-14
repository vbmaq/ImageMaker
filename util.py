import numpy as np
import matplotlib.pyplot as plt
import os

# # # # #
# HELPER FUNCTIONS


def draw_display(fig, dispsize=(1920, 1080), imagefile=None, returnOffset=False):
	"""Returns a matplotlib.plt Figure and its axes, with a size of
	dispsize, a black background colour, and optionally with an image drawn
	onto it

	arguments
	 fig  - fig of plt.fig

	dispsize  - tuple or list indicating the size of the display,
		e.g. (1024,768)

	keyword arguments

	imagefile  - full path to an image file over which the heatmap_wimg
		is to be laid, or None for no image; NOTE: the image
		may be smaller than the display size, the function
		assumes that the image was presented at the centre of
		the display (default = None)

	returns
	fig, ax  - matplotlib.plt Figure and its axes: field of zeros
		with a size of dispsize, and an image drawn onto it
		if an imagefile was passed
	"""
	x, y = 0,0
	# construct screen (black background)
	data_type = 'float32'
	if imagefile != None:
		_, ext = os.path.splitext(imagefile)
		ext = ext.lower()
		data_type = 'float32' if ext == '.png' else 'uint8'
	screen = np.zeros((dispsize[1], dispsize[0], 3), dtype=data_type)
	# if an image location has been passed, draw the image
	if imagefile != None:
		# check if the path to the image exists
		if not os.path.isfile(imagefile):
			raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
		# load image
		img = plt.imread(imagefile)

		# flip image over the horizontal axis
		# (do not do so on Windows, as the image appears to be loaded with
		# the correct side up there; what's up with that? :/)
		# if not os.name == 'nt':
		# img = np.flipud(img)
		# width and height of the image
		w, h = len(img[0]), len(img)
		# x and y position of the image on the display
		x = int(dispsize[0] / 2 - w / 2)
		y = int(dispsize[1] / 2 - h / 2)
		# draw the image on the screen
		screen[y:y + h, x:x + w, :] += img
	# dots per inch
	dpi = 100.0
	# determine the figure size in inches
	# figsize = (dispsize[0]/dpi, dispsize[1]/dpi)
	# create a figure
	# fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
	ax = plt.Axes(fig, [0, 0, 1, 1])
	ax.set_axis_off()
	fig.add_axes(ax)
	# plot display
	ax.axis([0, dispsize[0], 0, dispsize[1]])

	ax.imshow(screen)  # , origin='upper')

	if imagefile and returnOffset:
		return fig, ax, x, y
	else:
		return fig, ax


def gaussian(x, sx, y=None, sy=None):
	"""Returns an array of np arrays (a matrix) containing values between
	1 and 0 in a 2D Gaussian distribution

	arguments
	x  -- width in pixels
	sx  -- width standard deviation

	keyword argments
	y  -- height in pixels (default = x)
	sy  -- height standard deviation (default = sx)
	"""

	# square Gaussian if only x values are passed
	if y == None:
		y = x
	if sy == None:
		sy = sx
	# centers
	xo = x / 2
	yo = y / 2
	# matrix of zeros
	M = np.zeros([y, x], dtype=float)
	# gaussian matrix
	for i in range(x):
		for j in range(y):
			M[j, i] = np.exp(-1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))

	return M


def parse_fixations(fixations):
	"""Returns all relevant data from a list of fixation ending events

	arguments

	fixations  - a list of fixation ending events from a single trial,
		as produced by edfreader.read_edf, e.g.
		edfdata[trialnr]['events']['Efix']

	returns

	fix  - a dict with three keys: 'x', 'y', and 'dur' (each contain
	   a np array) for the x and y coordinates and duration of
	   each fixation
	"""

	# empty arrays to contain fixation coordinates
	fix = {'x':   np.zeros(len(fixations)),
	       'y':   np.zeros(len(fixations)),
	       'dur': np.zeros(len(fixations))}
	# get all fixation coordinates
	for fixnr in range(len(fixations)):
		stime, etime, dur, ex, ey = fixations[fixnr]
		fix['x'][fixnr] = ex
		fix['y'][fixnr] = ey
		fix['dur'][fixnr] = dur

	return fix








