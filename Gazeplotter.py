from enum import Enum
from typing import Dict, List, Union
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import ndarray
from scipy.stats import kde
import os

X = 0
Y = 1
DUR = 2
MARKER = 0
COLOR = 1
SIZE = 2


def draw_display(fig, dispsize=(1920, 1080), imagefile=None, returnOffset=False, backgroundValue: float=0):
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

	background - a grayscale value from 0 to 1 (1 = white)

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
	screen = np.zeros((dispsize[1], dispsize[0], 3), dtype=data_type) + backgroundValue


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


def calculate_offset(dispSize, originalSize):
	"""
	Returns x and y offset
	:param dispSize: display size (w, h)
	:param originalSize: original size (w, h)
	:return: offset_x, offset_y
	"""
	offset_x = (dispSize[X] - originalSize[X]) / 2
	offset_y = (dispSize[Y] - originalSize[Y]) / 2
	return offset_x, offset_y


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


class Draw(Enum):
	AOI = "draw_AOI"
	HEATMAP = "draw_heatmap"
	GAZE = "draw_gaze"
	SACCADES = "draw_saccades"
	FIXATIONS = "draw_fixations"
	FIXATIONS_AGG = "draw_fixations_aggregate"


class GazePlotter:
	fixations: Union[ndarray, None]
	saccades: Union[ndarray, None]
	gaze: Union[ndarray, None]

	def __init__(self, shape_roi: Dict, shape_roi_unif: List, aoi_col: Dict, aoi_loc: Dict,
	             fig=None, ax=None,
	             saccades: Union[List, ndarray]=None, fixations: Union[List, ndarray]=None, gaze: Union[List, ndarray]=None,
	             x_offset: int = 0, y_offset: int = 0,
	             # display_size: tuple, original_size: tuple, imagefile: str = None,
	             save_dir: str = ""
	             ):
		"""
		:param shape_roi: Dict of keys=aoi containing values of type list of roi shape configurations [marker, color, size]
		:param shape_roi_unif: List of shape configuration that is uniform to all aoi's [marker, color, size]
		:param aoi_col: Dict of keys=aoi containing the color for each aoi
		:param aoi_loc: Dict of keys=aoi containing the location for each aoi. The value is a list of coordinates.
		:param saccades: sequence of [start time, end time, duration, start x, start y, end x, end y] shape: (numSaccades, 7)
		:param fixations: sequence of [x, y, duration] shape: (numFixations, 3)
		:param gaze: sequence of [x, y] shape: (numGaze, 2)
		"""

		self.shape_roi: Dict[str, List] = shape_roi

		self.shape_roi_uniform = {}
		for key in self.shape_roi.keys():
			self.shape_roi_uniform[key] = shape_roi_unif

		self.aoi_color: Dict[str, str] = aoi_col
		self.aoi_location: Dict[str, List] = aoi_loc

		self.fig = fig
		self.ax = ax

		self.x_offset: int = x_offset
		self.y_offset: int = y_offset

		self.fixations: Union[ndarray, None] = np.array(fixations)
		self.saccades: Union[ndarray, None] = np.array(saccades)
		self.gaze: Union[ndarray, None] = np.array(gaze)

		self.save_dir = save_dir

	def run_pipeline(self, pipe: Dict):
		for fun, params in pipe.items():
			if params:
				getattr(self, fun)(**params)
			else:
				getattr(self, fun)()

	def draw_heatmap(self, alpha=1, cmap='PuRd', nbins=300, sample_strength=100):

		fixations = self.fixations.repeat(sample_strength, axis=0)

		x = fixations[:, X] + self.x_offset
		y = fixations[:, Y] + self.y_offset

		k = kde.gaussian_kde([x, y])
		xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
		zi = k(np.vstack([xi.flatten(), yi.flatten()]))

		self.ax.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=alpha, cmap=cmap)

		return self.fig

	def draw_fixations(self, radius=108, set_uniform_fixations=False, alpha=0.5, edgecolors='white', zorder=50
	                   ):
		aois_offset = offset_aoi(self.aoi_location, self.x_offset, self.y_offset)

		for x, y, _ in self.fixations:

			_aoi = locate_aoi(x + self.x_offset, y + self.y_offset,
			                  aoi_dict=aois_offset, radius=radius, default="outside")
			self.ax.scatter(x + self.x_offset, y + self.y_offset,
			                s=self.get_shape_config(_aoi, set_uniform_fixations)[SIZE] * 1000,
				            c=self.get_shape_config(_aoi, set_uniform_fixations)[COLOR],
				            marker=self.get_shape_config(_aoi, set_uniform_fixations)[MARKER],
			                alpha=alpha, edgecolors=edgecolors, zorder=zorder)

	def draw_fixations_aggregate(self, radius=108, set_uniform_fixations=False,
	                             alpha=0.5, cmap_fixations='magma', n_level_colors=20, edgecolors='white', zorder=50):
		# color map of fixations
		colorsFix = matplotlib.cm.get_cmap(cmap_fixations, n_level_colors)(np.arange(n_level_colors))

		for key, aois in self.aoi_location.items():
			for aoi in aois:
				concentration = get_aoi_concentration(aoi_x=aoi[X], aoi_y=aoi[Y], radius=radius,
				                                      xs=self.fixations[:, X], ys=self.fixations[:, Y],
				                                      offset_x=self.x_offset, offset_y=self.y_offset)
				if concentration <= 0:
					continue
				elif n_level_colors > concentration > 0:
					c_ = colorsFix[concentration]
				else:  # concentration >= nlevelColors:
					c_ = colorsFix[n_level_colors - 1]

				self.ax.scatter(aoi[X] + self.x_offset, aoi[Y] + self.y_offset,
				           s=self.get_shape_config(key, set_uniform_fixations)[SIZE] * 1000,
				           color=c_,
				           marker=self.get_shape_config(key, set_uniform_fixations)[MARKER], alpha=alpha, edgecolors=edgecolors,
				           zorder=zorder)

		# plot outside aoi
		aois_offset = offset_aoi(self.aoi_location, self.x_offset, self.y_offset)

		# for i in range(len(self.fixations)):
		for x, y, _ in self.fixations:
			_aoi = locate_aoi(x + self.x_offset, y + self.y_offset, aoi_dict=aois_offset, radius=radius,
			                  default="outside")
			if _aoi not in list(self.aoi_location.keys()):
				self.ax.scatter(x + self.x_offset, y + self.y_offset,
				           s=self.get_shape_config(_aoi, set_uniform_fixations)[SIZE] * 1000,
				           c=self.get_shape_config(_aoi, set_uniform_fixations)[COLOR],
				           marker=self.get_shape_config(_aoi, set_uniform_fixations)[MARKER], alpha=alpha, edgecolors=edgecolors,
				           zorder=zorder)

	def draw_saccades(self,
	                  preserve_saccade_temporal_info=True, cmap_saccades='winter', default_color='green'
	                  , alpha=0.5, linewidth=2):
		"""
		:param preserve_saccade_temporal_info: uses cmap_saccades if set to True else uses a default color
		:param cmap_saccades: default='winter' choose a sequential cmap here otherwise there's no temporal info
		:param default_color: this value is ignored if preserve_saccade_temporal_info is True
		:param alpha:
		:param linewidth:
		"""
		n = len(self.saccades)

		if preserve_saccade_temporal_info:
			colors = matplotlib.cm.get_cmap(cmap_saccades, n)
			colors = colors(range(n))
		else:
			colors = [default_color for _ in range(n)]

		for i, (_, _, _, sx, sy, ex, ey) in enumerate(self.saccades):
			self.ax.plot([sx + self.x_offset, ex + self.x_offset], [sy + self.y_offset, ey + self.y_offset], c=colors[i],
			             linewidth=linewidth, zorder=49, alpha=alpha)

	def draw_aoi(self, radius=108, zorder=48):
		for key, aois in self.aoi_location.items():
			for aoi in aois:
				circle = plt.Circle((aoi[X] + self.x_offset, aoi[Y] + self.y_offset), radius, color=self.aoi_color.get(key), zorder=zorder)
				self.ax.add_patch(circle)

	def draw_gaze(self, color='white', marker='o', size=1.75, alpha=0.5, linewidth=0):
		x = self.gaze[:, X]
		y = self.gaze[:, Y]
		self.ax.scatter(x + self.x_offset, y + self.y_offset, c=color, marker=marker, s=size, alpha=alpha, linewidth=linewidth)

	def get_shape_config(self, aoi_key, is_uniform=False):
		if is_uniform:
			return self.shape_roi_uniform.get(aoi_key)
		else:
			return self.shape_roi.get(aoi_key)

	def clear_fig(self):
		self.fig.clf()
		self.fixations = None
		self.saccades = None
		self.gaze = None

	def save_fig(self, file="figure.png"):
		self.ax.invert_yaxis()
		self.fig.savefig(os.path.join(self.save_dir, file))