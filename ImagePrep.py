import pandas as pd
import yaml

from Tools.EdfReader import read_edf
import os
from configs.aoi_configs import *
from pathlib import Path
import configparser
import matplotlib.pyplot as plt

from Gazeplotter import GazePlotter, calculate_offset


# <editor-fold desc="Load configs">
cfg = configparser.ConfigParser()
cfg.read('configs/imageprep.ini')

PARENT_PATH = cfg['PATH']['parent_path']

PATH_TRAIN_CSV = os.path.join(PARENT_PATH, cfg['PATH']['train_csv'])
PATH_VAL_CSV = os.path.join(PARENT_PATH, cfg['PATH']['val_csv'])
PATH_TEST_CSV = os.path.join(PARENT_PATH, cfg['PATH']['test_csv'])

DIR_STIMULI = os.path.join(PARENT_PATH, cfg['PATH']['stimuli_dir'])
DIR_ASC = os.path.join(PARENT_PATH, cfg['PATH']['asc_dir'])


### load data
df_test = pd.read_csv(PATH_TEST_CSV)
df_val = pd.read_csv(PATH_VAL_CSV)
df_train = pd.read_csv(PATH_TRAIN_CSV)

# </editor-fold>

def throwForbidden():
	raise Exception("THIS FUNCTION IS FORBIDDEN AS THE DATA HAS ALREADY BEEN COMPILED.")

# <editor-fold desc="Dataset-specific code">
"""
This section contains code specific to the dataset used for our paper but can easily be changed to fit your needs. 
Worth checking out are the following:

- get_label : returns the label given a dataframe, subject and index 
- create_label_dirs : builds a directory for your image files as separated by its labels (this is compatible with
	pytorch's ImageFolder https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html)
- plot_scanpaths : shows an example of how to use Gazeplotter    

"""

def get_asc_file_names(df: pd.DataFrame):
	"""
	#### Loop to get ASC file names from df
	:param df: Pandas DataFrame
	:return dict of subject with filenames:
	"""
	ascName = {}
	n = 0
	for iTrial in range(df.shape[0]):
		if n > 9:
			n = 0

		if n < 1:
			ascName[df.iloc[iTrial, 5][:-6]] = []
			name = df.iloc[iTrial, 5][:-6]
		ascName[name].append(df.iloc[iTrial, 5][-6:-4])
		n += 1

	for i in ascName:
		for idx in range(len(ascName[i])):
			ascName[i][idx] = ascName[i][idx].replace("_", "", 1)

	return ascName


def get_label(subject, idx, df):
	label = df[df['code'] == f"{subject}_{idx}.jpg"]['Eq'].to_numpy()
	assert np.unique(label).size == 1, f"More than one labels found for subject {subject}_{idx}"

	return label[0]


def create_label_dirs(labels, from_dir):
	for p in from_dir:
		for l in labels:
			Path(os.path.join(p, l)).mkdir(parents=True, exist_ok=True)


def retrieve_image_file(idx):
	"""Finds image file from stimuli directory given an index"""
	# image file index are in 0X format
	image_file = f"stim_{str(idx).zfill(2)}.jpg"

	return os.path.join(DIR_STIMULI, image_file)


def create_savefilename(dir, subject, idx):
	return os.path.join(dir,
	             f"{str(subject)}_{idx}.jpg")


def plot_scanpaths(df, savePath, pipeline,
                   add_image_underlay=False,
                   backdrop_value=0,
                   display_size=(1920, 1920), original_size=(1920, 1080), dpi=100,
                   ):
	ascName = get_asc_file_names(df)

	fig_dim = max(display_size)
	fig = plt.figure(figsize=(fig_dim / 100, fig_dim / 100), dpi=dpi, frameon=False)
	offset_x, offset_y = calculate_offset(display_size, original_size)

	gp = GazePlotter(shape_roi=SHAPE_ROI,
	                 shape_roi_unif=SHAPE_ROI_UNIF,
	                 aoi_col=AOI_COL,
	                 aoi_loc=AOI_LOC,
	                 x_offset=offset_x,
	                 y_offset=offset_y,
	                 save_dir=savePath)

	####### for loop to make images
	for isub, subject in enumerate(ascName):
		print(f'\nPROCCESSING {subject}: [{len(ascName[subject])} images]', end=" ")
		datafile = read_edf(DIR_ASC + subject + '.asc', start='START',
		                    stop=None, missing=0.0, debug=False)
		for idx in ascName[subject]:
			print('.', end="")

			# ============================ Prepare file stuff =============================
			savefilename = create_savefilename(dir=str(get_label(subject, idx, df)), subject=subject, idx=idx) if savePath else None
			imagefile = retrieve_image_file(idx) if add_image_underlay else None

			# ================== Retrieve gaze information from datafile ==================
			saccades = datafile[int(idx) - 1]['events']['Esac']
			fixations = np.array(datafile[int(idx) - 1]['events']['Efix'])[:, [3, 4, 2]]
			gazes = np.array([datafile[int(idx) - 1]['x'], datafile[int(idx) - 1]['y']]).T

			# ========================= Gaze plotting begins here =========================
			fig, ax = GazePlotter.draw_display(fig=fig, display_size=display_size, image_file=imagefile,
			                                   background_value=backdrop_value)
			gp.fixations = fixations
			gp.saccades = saccades
			gp.gaze = gazes
			gp.fig = fig
			gp.ax = ax
			gp.filename = savefilename

			gp.run_pipeline(pipeline)


def run(config):
	with open(config, "r") as stream:
		config = yaml.safe_load(stream)

		save_dir = config["save_dir"]
		pipeline = config["pipeline"]
		labels = config["labels"]

		save_train = os.path.join(save_dir, "train")
		save_val = os.path.join(save_dir, "validate")
		save_test = os.path.join(save_dir, "test")

		create_label_dirs(labels, [save_train, save_val, save_test])

		# uniform parameters for plot_scanpaths
		kwargs = {"add_image_underlay": config.get("has_background", False),
		          "backdrop_value":     config.get("backdrop_value", 0)
		          }
		plot_scanpaths(df_train, savePath=save_train, pipeline=pipeline, **kwargs)
		plot_scanpaths(df_val, savePath=save_val, pipeline=pipeline, **kwargs)
		plot_scanpaths(df_test, savePath=save_test, pipeline=pipeline, **kwargs)

# </editor-fold>










