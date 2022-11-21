import pandas as pd
import seaborn as sns
from util import *
from EdfReader import read_edf
from Gazeplotter_old import *
from pathlib import Path
from multiplotter import TPlotter
import sys

from Gazeplotter import GazePlotter

def throwForbidden():
	raise Exception("THIS FUNCTION IS FORBIDDEN AS THE DATA HAS ALREADY BEEN COMPILED.")


#<editor-fold desc="Constants for image prep">

NUM_THREADS = 10

### set parent path (you'll need this if you're running from colab with a mounted drive)
PARENT_PATH = ''

### set location of Test.csv, Validation.csv and Train.csv within parent path
PATH_TRAIN_CSV = 'data/train.csv'
PATH_VAL_CSV = 'data/validation.csv'
PATH_TEST_CSV = 'data/test.csv'

### path to stimuli
DIR_STIMULI = "stimoli-phase-1/"

### set location of ASCfiles used for making the images within parent path
DIR_ASC = os.path.join(PARENT_PATH, 'ASCFiles/fase1/')

### AOI info
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


### load data
df_test = pd.read_csv(PARENT_PATH + PATH_TEST_CSV)
df_val = pd.read_csv(PARENT_PATH + PATH_VAL_CSV)
df_train = pd.read_csv(PARENT_PATH + PATH_TRAIN_CSV)

#</editor-fold>

#<editor-fold desc="Funcs for image prep">
#<editor-fold desc="old code">

### maps Color
# cmaps = {}
# cmaps['Fixations '] = ['magma', 'jet']
# cmaps['Saccades  '] = ['winter', 'magma']

### plotting color maps
# gradient = np.linspace(0, 1, 256)
# gradient = np.vstack((gradient, gradient))

### Percentage of scanpath (15%, 30%, 50%, 80% ) used for test pictures.
# percentageList = [0.15, 0.3, 0.5, 0.8]
##### Scanpath using timein miliseconds (2s, 5s, 10s,15s)
# Timings = [2000, 5000, 10000, 15000]  # miliseconds
# os.mkdir(pathSave_test)
# pathSave_percentage = PARENT_PATH + 'data/test_percentages/'
# os.mkdir(pathSave_percentage)
# for percentage in percentageList:
#   os.mkdir(pathSave_percentage + 'test_' + str(int(percentage*100))+'/')

# pathSave_timing = PARENT_PATH + 'data/test_timings/'
# os.mkdir(pathSave_timing)
# for elapseTime in Timings:
#   os.mkdir(pathSave_timing +'test_' + str(int(elapseTime/1000))+'s/')

# os.mkdir(pathSave_val)

# os.mkdir(pathSave_train)

# ### make directories and save paths for images
# DIR_SAVE_TRAIN = PARENT_PATH + 'data/train/'
# DIR_SAVE_VAL = PARENT_PATH + 'data/validation/'
# DIR_SAVE_TEST = PARENT_PATH + 'data/test/'
#
# def plot_color_gradients(cmap_category, cmap_list):
# 	# Create figure and adjust figure height to number of colormaps
# 	'''
#     Nothing useful just colourbar
#     '''
#
# 	nrows = len(cmap_list)
# 	figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
# 	fig, axs = plt.subplots(nrows=nrows + 1, figsize=(10.5, figh))
# 	fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
# 	                    left=0.2, right=0.99)
# 	axs[0].set_title(cmap_category + ' colormaps', fontsize=14)
#
# 	for ax, name in zip(axs, cmap_list):
# 		ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
# 		ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
# 		        transform=ax.transAxes)
#
# 	# Turn off *all* ticks & spines, not just the ones with colormaps.
# 	for ax in axs:
# 		ax.set_axis_off()


# for cmap_category, cmap_list in cmaps.items():
#     plot_color_gradients(cmap_category, cmap_list)
#
# plt.show()

#
# def save_test_images():
# 	#### Loop to get ASC file names from df
# 	ascName = {}
# 	n = 0
# 	for iTrial in range(df_test.shape[0]):
# 		if n > 9:
# 			n = 0
#
# 		if n < 1:
# 			ascName[df_test.iloc[iTrial, 5][:-6]] = []
# 			name = df_test.iloc[iTrial, 5][:-6]
# 		ascName[name].append(df_test.iloc[iTrial, 5][-6:-4])
# 		n += 1
#
# 	for i in ascName:
# 		for idx in range(len(ascName[i])):
# 			ascName[i][idx] = ascName[i][idx].replace("_", "", 1)
#
# 	####### for loop to make images
# 	fig = plt.figure(figsize=(1920 / 100, 1920 / 100), dpi=100.0, frameon=False)
# 	for isub, subject in enumerate(ascName):
# 		print('PROCCESSING ' + subject)
# 		datafile = read_edf(DIR_ASC + subject + '.asc', start='START',
# 		                    stop=None, missing=0.0, debug=False)
# 		for idx in ascName[subject]:
# 			draw_scanpath_fixations_AOI(datafile[int(idx) - 1]['events']['Esac'],
# 			                            datafile[int(idx) - 1]['events']['Efix'],
# 			                            fig, dispsize=(1920, 1920), originalSize=(1920, 1080),
# 			                            cmap_saccades='winter', cmap_fixations='magma',
# 			                            linewidth=8, radius=108, loop=True, alpha=1,
# 			                            savefilename=DIR_SAVE_TEST + str(subject) + '_' + idx + '.jpg')
#
#
# def save_val_images():
# 	#### Loop to get ASC file names from df
# 	ascName = {}
# 	n = 0
# 	for iTrial in range(df_val.shape[0]):
# 		if n > 9:
# 			n = 0
#
# 		if n < 1:
# 			ascName[df_val.iloc[iTrial, 5][:-6]] = []
# 			name = df_val.iloc[iTrial, 5][:-6]
# 		ascName[name].append(df_val.iloc[iTrial, 5][-6:-4])
# 		n += 1
#
# 	for i in ascName:
# 		for idx in range(len(ascName[i])):
# 			ascName[i][idx] = ascName[i][idx].replace("_", "", 1)
#
# 	###### for loop to make images
# 	fig = plt.figure(figsize=(1920 / 100, 1920 / 100), dpi=100.0, frameon=False)
# 	for isub, subject in enumerate(ascName):
# 		print('PROCCESSING ' + subject)
# 		datafile = read_edf(DIR_ASC + subject + '.asc', start='START',
# 		                    stop=None, missing=0.0, debug=False)
# 		for idx in ascName[subject]:
# 			draw_scanpath_fixations_AOI(datafile[int(idx) - 1]['events']['Esac'],
# 			                            datafile[int(idx) - 1]['events']['Efix'],
# 			                            fig, dispsize=(1920, 1920), originalSize=(1920, 1080), cmap_saccades='winter',
# 			                            cmap_fixations='magma',
# 			                            linewidth=8, radius=108, loop=True, alpha=1,
# 			                            savefilename=DIR_SAVE_VAL + str(subject) + '_' + idx + '.jpg')
#
#
# def save_train_images():
# 	#### Loop to get ASC file names from df
# 	ascName = {}
# 	n = 0
# 	for iTrial in range(df_train.shape[0]):
# 		if n > 9:
# 			n = 0
#
# 		if n < 1:
# 			ascName[df_train.iloc[iTrial, 5][:-6]] = []
# 			name = df_train.iloc[iTrial, 5][:-6]
# 		ascName[name].append(df_train.iloc[iTrial, 5][-6:-4])
# 		n += 1
#
# 	for i in ascName:
# 		for idx in range(len(ascName[i])):
# 			ascName[i][idx] = ascName[i][idx].replace("_", "", 1)
#
# 	####### for loop to make images
#
# 	fig = plt.figure(figsize=(1920 / 100, 1920 / 100), dpi=100.0, frameon=False)
#
# 	for isub, subject in enumerate(ascName):
# 		print('PROCCESSING ' + subject)
# 		datafile = read_edf(DIR_ASC + subject + '.asc', start='START',
# 		                    stop=None, missing=0.0, debug=False)
# 		for idx in ascName[subject]:
# 			draw_scanpath_fixations_AOI(datafile[int(idx) - 1]['events']['Esac'],
# 			                            datafile[int(idx) - 1]['events']['Efix'],
# 			                            fig, dispsize=(1920, 1920), originalSize=(1920, 1080), cmap_saccades='winter',
# 			                            cmap_fixations='magma',
# 			                            linewidth=8, radius=108, loop=True, alpha=1,
# 			                            savefilename=DIR_SAVE_TRAIN + str(subject) + '_' + idx + '.jpg')
#</editor-fold>

#<editor-fold desc="    NEW CODE ">
# vvv revised code by Virmarie vvv
def save_images(df, includeFixations, includeSaccades, includeTemporalInfo, includeAoi, savePath, alpha=1,
                cmapSaccades="winter", aggregateFixations=True, setUniformFixations=False, drawHeatmap=False,
				addImageUnderlay=False
                ):
	#### Loop to get ASC file names from df
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

	threads = []
	####### for loop to make images
	fig = plt.figure(figsize=(1920 / 100, 1920 / 100), dpi=100.0, frameon=False)
	for isub, subject in enumerate(ascName):
		print('PROCCESSING ' + subject)

		datafile = read_edf(DIR_ASC + subject + '.asc', start='START',
		                    stop=None, missing=0.0, debug=False)
		for idx in ascName[subject]:

			label = df[df['code'] == f"{subject}_{idx}.jpg"]['Eq'].to_numpy()
			assert np.unique(label).size == 1, f"More than one labels found for subject {subject}_{idx}"
			label = label[0]

			savefilename = os.path.join(savePath, str(label), f"{str(subject)}_{idx}.jpg")
			if addImageUnderlay:
				imagefile = os.path.join(DIR_STIMULI, f"stim_{str(idx).zfill(2)}.jpg")
			else:
				imagefile = None
			if drawHeatmap:
				draw_heatmap(None,
				             fig, dispsize=(1920, 1080), originalSize=(1920, 1080),
				        imagefile=imagefile,
						cmap_saccades=cmapSaccades, cmap_fixations='magma',
						linewidth=8, radius=108, loop=True, alpha=alpha,
						savefilename=savefilename,
						preserveSaccadeTemporalInfo=includeTemporalInfo,
						drawAOI=includeAoi)
			elif aggregateFixations:
				fig = draw_scanpath_fixations_AOI(
						datafile[int(idx) - 1]['events']['Esac'] if includeSaccades else None,
						datafile[int(idx) - 1]['events']['Efix'] if includeFixations else None,
						fig, dispsize=(1920, 1920), originalSize=(1920, 1080),
						imagefile=imagefile,
						cmap_saccades=cmapSaccades, cmap_fixations='magma',
						linewidth=8, radius=108, loop=True, alpha=alpha,
						savefilename=savefilename,
						preserveSaccadeTemporalInfo=includeTemporalInfo,
						drawAOI=includeAoi
				)
			else:
				if includeAoi:
					fig = draw_scanpath_fixations_AOI(
							datafile[int(idx) - 1]['events']['Esac'] if includeSaccades else None,
							None,
							fig, dispsize=(1920, 1920), originalSize=(1920, 1080),
							cmap_saccades=cmapSaccades, cmap_fixations='magma',
							linewidth=8, radius=108, loop=False, alpha=alpha,
							savefilename=None, imagefile=imagefile,
							preserveSaccadeTemporalInfo=includeTemporalInfo,
							drawAOI=includeAoi)

				fig = draw_scanpath_fixations_color(datafile[int(idx) - 1]['events']['Esac'] if includeSaccades else None,
					                                datafile[int(idx) - 1]['events']['Efix'] if includeFixations else None,
												    fig, dispsize=(1920, 1920), originalSize=(1920, 1080),
												    cmap_saccades=cmapSaccades, alpha=alpha,
											        preserveSaccadeTemporalInfo=includeTemporalInfo, defaultSaccadeCol="green",
				                                    setUniformFixations=setUniformFixations,
			                                        savefilename=savefilename, imagefile=imagefile,
					                                loop=True
					                                )

			# threads.append(TPlotter(savefilename, fig))

		while len(threads) != 0:
			current_threads = [threads.pop(0) for _ in range(min(NUM_THREADS, len(threads)))]
			for t in current_threads:
				t.start()
			current_threads[-1].join()
def save_saccades_only():
	throwForbidden()
	savePath_train = 'Data/saccades_only/train/'
	savePath_test = 'Data/saccades_only/test/'
	savePath_val = 'Data/saccades_only/validation/'

	for p in [savePath_train, savePath_test, savePath_val]:
		for l in ["0", "1"]:
			Path(os.path.join(p, l)).mkdir(parents=True, exist_ok=True)

	save_images(df_train, includeFixations=False, includeSaccades=True, includeTemporalInfo=False, includeAoi=False,
	            savePath=savePath_train)
	save_images(df_test, includeFixations=False, includeSaccades=True, includeTemporalInfo=False, includeAoi=False,
	            savePath=savePath_test)
	save_images(df_val, includeFixations=False, includeSaccades=True, includeTemporalInfo=False, includeAoi=False,
	            savePath=savePath_val)


def save_saccades_temporal(cmapSaccades='winter', isSequential=True):
	throwForbidden()
	extraPathIfNonSeq = "" if isSequential else "_nonSequentialCmap"

	savePath_train = f'Data/saccades_temporal{extraPathIfNonSeq}/train/'
	savePath_test = f'Data/saccades_temporal{extraPathIfNonSeq}/test/'
	savePath_val = f'Data/saccades_temporal{extraPathIfNonSeq}/validation/'

	for p in [savePath_train, savePath_test, savePath_val]:
		for l in ["0", "1"]:
			Path(os.path.join(p, l)).mkdir(parents=True, exist_ok=True)

	save_images(df_train, includeFixations=False, includeSaccades=True, includeTemporalInfo=True, includeAoi=False,
	            cmapSaccades=cmapSaccades, savePath=savePath_train)
	save_images(df_test, includeFixations=False, includeSaccades=True, includeTemporalInfo=True, includeAoi=False,
	            cmapSaccades=cmapSaccades, savePath=savePath_test)
	save_images(df_val, includeFixations=False, includeSaccades=True, includeTemporalInfo=True, includeAoi=False,
	            cmapSaccades=cmapSaccades, savePath=savePath_val)


def save_saccades_temporal_fixations():
	throwForbidden()
	savePath_train = 'Data/saccades_temporal_fixations/train/'
	savePath_test = 'Data/saccades_temporal_fixations/test/'
	savePath_val = 'Data/saccades_temporal_fixations/validation/'
	for p in [savePath_train, savePath_test, savePath_val]:
		for l in ["0", "1"]:
			Path(os.path.join(p, l)).mkdir(parents=True, exist_ok=True)

	save_images(df_train, includeFixations=True, includeSaccades=True, includeTemporalInfo=True, includeAoi=False,
	            savePath=savePath_train)
	save_images(df_test, includeFixations=True, includeSaccades=True, includeTemporalInfo=True, includeAoi=False,
	            savePath=savePath_test)
	save_images(df_val, includeFixations=True, includeSaccades=True, includeTemporalInfo=True, includeAoi=False,
	            savePath=savePath_val)


def save_saccades_temporal_aoi():
	throwForbidden()
	savePath_train = 'Data/saccades_temporal_aoi/train/'
	savePath_test = 'Data/saccades_temporal_aoi/test/'
	savePath_val = 'Data/saccades_temporal_aoi/validation/'
	for p in [savePath_train, savePath_test, savePath_val]:
		for l in ["0", "1"]:
			Path(os.path.join(p, l)).mkdir(parents=True, exist_ok=True)

	save_images(df_train, includeFixations=False, includeSaccades=True, includeTemporalInfo=True, includeAoi=True,
	            savePath=savePath_train)
	save_images(df_test, includeFixations=False, includeSaccades=True, includeTemporalInfo=True, includeAoi=True,
	            savePath=savePath_test)
	save_images(df_val, includeFixations=False, includeSaccades=True, includeTemporalInfo=True, includeAoi=True,
	            savePath=savePath_val)


def save_saccades_temporal_fixations_aoi_():
	throwForbidden()
	savePath_train = 'Data/saccades_temporal_fixations_aoi/train/'
	savePath_test = 'Data/saccades_temporal_fixations_aoi/test/'
	savePath_val = 'Data/saccades_temporal_fixations_aoi/validation/'
	for p in [savePath_train, savePath_test, savePath_val]:
		for l in ["0", "1"]:
			Path(os.path.join(p, l)).mkdir(parents=True, exist_ok=True)

	save_images(df_train, includeFixations=True, includeSaccades=True, includeTemporalInfo=True, includeAoi=True,
	            savePath=savePath_train)
	save_images(df_test, includeFixations=True, includeSaccades=True, includeTemporalInfo=True, includeAoi=True,
	            savePath=savePath_test)
	save_images(df_val, includeFixations=True, includeSaccades=True, includeTemporalInfo=True, includeAoi=True,
	            savePath=savePath_val)


def save_rawGaze_images(df, dispsize=(1920, 1920), savePath=None,
                        marker=".", size=1.75, alpha=0.5, color="white", plotHeatmap=False, numThreads=10,
						addImageUnderlay=False
                        ):
	# throwForbidden()
	#### Loop to get ASC file names from df
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

	threads = []
	####### for loop to make images

	for isub, subject in enumerate(ascName):
		print('PROCCESSING ' + subject)
		datafile = read_edf(DIR_ASC + subject + '.asc', start='START',
		                    stop=None, missing=0.0, debug=False)

		for idx in ascName[subject]:
			label = df[df['code'] == f"{subject}_{idx}.jpg"]['Eq'].to_numpy()
			assert np.unique(label).size == 1, f"More than one labels found for subject {subject}_{idx}"
			label = label[0]

			savefile = os.path.join(savePath, str(label), f"{str(subject)}_{idx}.jpg")
			if os.path.isfile(savefile):
				print(f"{savefile} already exists. Skipping...")
				continue

			if addImageUnderlay:
				imagefile = os.path.join(DIR_STIMULI, f"stim_{str(idx).zfill(2)}.jpg")
			else:
				imagefile = None

			x = datafile[int(idx) - 1]['x']
			y = datafile[int(idx) - 1]['y']
			fig = plt.figure(figsize=(1920 / 100, 1920 / 100), dpi=100, frameon=False)
			fig, ax, xoff, yoff = draw_display(fig=fig, dispsize=dispsize, imagefile=imagefile, returnOffset=True)

			if plotHeatmap:
				ax = sns.heatmap()

			else:

				ax.scatter(x + xoff, y+yoff, c=color, marker=marker, s=size, alpha=alpha, linewidth=0)

			ax.invert_yaxis()

			threads.append(TPlotter(savefile, fig))

		while len(threads) != 0:
			current_threads = [threads.pop(0) for _ in range(min(numThreads, len(threads)))]
			for t in current_threads:
				t.start()
			for t in current_threads:
				t.join()


def save_gaze():
	throwForbidden()
	savePath_train = 'Data/gazeraw/train/'
	savePath_test = 'Data/gazeraw/test/'
	savePath_val = 'Data/gazeraw/validation/'

	for p in [savePath_train, savePath_test, savePath_val]:
		for l in ["0", "1"]:
			Path(os.path.join(p, l)).mkdir(parents=True, exist_ok=True)

	save_rawGaze_images(df_train, savePath=savePath_train)
	save_rawGaze_images(df_test, savePath=savePath_test)
	save_rawGaze_images(df_val, savePath=savePath_val)


# TODO
def save_saccades_temporal_fixationsNonAggregated_aoi():
	throwForbidden()
	savePath_train = 'Data2/saccades_temporal_fixationsNonAggregated_aoi/train/'
	savePath_test = 'Data2/saccades_temporal_fixationsNonAggregated_aoi/test/'
	savePath_val = 'Data2/saccades_temporal_fixationsNonAggregated_aoi/validation/'
	for p in [savePath_train, savePath_test, savePath_val]:
		for l in ["0", "1"]:
			Path(os.path.join(p, l)).mkdir(parents=True, exist_ok=True)

	save_images(df_train, includeFixations=True, includeSaccades=True, includeTemporalInfo=True, includeAoi=True,
	            savePath=savePath_train, aggregateFixations=False)
	save_images(df_test, includeFixations=True, includeSaccades=True, includeTemporalInfo=True, includeAoi=True,
	            savePath=savePath_test, aggregateFixations=False)
	save_images(df_val, includeFixations=True, includeSaccades=True, includeTemporalInfo=True, includeAoi=True,
	            savePath=savePath_val, aggregateFixations=False)


def save_heatmaps():
	throwForbidden()
	savePath_train = 'Data2/heatmap_wimg/train/'
	savePath_test = 'Data2/heatmap_wimg/test/'
	savePath_val = 'Data2/heatmap_wimg/validation/'

	for p in [savePath_train, savePath_test, savePath_val]:
		for l in ["0", "1"]:
			Path(os.path.join(p, l)).mkdir(parents=True, exist_ok=True)

	# raise Exception("Not implemented")

	"""
	Plan:
	- get fixations 
	- apply 2d density on top
	- add image underlay 
	"""
	PARAMS = {"includeFixations": True
		, "includeSaccades":      True
		, "includeTemporalInfo":  False
		, "includeAoi":           False
		, "drawHeatmap":          True
		, "addImageUnderlay":     True
		, "aggregateFixations":   False
		, "setUniformFixations":  True
	          }
	save_images(df=df_train
	            , savePath=savePath_train
	            , **PARAMS
	            )
	save_images(df=df_test
	            , savePath=savePath_test
	            , **PARAMS
	            )
	save_images(df=df_val
	            , savePath=savePath_val
	            , **PARAMS
	            )


def save_heatmaps_noimg():
	throwForbidden()
	print("Saving heatmaps without image")
	savePath_train = 'Data2/heatmap_noimg/train/'
	savePath_test = 'Data2/heatmap_noimg/test/'
	savePath_val = 'Data2/heatmap_noimg/validation/'

	for p in [savePath_train, savePath_test, savePath_val]:
		for l in ["0", "1"]:
			Path(os.path.join(p, l)).mkdir(parents=True, exist_ok=True)

	# raise Exception("Not implemented")

	"""
	Plan:
	- get fixations 
	- apply 2d density on top
	- add image underlay 
	"""
	PARAMS = {"includeFixations": True
		, "includeSaccades":      True
		, "includeTemporalInfo":  False
		, "includeAoi":           False
		, "drawHeatmap":          True
	    , "addImageUnderlay":     False
		, "aggregateFixations":   False
		, "setUniformFixations":  True
	          }
	save_images(df=df_train
	            , savePath=savePath_train
	            , **PARAMS
	            )
	save_images(df=df_test
	            , savePath=savePath_test
	            , **PARAMS
	            )
	save_images(df=df_val
	            , savePath=savePath_val
	            , **PARAMS
	            )


def save_saccades_fixations_oneshape_onecolor():
	throwForbidden()
	savePath_train = 'Data2/saccades_fixations_1shape1color/train/'
	savePath_test = 'Data2/saccades_fixations_1shape1color/test/'
	savePath_val = 'Data2/saccades_fixations_1shape1color/validation/'

	for p in [savePath_train, savePath_test, savePath_val]:
		for l in ["0", "1"]:
			Path(os.path.join(p, l)).mkdir(parents=True, exist_ok=True)

	PARAMS = {"includeFixations" : True
			, "includeSaccades" : True
			, "includeTemporalInfo" : False
			, "includeAoi" : False
			, "aggregateFixations" : False
			, "setUniformFixations" : True
	}
	save_images(df=df_train
	            , savePath=savePath_train
	            , **PARAMS
	            )
	save_images(df=df_test
	            , savePath=savePath_test
	            , **PARAMS
	            )
	save_images(df=df_val
	            , savePath=savePath_val
	            , **PARAMS
	            )


def save_gazeraw_with_gameboard():
	throwForbidden()
	savePath_train = 'Data2/gazeraw_gb/train/'
	savePath_test = 'Data2/gazeraw_gb/test/'
	savePath_val = 'Data2/gazeraw_gb/validation/'

	for p in [savePath_train, savePath_test, savePath_val]:
		for l in ["0", "1"]:
			Path(os.path.join(p, l)).mkdir(parents=True, exist_ok=True)

	PARAMS = {
	  "addImageUnderlay": True
	, "size": 35
	, "color": "green"
	}

	save_rawGaze_images(df_train, savePath=savePath_train,
	                    **PARAMS)
	save_rawGaze_images(df_test, savePath=savePath_test,
	                    **PARAMS)
	save_rawGaze_images(df_val, savePath=savePath_val,
	                    **PARAMS)


def save_saccades_temporal_fixations_with_gameboard():
	throwForbidden()
	savePath_train = 'Data2/saccades_temporal_fixations_gb/train/'
	savePath_test = 'Data2/saccades_temporal_fixations_gb/test/'
	savePath_val = 'Data2/saccades_temporal_fixations_gb/validation/'
	for p in [savePath_train, savePath_test, savePath_val]:
		for l in ["0", "1"]:
			Path(os.path.join(p, l)).mkdir(parents=True, exist_ok=True)

	PARAMS = {"includeFixations": True
			, "includeSaccades": True
			, "includeTemporalInfo": True
			, "includeAoi": False
			, "addImageUnderlay": True
			, "alpha": 0.5

	          }

	save_images(df_train
	            , savePath=savePath_train
	            , **PARAMS
	            )
	save_images(df_test
	            , savePath=savePath_test
	            , **PARAMS
	            )
	save_images(df_val
	            , savePath=savePath_val
	            , **PARAMS
	            )

#</editor-fold>


def get_asc_file_names(df:pd.DataFrame):
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


def save_images_test(df, includeFixations, includeSaccades, includeTemporalInfo, includeAoi, savePath, alpha=1,
                cmapSaccades="winter", aggregateFixations=True, setUniformFixations=False, drawHeatmap=False,
				addImageUnderlay=False, dispsize=(1920, 1920), originalSize=(1920, 1080)
                     ):
		ascName = get_asc_file_names(df)
		fig = plt.figure(figsize=(1920 / 100, 1920 / 100), dpi=100.0, frameon=False)
		offset_x, offset_y = calculate_offset(dispsize, originalSize)

		gp = GazePlotter(
		                 shape_roi=SHAPE_ROI,
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

				savefilename = os.path.join(str(get_label(subject, idx, df)), f"{str(subject)}_{idx}.jpg") if savePath else None
				imagefile = os.path.join(DIR_STIMULI, f"stim_{str(idx).zfill(2)}.jpg") if addImageUnderlay else None

				fig, ax = draw_display(fig=fig, dispsize=dispsize, imagefile=imagefile)

				fixations = np.array(datafile[int(idx) - 1]['events']['Efix'])[:, [3,4,2]]

				gazes = np.array( [datafile[int(idx) - 1]['x'], datafile[int(idx) - 1]['y']]).T

				gp.fixations = fixations
				gp.saccades = datafile[int(idx) - 1]['events']['Esac']
				gp.gaze = gazes
				gp.fig = fig
				gp.ax = ax

				gp.draw_AOI()
				gp.draw_heatmap()
				gp.draw_gaze()
				gp.draw_saccades()
				gp.draw_fixations()
				gp.draw_fixations_aggregate()

				gp.save_fig(savefilename)
				gp.clear_fig()


				# draw_fixations(
				# 		fixations=datafile[int(idx) - 1]['events']['Efix'],
				# 		ax=ax,
				# 		x_offset=offset_x, y_offset=offset_y,
				# 		set_uniform_fixations=setUniformFixations, alpha=alpha, radius=108,
				# )
				#
				# draw_fixations_aggregate(fixations=datafile[int(idx) - 1]['events']['Efix'],
				#                          ax=ax,
				#                                x_offset=offset_x, y_offset=offset_y,
				#                                radius=108, alpha=alpha,
				#                                n_level_colors=20)
				#
				# draw_saccades(datafile[int(idx) - 1]['events']['Esac'],
				#               ax=ax,
				#                     x_offset=offset_x, y_offset=offset_y,
				#                     alpha=alpha, linewidth=2
				#                     )
				#
				# draw_AOI(ax, x_offset=offset_x, y_offset=offset_y, radius=108)
				#
				# x = datafile[int(idx) - 1]['x']
				# y = datafile[int(idx) - 1]['y']
				#
				# draw_gaze(ax, x, y, offset_x, offset_y,
				#           marker="o", color="purple", alpha=1)
				#
				# draw_heatmap(fixations=datafile[int(idx) - 1]['events']['Efix'], ax=ax,
				#              x_offset=offset_x, y_offset=offset_y,
				#              alpha=1, cmap='PuRd', nbins=300, sample_strength=100)



				# invert the y axis, as (0,0) is top left on a display
				ax.invert_yaxis()
				# if savefilename:
				# 	fig.savefig(savefilename)


				# if drawHeatmap:
				# 	draw_heatmap(None,
				# 	             datafile[int(idx) - 1]['events']['Efix'] if includeFixations else None,
				# 	             fig, dispsize=(1920, 1080), originalSize=(1920, 1080),
				# 	             imagefile=imagefile,
				# 	             cmap_saccades=cmapSaccades, cmap_fixations='magma',
				# 	             linewidth=8, radius=108, loop=True, alpha=alpha,
				# 	             savefilename=savefilename,
				# 	             preserveSaccadeTemporalInfo=includeTemporalInfo,
				# 	             drawAOI=includeAoi)
				# elif aggregateFixations:
				# 	fig = draw_scanpath_fixations_AOI(
				# 			datafile[int(idx) - 1]['events']['Esac'] if includeSaccades else None,
				# 			datafile[int(idx) - 1]['events']['Efix'] if includeFixations else None,
				# 			fig, dispsize=(1920, 1920), originalSize=(1920, 1080),
				# 			imagefile=imagefile,
				# 			cmap_saccades=cmapSaccades, cmap_fixations='magma',
				# 			linewidth=8, radius=108, loop=True, alpha=alpha,
				# 			savefilename=savefilename,
				# 			preserveSaccadeTemporalInfo=includeTemporalInfo,
				# 			drawAOI=includeAoi
				# 	)
				# else:
				# 	if includeAoi:
				# 		fig = draw_scanpath_fixations_AOI(
				# 				datafile[int(idx) - 1]['events']['Esac'] if includeSaccades else None,
				# 				None,
				# 				fig, dispsize=(1920, 1920), originalSize=(1920, 1080),
				# 				cmap_saccades=cmapSaccades, cmap_fixations='magma',
				# 				linewidth=8, radius=108, loop=False, alpha=alpha,
				# 				savefilename=None, imagefile=imagefile,
				# 				preserveSaccadeTemporalInfo=includeTemporalInfo,
				# 				drawAOI=includeAoi)
				#
				# 	fig = draw_scanpath_fixations_color(
				# 		datafile[int(idx) - 1]['events']['Esac'] if includeSaccades else None,
				# 		datafile[int(idx) - 1]['events']['Efix'] if includeFixations else None,
				# 		fig, dispsize=(1920, 1920), originalSize=(1920, 1080),
				# 		cmap_saccades=cmapSaccades, alpha=alpha,
				# 		preserveSaccadeTemporalInfo=includeTemporalInfo, defaultSaccadeCol="green",
				# 		setUniformFixations=setUniformFixations,
				# 		savefilename=savefilename, imagefile=imagefile,
				# 		loop=True
				# 		)


def test():
	deletemePath = "deleteme"
	for p in [deletemePath]:
		for l in ["0", "1"]:
			Path(os.path.join(p, l)).mkdir(parents=True, exist_ok=True)
	PARAMS = {
		"includeFixations":               True
		, "includeSaccades":              False
		, "includeTemporalInfo":          False
		, "includeAoi":                   False
		, "savePath":                     deletemePath
		, "alpha":                        1
		, "cmapSaccades":                 "winter"
		, "aggregateFixations":           True
		, "setUniformFixations":          False
		, "drawHeatmap":                  False
		, "addImageUnderlay":             True}

	save_images_test(df_train
	            , **PARAMS
	            )


if __name__ == '__main__':
	seqCmap = ["winter", True]  # [cmapName, isSequential]
	nonseqCmap = ["prism", False]

	test()
	# opt = sys.argv[1]
	# # opt="gbstf"
	# # opt = "gbraw"
	#
	# if opt == "img":
	# 	save_heatmaps()
	# if opt == "noimg":
	# 	save_heatmaps_noimg()
	# if opt == "gbstf":
	# 	save_saccades_temporal_fixations_with_gameboard()
	# if opt == "gbraw":
	# 	save_gazeraw_with_gameboard()

	# save_saccades_fixations_oneshape_onecolor()
	# save_saccades_temporal_fixations_aoi_()
	# save_saccades_only()
	# save_saccades_temporal(*seqCmap)
	# save_saccades_temporal(*nonseqCmap)
	#
	# save_saccades_temporal_aoi()
	# save_saccades_temporal_fixations()
	# save_saccades_temporal_fixationsNonAggregated_aoi()









