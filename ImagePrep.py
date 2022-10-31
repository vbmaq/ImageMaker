import pandas as pd
import seaborn as sns
from util import *
from EdfReader import read_edf
from Gazeplotter import *
from pathlib import Path
from multiplotter import TPlotter

#<editor-fold desc="Constants for image prep">

NUM_THREADS = 10
### set parent path
path = ''

### set location of Test.csv, Validation.csv and Train.csv within parent path
testCSV = 'data/test.csv'
validationCSV = 'data/validation.csv'
trainCSV = 'data/train.csv'

### set location of ASCfiles used for making the images within parent path
folderASC = 'ASCFiles/fase1/'

### Percentage of scanpath (15%, 30%, 50%, 80% ) used for test pictures.
percentageList = [0.15, 0.3, 0.5, 0.8]

##### Scanpath using timein miliseconds (2s, 5s, 10s,15s)
Timings = [2000, 5000, 10000, 15000]  # miliseconds

### load data
df_test = pd.read_csv(path + testCSV)
df_val = pd.read_csv(path + validationCSV)
df_train = pd.read_csv(path + trainCSV)

### path of ASC files
pathASC = path + folderASC

### make directories and save paths for images
pathSave_test = path + 'data/test/'
# os.mkdir(pathSave_test)

pathSave_percentage = path + 'data/test_percentages/'
# os.mkdir(pathSave_percentage)
# for percentage in percentageList:
#   os.mkdir(pathSave_percentage + 'test_' + str(int(percentage*100))+'/')

pathSave_timing = path + 'data/test_timings/'
# os.mkdir(pathSave_timing)
# for elapseTime in Timings:
#   os.mkdir(pathSave_timing +'test_' + str(int(elapseTime/1000))+'s/')

pathSave_val = path + 'data/validation/'
# os.mkdir(pathSave_val)

pathSave_train = path + 'data/train/'
# os.mkdir(pathSave_train)

### maps Color
cmaps = {}
cmaps['Fixations '] = ['magma', 'jet']
cmaps['Saccades  '] = ['winter', 'magma']

### plotting color maps
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

#</editor-fold>

#<editor-fold desc="Funcs for image prep">
# vvv original code vvv
def plot_color_gradients(cmap_category, cmap_list):
	# Create figure and adjust figure height to number of colormaps
	'''
    Nothing useful just colourbar
    '''

	nrows = len(cmap_list)
	figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
	fig, axs = plt.subplots(nrows=nrows + 1, figsize=(10.5, figh))
	fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
	                    left=0.2, right=0.99)
	axs[0].set_title(cmap_category + ' colormaps', fontsize=14)

	for ax, name in zip(axs, cmap_list):
		ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
		ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
		        transform=ax.transAxes)

	# Turn off *all* ticks & spines, not just the ones with colormaps.
	for ax in axs:
		ax.set_axis_off()


# for cmap_category, cmap_list in cmaps.items():
#     plot_color_gradients(cmap_category, cmap_list)
#
# plt.show()


def save_test_images():
	#### Loop to get ASC file names from df
	ascName = {}
	n = 0
	for iTrial in range(df_test.shape[0]):
		if n > 9:
			n = 0

		if n < 1:
			ascName[df_test.iloc[iTrial, 5][:-6]] = []
			name = df_test.iloc[iTrial, 5][:-6]
		ascName[name].append(df_test.iloc[iTrial, 5][-6:-4])
		n += 1

	for i in ascName:
		for idx in range(len(ascName[i])):
			ascName[i][idx] = ascName[i][idx].replace("_", "", 1)

	####### for loop to make images
	fig = plt.figure(figsize=(1920 / 100, 1920 / 100), dpi=100.0, frameon=False)
	for isub, subject in enumerate(ascName):
		print('PROCCESSING ' + subject)
		datafile = read_edf(pathASC + subject + '.asc', start='START',
		                    stop=None, missing=0.0, debug=False)
		for idx in ascName[subject]:
			draw_scanpath_fixations_AOI(datafile[int(idx) - 1]['events']['Esac'],
			                            datafile[int(idx) - 1]['events']['Efix'],
			                            fig, dispsize=(1920, 1920), originalSize=(1920, 1080),
			                            cmap_saccades='winter', cmap_fixations='magma',
			                            linewidth=8, radius=108, loop=True, alpha=1,
			                            savefilename=pathSave_test + str(subject) + '_' + idx + '.jpg')


def save_val_images():
	#### Loop to get ASC file names from df
	ascName = {}
	n = 0
	for iTrial in range(df_val.shape[0]):
		if n > 9:
			n = 0

		if n < 1:
			ascName[df_val.iloc[iTrial, 5][:-6]] = []
			name = df_val.iloc[iTrial, 5][:-6]
		ascName[name].append(df_val.iloc[iTrial, 5][-6:-4])
		n += 1

	for i in ascName:
		for idx in range(len(ascName[i])):
			ascName[i][idx] = ascName[i][idx].replace("_", "", 1)

	###### for loop to make images
	fig = plt.figure(figsize=(1920 / 100, 1920 / 100), dpi=100.0, frameon=False)
	for isub, subject in enumerate(ascName):
		print('PROCCESSING ' + subject)
		datafile = read_edf(pathASC + subject + '.asc', start='START',
		                    stop=None, missing=0.0, debug=False)
		for idx in ascName[subject]:
			draw_scanpath_fixations_AOI(datafile[int(idx) - 1]['events']['Esac'],
			                            datafile[int(idx) - 1]['events']['Efix'],
			                            fig, dispsize=(1920, 1920), originalSize=(1920, 1080), cmap_saccades='winter',
			                            cmap_fixations='magma',
			                            linewidth=8, radius=108, loop=True, alpha=1,
			                            savefilename=pathSave_val + str(subject) + '_' + idx + '.jpg')


def save_train_images():
	#### Loop to get ASC file names from df
	ascName = {}
	n = 0
	for iTrial in range(df_train.shape[0]):
		if n > 9:
			n = 0

		if n < 1:
			ascName[df_train.iloc[iTrial, 5][:-6]] = []
			name = df_train.iloc[iTrial, 5][:-6]
		ascName[name].append(df_train.iloc[iTrial, 5][-6:-4])
		n += 1

	for i in ascName:
		for idx in range(len(ascName[i])):
			ascName[i][idx] = ascName[i][idx].replace("_", "", 1)

	####### for loop to make images

	fig = plt.figure(figsize=(1920 / 100, 1920 / 100), dpi=100.0, frameon=False)

	for isub, subject in enumerate(ascName):
		print('PROCCESSING ' + subject)
		datafile = read_edf(pathASC + subject + '.asc', start='START',
		                    stop=None, missing=0.0, debug=False)
		for idx in ascName[subject]:
			draw_scanpath_fixations_AOI(datafile[int(idx) - 1]['events']['Esac'],
			                            datafile[int(idx) - 1]['events']['Efix'],
			                            fig, dispsize=(1920, 1920), originalSize=(1920, 1080), cmap_saccades='winter',
			                            cmap_fixations='magma',
			                            linewidth=8, radius=108, loop=True, alpha=1,
			                            savefilename=pathSave_train + str(subject) + '_' + idx + '.jpg')


# vvv revised code by Virmarie vvv
def save_images(df, includeFixations, includeSaccades, includeTemporalInfo, includeAoi, savePath,
                cmapSaccades="winter", aggregateFixations=True):
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

		datafile = read_edf(pathASC + subject + '.asc', start='START',
		                    stop=None, missing=0.0, debug=False)
		for idx in ascName[subject]:

			label = df[df['code'] == f"{subject}_{idx}.jpg"]['Eq'].to_numpy()
			assert np.unique(label).size == 1, f"More than one labels found for subject {subject}_{idx}"
			label = label[0]

			savefilename = os.path.join(savePath, str(label), f"{str(subject)}_{idx}.jpg")
			fig = plt.figure(figsize=(1920 / 100, 1920 / 100), dpi=100.0, frameon=False)
			if aggregateFixations:
				fig = draw_scanpath_fixations_AOI(
						datafile[int(idx) - 1]['events']['Esac'] if includeSaccades else None,
						datafile[int(idx) - 1]['events']['Efix'] if includeFixations else None,
						fig, dispsize=(1920, 1920), originalSize=(1920, 1080),
						cmap_saccades=cmapSaccades, cmap_fixations='magma',
						linewidth=8, radius=108, loop=True, alpha=1,
						savefilename=savefilename,
						preserveSaccadeTemporalInfo=includeTemporalInfo,
						drawAOI=includeAoi
				)
			else:
				fig = draw_scanpath_fixations_AOI(
						datafile[int(idx) - 1]['events']['Esac'] if includeSaccades else None,
						None,
						fig, dispsize=(1920, 1920), originalSize=(1920, 1080),
						cmap_saccades=cmapSaccades, cmap_fixations='magma',
						linewidth=8, radius=108, loop=True, alpha=1,
						savefilename=None,
						preserveSaccadeTemporalInfo=includeTemporalInfo,
						drawAOI=includeAoi)

				fig = draw_scanpath_fixations_color(datafile[int(idx) - 1]['events']['Esac'] if includeSaccades else None,
				                              datafile[int(idx) - 1]['events']['Efix'] if includeFixations else None,
											  fig,
											  dispsize=(1920, 1920), originalSize=(1920, 1080),
											  cmap_saccades=cmapSaccades,
				                              savefilename=savefilename,
				                              loop=False
				                              )



			# threads.append(TPlotter(savefilename, fig))

		while len(threads) != 0:
			current_threads = [threads.pop(0) for _ in range(min(NUM_THREADS, len(threads)))]
			for t in current_threads:
				t.start()
			current_threads[-1].join()



def save_saccades_only():
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


def save_rawGaze_images(df, dispsize=(1920, 1920), savePath=None, imagefile=None, loop=None,
                        marker=".", size=1.75, color="white", plotHeatmap=False, numThreads=10
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

	for isub, subject in enumerate(ascName):
		print('PROCCESSING ' + subject)
		datafile = read_edf(pathASC + subject + '.asc', start='START',
		                    stop=None, missing=0.0, debug=False)

		for idx in ascName[subject]:
			label = df[df['code'] == f"{subject}_{idx}.jpg"]['Eq'].to_numpy()
			assert np.unique(label).size == 1, f"More than one labels found for subject {subject}_{idx}"
			label = label[0]

			savefile = os.path.join(savePath, str(label), f"{str(subject)}_{idx}.jpg")
			if os.path.isfile(savefile):
				print(f"{savefile} already exists. Skipping...")
				continue

			x = datafile[int(idx) - 1]['x']
			y = datafile[int(idx) - 1]['y']
			fig = plt.figure(figsize=(1920 / 100, 1920 / 100), dpi=100, frameon=False)
			fig, ax = draw_display(fig=fig, dispsize=dispsize, imagefile=imagefile)

			if plotHeatmap:
				ax = sns.heatmap()

			else:
				ax.scatter(x, y, c=color, marker=marker, s=size)

			ax.invert_yaxis()

			threads.append(TPlotter(savefile, fig))

		while len(threads) != 0:
			current_threads = [threads.pop(0) for _ in range(min(numThreads, len(threads)))]
			for t in current_threads:
				t.start()
			for t in current_threads:
				t.join()


def save_gaze():
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
	savePath_train = 'Data/saccades_temporal_fixationsNonAggregated_aoi/train/'
	savePath_test = 'Data/saccades_temporal_fixationsNonAggregated_aoi/test/'
	savePath_val = 'Data/saccades_temporal_fixationsNonAggregated_aoi/validation/'
	for p in [savePath_train, savePath_test, savePath_val]:
		for l in ["0", "1"]:
			Path(os.path.join(p, l)).mkdir(parents=True, exist_ok=True)

	save_images(df_train, includeFixations=True, includeSaccades=True, includeTemporalInfo=True, includeAoi=True,
	            savePath=savePath_train, aggregateFixations=False)
	save_images(df_test, includeFixations=True, includeSaccades=True, includeTemporalInfo=True, includeAoi=True,
	            savePath=savePath_test, aggregateFixations=False)
	save_images(df_val, includeFixations=True, includeSaccades=True, includeTemporalInfo=True, includeAoi=True,
	            savePath=savePath_val, aggregateFixations=False)


# TODO
def save_heatmaps():
	savePath_train = 'Data/heatmap/train/'
	savePath_test = 'Data/heatmap/test/'
	savePath_val = 'Data/heatmap/validation/'

	for p in [savePath_train, savePath_test, savePath_val]:
		for l in ["0", "1"]:
			Path(os.path.join(p, l)).mkdir(parents=True, exist_ok=True)

	raise Exception("Not implemented")

	save_images(df_train, includeFixations=True, includeSaccades=True, includeTemporalInfo=True, includeAoi=True,
	            savePath=savePath_train, aggregateFixations=False)
	save_images(df_test, includeFixations=True, includeSaccades=True, includeTemporalInfo=True, includeAoi=True,
	            savePath=savePath_test, aggregateFixations=False)
	save_images(df_val, includeFixations=True, includeSaccades=True, includeTemporalInfo=True, includeAoi=True,
	            savePath=savePath_val, aggregateFixations=False)


# TODO
def save_saccades_fixations_oneshape_onecolor():
	savePath_train = 'Data2/saccades_fixations_1shape1color/train/'
	savePath_test = 'Data2/saccades_fixations_1shape1color/test/'
	savePath_val = 'Data2/saccades_fixations_1shape1color/validation/'

	for p in [savePath_train, savePath_test, savePath_val]:
		for l in ["0", "1"]:
			Path(os.path.join(p, l)).mkdir(parents=True, exist_ok=True)

	save_images(df_train, includeFixations=True, includeSaccades=True, includeTemporalInfo=True, includeAoi=True,
	            savePath=savePath_train, aggregateFixations=False)
	save_images(df_test, includeFixations=True, includeSaccades=True, includeTemporalInfo=True, includeAoi=True,
	            savePath=savePath_test, aggregateFixations=False)
	save_images(df_val, includeFixations=True, includeSaccades=True, includeTemporalInfo=True, includeAoi=True,
	            savePath=savePath_val, aggregateFixations=False)

#</editor-fold>


if __name__ == '__main__':
	seqCmap = ["winter", True]  # [cmapName, isSequential]
	nonseqCmap = ["prism", False]

	save_saccades_fixations_oneshape_onecolor()

	# save_saccades_temporal_fixations_aoi_()
	# save_saccades_only()
	# save_saccades_temporal(*seqCmap)
	# save_saccades_temporal(*nonseqCmap)
	#
	# save_saccades_temporal_aoi()
	# save_saccades_temporal_fixations()
	# save_saccades_temporal_fixationsNonAggregated_aoi()









