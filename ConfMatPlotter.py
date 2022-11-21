import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

MAPPING = {
		"gazeraw":                                              [0, "Raw_Gaze"],
		"gazeraw_wimg":                                         [1, "Raw_Gaze_with_Background"],
		"saccades_temporal_fixations_wimg":                     [2, "Saccades_Temporal_Fixations_with_Background"],

		"heatmap_wimg":                                         [5, "Saliency_Map_with_Background"],
		"heatmap_noimg":                                        [6, "Saliency_Map"],

		"saccades_only":                                        [10, "Saccades"],
		"saccades_temporal":                                    [11, "Saccades_Temporal"],
		"saccades_temporal_nonSequentialCmap":                  [12, "Saccades_Temporal_Nonsequential_Colormap"],

		"saccades_fixations_1shape1color":                      [15, "Saccades_Fixations_Single_Shape_Single_Color"],
		"saccades_temporal_fixationsNonAggregated":             [16, "Saccades_Temporal_Non-aggregated_Fixations"],
		"saccades_temporal_aoi" :                               [17, "Saccades_Temporal_Aoi"],
		"saccades_temporal_fixations" :                         [18, "Saccades_Temporal_Fixations"],
		"saccades_temporal_fixations_aoi" :                     [19, "Saccades_Temporal_Fixations_AOI"],
	}

romanvals = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv"]
IDX, NUM, NAME = 0,1,2

for i, (key, value) in enumerate(MAPPING.items()):
	MAPPING[key] = [value[0], romanvals[i+1]+".", value[-1]]


def getName(ds):
	return MAPPING.get(ds, "NOT FOUND")



if __name__ == '__main__':
	df = pd.read_csv('archive/forConfMat.csv')
	df = df.reset_index()

	fig, ax = plt.subplots(4,5, sharex="all", sharey="all")
	ax = ax.flatten()
	cbar_ax = fig.add_axes([0.92, .12, .02, .75])

	for a in ax:
		a.set_axis_off()

	for i, row in df.iterrows():

		idx, title, _ = getName(row['dataset'])
		# title = "".join([i[0] for i in title.split('_')]) # initials

		fpr = row['FPR']
		fnr = row['FNR']
		tpr = row['TPR']
		tnr = row['TNR']

		confmat = [
			[tpr, fpr],
			[fnr, tnr]
		]


		labels = [1,0]
		dfcm = pd.DataFrame(confmat, columns=labels, index=labels)


		ax[idx].set_axis_on()
		ax[idx].set_title(title)
		sns.heatmap(dfcm, annot=True, vmin=0, vmax=1, ax=ax[idx], fmt='.3g',
		            cmap=sns.cubehelix_palette(as_cmap=True), cbar=i==0, cbar_ax=None if i else cbar_ax)

	fig.text(0.5, 0.04, 'Actual', ha='center', fontsize='x-large')
	fig.text(0.06, 0.5, 'Predicted', va='center', rotation='vertical',fontsize='x-large')
	# plt.suptitle("Confusion matrices representing the average performance of each scanpath dataset on a 5-fold cross-validated pre-trained VGG-16 model")

	plt.show()
		# plt.savefig(f'{title}.png')




