import os
from pathlib import Path

from archive.ImagePrep import df_train
from multiplotter import save_rawGaze_images

if __name__ == '__main__':
	savePath_train = 'Data/gazeraw/train/'
	savePath_test = 'Data/gazeraw/test/'
	savePath_val = 'Data/gazeraw/validation/'

	for p in [savePath_train, savePath_test, savePath_val]:
		for l in ["0", "1"]:
			Path(os.path.join(p, l)).mkdir(parents=True, exist_ok=True)

	save_rawGaze_images(df_train, savePath=savePath_train, loop=True)
	# save_rawGaze_images(df_test, savePath=savePath_test)
	# save_rawGaze_images(df_val, savePath=savePath_val)