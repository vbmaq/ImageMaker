import sys
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
# import matplotlib.pyplot as plt
# from skimage.transform import resize
# from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score, confusion_matrix, RocCurveDisplay
import pickle

# Define paths
TRAIN = 'train'
VAL = 'validation'
TEST = 'test'

pathtoData = "dataframes"
pathtoSave = "dataframes"
pathtoModel = "dataframes"

def load_dataset(pathtoTrainData, pathtoValData, pathtoTestData):
	def loadxy(pathToData):
		df = pd.read_pickle(pathToData)
		x=df['X'].to_numpy().tolist()
		y=df['y'].to_numpy()
		return x, y

	x_train, y_train = loadxy(pathtoTrainData)
	x_val, y_val = loadxy(pathtoValData)
	x_test, y_test = loadxy(pathtoTestData)

	return x_train, y_train, x_val, y_val, x_test, y_test


def print_gridsearch_estimator(dataset):
	pathtoTrainData = os.path.join(pathtoData, "noaugs", f"{dataset}_{TRAIN}.pkl")
	pathtoValData = os.path.join(pathtoData, "noaugs", f"{dataset}_{VAL}.pkl")
	pathtoTestData = os.path.join(pathtoData, "noaugs", f"{dataset}_{TEST}.pkl")

	x_train, y_train, _, _, _, _ = load_dataset(pathtoTrainData, pathtoValData, pathtoTestData)
	print(f"x_train {x_train[0].shape}")

	param_grid = {'C':      [0.1, 10, 100],
                  'gamma':  [10, 0.1, 0.0001],
                  'kernel': ['rbf']}
	svc = svm.SVC(probability=False)
	print(f"Initializing gridsearch for dataset {dataset}")
	grid = GridSearchCV(svc, param_grid)

    # fitting the model for grid search
	print("Fitting model for grid search")
	grid.fit(x_train, y_train)

    # print best parameter after tuning
	print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
	print(grid.best_estimator_)


if __name__ == '__main__':
    print_gridsearch_estimator(sys.argv[1])
