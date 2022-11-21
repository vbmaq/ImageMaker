import numpy as np


SHAPE_ROI = {"own":     ['^', 'blue', 1.75],
             "other":   ['D', 'lime', 1.75],
             "outside": ['.', 'fuchsia', 1],
             }

SHAPE_ROI_UNIF = ['^', 'green', 1.75]

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