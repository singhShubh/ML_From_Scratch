'''
1. Number of Attributes: 10 plus the class attribute

2. Attribute Information: (class attribute has been moved to last column)

   #  Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)

3. Missing attribute values: 16

   There are 16 instances in Groups 1 to 6 that contain a single missing
   (i.e., unavailable) attribute value, now denoted by "?".

'''

import pandas as pd
import numpy as np
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path+ "/../lib")
from LogisticRegression import LogisticRegression


df = pd.read_csv('./../data/breast-cancer-wisconsin.data')
df.drop(labels=['id'], axis = 1, inplace=True)
df.replace('?',np.NaN,inplace=True)
df.dropna(axis=0,inplace=True)
df = df.sample(frac=1).reset_index(drop=True)
x = df.as_matrix(['clump_thickness',
                  'unnif_cell_size',
                  'unif_cell_shape',
                  'marg_adhesion',
                  'single_epith_cell_size',
                  'bare_nuclei',
                  'bland_chrom',
                  'norm_nucleoli',
                  'mitoses'])

y = (df.as_matrix(['class'])==2) + 0
x = x.astype('float')
clf = LogisticRegression()
clf.optm(x,y)
predicted_y = clf.predict(x)
print(clf.accuracy(predicted_y, y))
