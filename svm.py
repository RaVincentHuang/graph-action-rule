from math import e
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
import sys
sys.path.append('/home/vincent/graphrule/src')
from task.game24 import calc_exprs_4, calc_exprs_3, calc_exprs_2, calc_exprs

import pandas as pd
import numpy as np
from itertools import permutations
from tqdm import tqdm

clf = svm.SVC(kernel='rbf')
X = []
task_path = "/home/vincent/graphrule/data/tasks/24.csv"
for chunk in pd.read_csv(task_path, usecols=['Rank', 'Puzzles'], chunksize=1):
    for _, row in chunk.iterrows():
        task = row['Puzzles']
        data = list(map(lambda x: int(x), task.split()))
        accs = calc_exprs(*data)
        if not accs:
            continue
        unique_exprs = set()
        for expr in accs:
            unique_exprs.add(expr)
        for expr in unique_exprs:
            


for i in range(128):
    for j in range(i, 128):
        for k in range(j, 128):
            for l in range(k, 128):
                X.append([l, k, j, i])

y = []


clf.fit(X, y)

import matplotlib.pyplot as plt

# Create a mesh to plot in
h = .02  # step size in the mesh
x_min, x_max = np.array(X)[:, 0].min() - 1, np.array(X)[:, 0].max() + 1
y_min, y_max = np.array(X)[:, 1].min() - 1, np.array(X)[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Plot the decision boundary
plt.subplot(1, 1, 1)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.8)

# Plot also the training points
plt.scatter(np.array(X)[:, 0], np.array(X)[:, 1], c=y, cmap='coolwarm', s=20, edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('SVM Decision Boundary')
plt.show()
                