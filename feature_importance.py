import numpy as np 
import pandas as pd 

pif = np.loadtxt('processed_imputed_features.txt')

y = pd.read_csv('train.csv')['Complaint-Status']
train_length = len(y)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier()
etc.fit(pif[:train_length, :], y)
print(etc.feature_importances_)
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pcad = pca.fit_transform(pif[:, :3])


import matplotlib
import matplotlib.pyplot as plt

colors = ['red','green','blue','purple', 'yellow']

plt.scatter(pcad[:train_length, 0], pcad[:train_length, 1], c = y, cmap = matplotlib.colors.ListedColormap(colors))
plt.show()


