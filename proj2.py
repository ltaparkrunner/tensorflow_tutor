# %tensorflow_version 2.x  # this line is not required unless you are in a notebook
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

# 2. Глушим стандартный вывод TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 3. Дополнительно перенаправляем stderr в пустоту на момент импорта, 
# если что-то всё ещё пытается пробиться
sys.stderr = open(os.devnull, 'w')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
# from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf
from sklearn.model_selection import train_test_split

sys.stderr = sys.__stderr__
# Load dataset.

df = pd.read_csv('https://raw.githubusercontent.com/dsindy/kaggle-titanic/refs/heads/master/data/train.csv') # training data
# dftrain = df.sample(frac=0.8, random_state=42)
# dfeval = df.drop(dftrain.index)
print(df.info())
# 1. Split the data
dftrain, dfeval = train_test_split(df, test_size=0.2, random_state=42)

# 2. Reset the index right away
dftrain = dftrain.reset_index(drop=True)
dfeval = dfeval.reset_index(drop=True)
dftrain = dftrain.reset_index()
dfeval = dfeval.reset_index()

y_train = dftrain.pop('Survived')
# print(y_train)
y_eval = dfeval.pop('Survived')
# print(dftrain.loc[0], y_train.loc[0])
# print(dftrain.describe())
# print(dftrain.head())
# print(dftrain.shape)
# print(y_train.head())
dftrain.Age.hist(bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

# 3. Display the window
plt.show()

dftrain.Sex.value_counts().plot(kind='barh')
plt.show()

dftrain['Pclass'].value_counts().plot(kind='barh')
plt.show()

pd.concat([dftrain, y_train], axis=1).groupby('Sex').Survived.mean().plot(kind='barh').set_xlabel('% survive')
plt.show()

print(f'dfeval.shape = {dfeval.shape}')