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

df = pd.read_csv('https://raw.githubusercontent.com/dsindy/kaggle-titanic/refs/heads/master/data/train.csv') # training data

print(df.info())
# 1. Split the data
dftrain, dfeval = train_test_split(df, test_size=0.2, random_state=42)

# 2. Reset the index right away
dftrain = dftrain.reset_index(drop=True)
dfeval = dfeval.reset_index(drop=True)
dftrain = dftrain.reset_index()
dfeval = dfeval.reset_index()

y_train = dftrain.pop('Survived')
y_eval = dfeval.pop('Survived')

CATEGORICAL_COLUMNS = ['Sex', 'SibSp', 'Parch', 'Pclass', 'Embarked']
NUMERIC_COLUMNS = ['Age', 'Fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# print(feature_columns)
# print(dftrain[feature_name].unique())
# print(dftrain['Sex'].unique())
# print(dftrain['Embarked'].unique())
# print(dftrain.head(10))

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# We create a linear estimtor by passing the feature columns we created earlier

linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

clear_output()  # clears consoke output
print(result['accuracy'])  # the result variable is simply a dict of stats about our model
