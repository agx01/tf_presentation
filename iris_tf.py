# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

print("TensorFlow version: {}".format(tf.__version__))
print("Eager Execution: {}".format(tf.executing_eagerly()))

train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname = os.path.basename(train_dataset_url),
                                           origin = train_dataset_url)

print("Local Copy of the dataset file: {}".format(train_dataset_fp))

iris_data = pd.read_csv(train_dataset_fp, header=None)
iris_data = iris_data.iloc[1:, :]
print(iris_data.head())

columns_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = columns_names[:-1]
label_names = columns_names[-1]

print("Features: {}".format(feature_names))
print("Labels: {}".format(label_names))

class_names = ['Iris Setosa', 'Iris versicolor', 'Iris virginica']

batch_size = 32

train_dataset = tf.data.experimental.make_csv_dataset(train_dataset_fp, 
                                                      batch_size,
                                                      column_names = columns_names,
                                                      label_name=label_names,
                                                      num_epochs=10)

features, labels = next(iter(train_dataset))
print("Features:")
print(features)
print("Labels:")
print(labels)

plt.scatter(features['petal_length'],
            features['sepal_length'],
            c = labels,
            cmap = 'viridis')

plt.xlabel("Petal Length")
plt.ylabel("Sepal Length")
plt.show()

def pack_features_vector(features, labels):
    """Pack the features into a single array"""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))

print(features[:5])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4, )),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)])

predictions = model(features)
predictions[:5]

tf.nn.softmax(predictions[:5])

print("Predictions: {}".format(tf.argmax(predictions, axis=1)))
print("     Labels: {}".format(labels))



