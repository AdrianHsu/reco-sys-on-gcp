#!/usr/bin/env python

import argparse
import os
import glob
import collections 

import tensorflow as tf
import numpy as np

from tensorflow.keras import Model

tf.enable_eager_execution()

USER_SHAPE = 943
ITEM_SHAPE = 1682

def load(data_dir):
  filenames = tf.io.gfile.glob(os.path.join(data_dir, "part-*"))
  # filenames = glob.glob(os.path.join("./", "train-dataset_part-*"))
  print(filenames)
  raw_dataset = tf.data.TFRecordDataset(filenames)
  return raw_dataset

def read_tfrecord_fn(example_proto):
  features = {"user": tf.FixedLenFeature((), tf.int64, default_value=0),
              "item": tf.FixedLenFeature((), tf.int64, default_value=0),
              "rating": tf.FixedLenFeature((), tf.int64, default_value=0)}

  parsed_features = tf.io.parse_single_example(example_proto, features)
  return parsed_features['user'], parsed_features['item'], parsed_features['rating']

def build_indices(dataset):
  indices = []
  values = []

  cnt = 0
  for arr in dataset:
    ts = tf.stack(values=[arr[0], arr[1]], axis=0)
    indices.append(ts)
    values.append(arr[2])

    if cnt % 10000 == 0:
      print("load data: ", cnt)
    cnt += 1

  return indices, values

def build_rating_sparse_tensor(tfrecord_dataset):

  dataset = tfrecord_dataset.map(read_tfrecord_fn)
  indices, values = build_indices(dataset)

  print(indices)
  print(values)

  st = tf.sparse.SparseTensor(
      indices=tf.stack(values = indices, axis = 0),
      values=tf.stack(values = values, axis = 0),
      dense_shape=[USER_SHAPE, ITEM_SHAPE])

  print(st)
  return st

class MyModel(Model):
  def __init__(self, embed_dim):
    # Initialize the embeddings using a normal distribution.
    super(MyModel, self).__init__()
    self.embedding_dim = embed_dim
    self.U = tf.Variable(tf.random.normal(
        [USER_SHAPE, embed_dim], stddev=1.), trainable=True)
    self.V = tf.Variable(tf.random.normal(
        [ITEM_SHAPE, embed_dim], stddev=1.), trainable=True)    

def build_model(train_dataset, eval_dataset, max_iterations, learning_rate=1, embed_dim = 3):

  # SparseTensor representation of the train and test datasets.
  A_train = build_rating_sparse_tensor(train_dataset)
  A_test = build_rating_sparse_tensor(eval_dataset)

  model = MyModel(embed_dim)
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  test_loss = tf.keras.metrics.Mean(name='test_loss')

  def train_step(A_train, learning_rate):
    with tf.GradientTape(persistent=True) as tape:
      predictions = tf.reduce_sum(
        tf.gather(model.U, A_train.indices[:, 0]) *
        tf.gather(model.V, A_train.indices[:, 1]),
        axis=1)
      loss_obj = tf.keras.losses.MSE(A_train.values, predictions)

    gradients = tape.gradient(loss_obj, model.trainable_variables)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    # print(model.trainable_variables)
    train_loss(loss_obj)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  def test_step(A_test):
    with tf.GradientTape(persistent=True) as tape:
      predictions = tf.reduce_sum(
        tf.gather(model.U, A_test.indices[:, 0]) *
        tf.gather(model.V, A_test.indices[:, 1]),
        axis=1)
      loss_obj = tf.keras.losses.MSE(A_test.values, predictions)
    test_loss(loss_obj)

  for i in range(max_iterations):
    train_loss.reset_states()
    test_loss.reset_states()
    train_step(A_train, learning_rate)
    test_step(A_test)
    print(
      f'Epoch {i + 1}, '
      f'Train Loss: {train_loss.result()}, '
      f'Test Loss: {test_loss.result()}, '
    )


def run(work_dir, max_iterations):
  print(args.work_dir)
  train_dataset = load(os.path.join(args.work_dir, 'train-dataset'))
  eval_dataset = load(os.path.join(args.work_dir, 'eval-dataset'))

  model = build_model(train_dataset, eval_dataset, max_iterations)

if __name__ == '__main__':
  """Main function called by AI Platform."""

  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      '--work-dir',
      required=False,
      default='gs://ahsu-movielens',
      help='Directory for staging and working files. '
           'This can be a Google Cloud Storage path.')

  parser.add_argument(
      '--max_iterations',
      type=int,
      default=1000,
      help='Number of iterations to train the model')

  args = parser.parse_args()

  run(
    args.work_dir,
    args.max_iterations)