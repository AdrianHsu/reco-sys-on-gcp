#!/usr/bin/env python

import argparse
import os
import glob
import collections 

import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

USER_SHAPE = 943
ITEM_SHAPE = 1682

def load(data_dir):
  # filenames = glob.glob(os.path.join(data_dir, "part-*"))
  filenames = glob.glob(os.path.join("./", "train-dataset_part-*"))
  print(filenames)
  raw_dataset = tf.data.TFRecordDataset(filenames)
  return raw_dataset

def read_tfrecord_fn(example_proto):
  features = {"input_feat": tf.SparseFeature(index_key=["user", "item"],
              value_key="rating",
              dtype=tf.int64,
              size=[USER_SHAPE, ITEM_SHAPE])}

  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features

# @tf.function
def build_rating_sparse_tensor(tfrecord_dataset, sess):

  dataset = tfrecord_dataset.map(read_tfrecord_fn)
  dataset = dataset.repeat()
  iterator = iter(dataset)

  print(iterator.get_next()['input_feat'])
  return iterator.get_next()['input_feat']

def sparse_mean_square_error(sparse_ratings, user_embeddings, movie_embeddings):
  """
  Args:
    sparse_ratings: A SparseTensor rating matrix, of dense_shape [N, M]
    user_embeddings: A dense Tensor U of shape [N, k] where k is the embedding
      dimension, such that U_i is the embedding of user i.
    movie_embeddings: A dense Tensor V of shape [M, k] where k is the embedding
      dimension, such that V_j is the embedding of movie j.
  Returns:
    A scalar Tensor representing the MSE between the true ratings and the
      model's predictions.
  """
  # predictions = tf.gather_nd(
  #     tf.matmul(user_embeddings, movie_embeddings, transpose_b=True),
  #     sparse_ratings.indices)
  predictions = tf.reduce_sum(
      tf.gather(user_embeddings, sparse_ratings.indices[:, 0]) *
      tf.gather(movie_embeddings, sparse_ratings.indices[:, 1]),
      axis=1)
  loss = tf.losses.mean_squared_error(sparse_ratings.values, predictions)
  return loss

class CFModel(object):
  """Simple class that represents a collaborative filtering model"""
  def __init__(self, embedding_vars, loss, metrics=None, sess=None):
    """Initializes a CFModel.
    Args:
      embedding_vars: A dictionary of tf.Variables.
      loss: A float Tensor. The loss to optimize.
      metrics: optional list of dictionaries of Tensors. The metrics in each
        dictionary will be plotted in a separate figure during training.
    """
    self._embedding_vars = embedding_vars
    self._loss = loss
    self._metrics = metrics
    self._embeddings = {k: None for k in embedding_vars}
    self._session = sess

  @property
  def embeddings(self):
    """The embeddings dictionary."""
    return self._embeddings

  def train(self, num_iterations=100, learning_rate=1.0, 
            optimizer=tf.compat.v1.train.GradientDescentOptimizer):
    """Trains the model.
    Args:
      iterations: number of iterations to run.
      learning_rate: optimizer learning rate.
      optimizer: the optimizer to use. Default to GradientDescentOptimizer.
    Returns:
      The metrics dictionary evaluated at the last iteration.
    """
    opt = optimizer(learning_rate)
    train_op = opt.minimize(self._loss)
    local_init_op = tf.group(
        tf.variables_initializer(opt.variables()),
        tf.local_variables_initializer())

    self._session.run(tf.global_variables_initializer())
    self._session.run(tf.tables_initializer())
    tf.train.start_queue_runners()

    local_init_op.run()
    iterations = []
    metrics = self._metrics or ({},)
    metrics_vals = [collections.defaultdict(list) for _ in self._metrics]

    # Train and append results.
    for i in range(num_iterations + 1):
      _, results = self._session.run((train_op, metrics))
      if (i % 10 == 0) or i == num_iterations:
        print("\r iteration %d: " % i + ", ".join(
              ["%s=%f" % (k, v) for r in results for k, v in r.items()]),
              end='')
        iterations.append(i)
        for metric_val, result in zip(metrics_vals, results):
          for k, v in result.items():
            metric_val[k].append(v)

    for k, v in self._embedding_vars.items():
      self._embeddings[k] = v.eval()

    return results

def build_model(train_dataset, eval_dataset, sess, embedding_dim=3, init_stddev=1.):
  """
  Args:
    ratings: a DataFrame of the ratings
    embedding_dim: the dimension of the embedding vectors.
    init_stddev: float, the standard deviation of the random initial embeddings.
  Returns:
    model: a CFModel.
  """

  # SparseTensor representation of the train and test datasets.
  A_train = build_rating_sparse_tensor(train_dataset, sess)
  A_test = build_rating_sparse_tensor(eval_dataset, sess)

  # Initialize the embeddings using a normal distribution.
  U = tf.Variable(tf.random_normal(
      [USER_SHAPE, embedding_dim], stddev=init_stddev))
  V = tf.Variable(tf.random_normal(
      [ITEM_SHAPE, embedding_dim], stddev=init_stddev))
  train_loss = sparse_mean_square_error(A_train, U, V)
  test_loss = sparse_mean_square_error(A_test, U, V)
  metrics = {
      'train_error': train_loss,
      'test_error': test_loss
  }
  embeddings = {
      "user_id": U,
      "movie_id": V
  }
  return CFModel(embeddings, train_loss, [metrics], sess)

def run(work_dir, max_iterations):

  sess = tf.InteractiveSession()
  train_dataset = load(os.path.join(args.work_dir, 'train-dataset'))
  eval_dataset = load(os.path.join(args.work_dir, 'eval-dataset'))

  # Build the CF model and train it.
  model = build_model(train_dataset, eval_dataset, sess, embedding_dim=30, init_stddev=0.5)
  model.train(num_iterations=max_iterations, learning_rate=10.)


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