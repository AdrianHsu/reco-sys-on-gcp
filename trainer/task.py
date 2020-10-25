#!/usr/bin/env python

import argparse
import os
import glob

import tensorflow as tf
import numpy as np

def load(data_dir):
  filenames = glob.glob(os.path.join(data_dir, "part-*"))
  raw_dataset = tf.data.TFRecordDataset(filenames)

  return raw_dataset

def run(work_dir, train_max_epoches):

  train_dataset = load(os.path.join(args.work_dir, 'train-dataset'))
  eval_dataset = load(os.path.join(args.work_dir, 'eval-dataset'))

  print(train_dataset)

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
      '--train-max-epoches',
      type=int,
      default=1000,
      help='Number of epoches to train the model')

  args = parser.parse_args()

  run(
    args.work_dir,
    args.train_max_epoches)