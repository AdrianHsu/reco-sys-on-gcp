#!/usr/bin/env python

import argparse
import os
import tensorflow as tf
from urllib.request import urlretrieve
import zipfile
import zlib

# 100k
# MOVIE_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
# UNZIP_FOLDER = "ml-100k"
# FILENAME = "u.data"
# DATANAME = "ratings-100k.csv"

# 25m
MOVIE_URL = "http://files.grouplens.org/datasets/movielens/ml-25m.zip"
UNZIP_FOLDER = "ml-25m"
FILENAME = "ratings.csv"
DATANAME = "ratings-25m.csv"


def run(data_sources, zip_filepath, data_name, data_dir):
  """Extracts the specified number of data files."""
  if not tf.gfile.Exists(data_dir):
    tf.gfile.MakeDirs(data_dir)

  data_file = os.path.join(data_dir, data_name)

  if not tf.gfile.Exists(data_file):
    with tf.gfile.Open(data_file, 'w') as f:
      urlretrieve(data_sources, zip_filepath)
      zip_ref = zipfile.ZipFile(zip_filepath, "r")
      zip_ref.extractall()

      f.write(zip_ref.read(os.path.join(UNZIP_FOLDER, FILENAME))) # write to GCS
      print('Extracted {}'.format(data_file))
  else:
    print('Found {}'.format(data_file))

if __name__ == '__main__':
  """Main function"""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      '--work-dir',
      required=False,
      default='gs://ahsu-movielens',
      help='Directory for staging and working files. '
           'This can be a Google Cloud Storage path.')

  parser.add_argument(
      '--data-sources',
      required=False,
      default=MOVIE_URL)

  parser.add_argument(
      '--zip-filepath',
      required=False,
      default='./movielens.zip')

  parser.add_argument(
      '--data-name',
      required=False,
      default=DATANAME)

  args = parser.parse_args()

  data_dir = os.path.join(args.work_dir, 'data')
  run(args.data_sources, args.zip_filepath, args.data_name, data_dir)
