#!/usr/bin/env python

import argparse
import os
import pickle
import tensorflow as tf
from urllib.request import urlretrieve
import zipfile
import zlib

MOVIE_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
# MOVIE_URL = "http://files.grouplens.org/datasets/movielens/ml-25m.zip"
UNZIP_FOLDER = "ml-100k"
# UNZIP_FOLDER = "ml-25m"
FILENAME = "u.data"
# FILENAME = "ratings.csv"


def dump(obj, filename):
  """ Wrapper to dump an object to a file."""
  with tf.io.gfile.GFile(filename, 'wb') as f:
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def run(data_sources, zip_filepath, data_name, data_dir):
  """Extracts the specified number of data files."""
  if not tf.gfile.Exists(data_dir):
    tf.gfile.MakeDirs(data_dir)

  data_file = os.path.join(data_dir, data_name)

  if not tf.gfile.Exists(data_file):
    urlretrieve(data_sources, zip_filepath)
    zip_ref = zipfile.ZipFile(zip_filepath, "r")
    zip_ref.extractall() # store locally
    obj = zip_ref.read(os.path.join(UNZIP_FOLDER, FILENAME))

    dump(obj, data_file)
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
      default='ratings.csv')

  args = parser.parse_args()

  data_dir = os.path.join(args.work_dir, 'data')
  run(args.data_sources, args.zip_filepath, args.data_name, data_dir)
