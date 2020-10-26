#!/usr/bin/env python

import argparse
import os
import random

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform.beam.impl as beam_impl

from apache_beam.io import tfrecordio
from apache_beam.options.pipeline_options import PipelineOptions

FILENAME = "ratings-100k.csv"
SKIP_HEADER = 0

# FILENAME = "ratings-25m.csv"
# SKIP_HEADER = 1

class DataToTfExampleDoFn(beam.DoFn):
  """
    Convert Data to TFExample
  """
  def __init__(self):
    pass

  @staticmethod
  def _int64_feature(value):
    """
        Get int64 feature
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  @staticmethod
  def _float_feature(value):
    """
        Get float feature
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

  def process(self, data_tuple):
    """
      Convert data to tf-example
    """
    example = tf.train.Example(features=tf.train.Features(
        feature={'user': self._int64_feature(data_tuple['user']),
                 'item': self._int64_feature(data_tuple['item']),
                 'rating': self._float_feature(data_tuple['rating']),
                 }))
    yield example

def run(work_dir, beam_options, data_dir, eval_percent = 20.0):

  if beam_options and not isinstance(beam_options, PipelineOptions):
    raise ValueError(
      '`beam_options` must be {}. '
      'Given: {} {}'.format(PipelineOptions,
        beam_options, type(beam_options)))

  if not work_dir:
    raise ValueError('invalid work_dir')

  tft_temp_dir = os.path.join(work_dir, 'tft-temp')
  train_dataset_dir = os.path.join(work_dir, 'train-dataset')
  eval_dataset_dir = os.path.join(work_dir, 'eval-dataset')

  def shift_by_one(data):
    """Converts string values to their appropriate type."""
    data['user'] = int(data['user']) - 1
    data['item'] = int(data['item']) - 1
    data['rating'] = float(data['rating']) # string to
    return data

  def partitioning(data, num_partitions, eval_percent):
    return int(random.uniform(0, 100) < eval_percent)

  with beam.Pipeline(options = beam_options) as p, beam_impl.Context(temp_dir = tft_temp_dir):

    dataset = (
      p
      | 'Read from GCS' >> beam.io.ReadFromText(os.path.join(data_dir, FILENAME), skip_header_lines = SKIP_HEADER)
      | 'Trim spaces' >> beam.Map(lambda x: x.split())
      | 'Format to dict' >> beam.Map(lambda x: {"user": x[0], "item": x[1], "rating": x[2]})
      | 'Shift by 1' >> beam.Map(shift_by_one)
      # | 'Write to GCS' >> beam.io.textio.WriteToText(os.path.join(data_dir, "processed"))
    )

    data_to_tfexample = DataToTfExampleDoFn()

    train_dataset, eval_dataset = (
      dataset
      | 'DataToTfExample' >> beam.ParDo(data_to_tfexample)
      | 'SerializeProto' >> beam.Map(lambda x: x.SerializeToString())
      | 'Split dataset' >> beam.Partition(partitioning, 2, eval_percent) # train, eval -> 2
    )

    train_dataset_prefix = os.path.join(train_dataset_dir, 'part')
    _ = (
      train_dataset
      | 'Write train dataset' >> tfrecordio.WriteToTFRecord(train_dataset_prefix)
      # | 'Write train dataset' >> beam.io.textio.WriteToText(train_dataset_prefix)
    )

    eval_dataset_prefix = os.path.join(eval_dataset_dir, 'part')
    _ = ( 
      eval_dataset
      | 'Write eval dataset' >> tfrecordio.WriteToTFRecord(eval_dataset_prefix)
      # | 'Write eval dataset' >> beam.io.textio.WriteToText(eval_dataset_prefix)
    )


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


  args, pipeline_args = parser.parse_known_args()
  print(pipeline_args)
  beam_options = PipelineOptions(pipeline_args, save_main_session=True)

  data_dir = os.path.join(args.work_dir, 'data')
  run(args.work_dir, beam_options, data_dir)