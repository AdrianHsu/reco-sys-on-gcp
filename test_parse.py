import tensorflow as tf
import numpy as np

tf.compat.v1.enable_eager_execution()
filenames = ['train-dataset_part-00000-of-00003']
raw_dataset = tf.data.TFRecordDataset(filenames)
arr = []
for raw_record in raw_dataset.take(3):
  # The repr() function returns a printable representation of the given object.
  print(repr(raw_record))
  arr.append(raw_record)

example = tf.train.Example()
example.ParseFromString(arr[0].numpy())
print(example)

# features {
#   feature {
#     key: "item"
#     value {
#       int64_list {
#         value: 755
#       }
#     }
#   }
#   feature {
#     key: "rating"
#     value {
#       int64_list {
#         value: 2
#       }
#     }
#   }
#   feature {
#     key: "user"
#     value {
#       int64_list {
#         value: 921
#       }
#     }
#   }
# }