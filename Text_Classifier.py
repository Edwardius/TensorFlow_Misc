import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

datasetDirectory = os.path.join(os.path.dirname(dataset), 'aclImdb')
trainDirectory = os.path.join(datasetDirectory, 'train')

# because the file contains additional useless folders, these must be removed
remove_dir = os.path.join(trainDirectory, 'unsup')
shutil.rmtree(remove_dir)

# create a validation set alongside the training and testing sets
batchSize = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size = batchSize,
    validation_split = 0.2, # 20% of the training set will be a validation set
    subset='training',
    seed=seed)







