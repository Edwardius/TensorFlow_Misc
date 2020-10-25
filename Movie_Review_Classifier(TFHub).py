import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

trainData, validationData, testData = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

train_examples_batch, train_labels_batch = next(iter(trainData.batch(10)))
print("\nTrain Examples\n", train_examples_batch)
print("\nTrain Labels\n", train_labels_batch)

# import a pretrained text embedding model from TFHub

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hubLayer = hub.KerasLayer(embedding, input_shape=[],
                          dtype=tf.string, trainable=True)

# shows the results of the embedding using the pretrained model
print("\nEmbedded examples\n", hubLayer(train_examples_batch[:3]))

# build the model
model = tf.keras.Sequential()
model.add(hubLayer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()

# compile using a crossentropy loss function and optimizer
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train
history = model.fit(trainData.shuffle(10000).batch(512),
                    epochs=10,
                    validation_data=validationData.batch(512),
                    verbose=1)

# Evaluate
results = model.evaluate(testData.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))


