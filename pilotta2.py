#!/usr/bin/python2

import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import load

                    
x, y, hands, trumps = load(["data/data1", "data/data2", "data/out1", "data/out2", "data/out3", "data/out4", "data/out5", "data/out6", "data/out7", "data/data3"])

print(len(x))
print(len(y))

y = np.array(y) / 162.
x = np.array(x)
train_data = x[:-20000]
train_labels = y[:-20000]
val_data = x[len(x)-20000:-10000]
val_labels = y[len(x)-20000:-10000]
test_data = x[len(x)-10000:]
test_labels = y[len(x)-10000:]

model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.elu,
                       input_shape=(128,)),
    keras.layers.Dense(64, activation=tf.nn.elu),
    keras.layers.Dense(64, activation=tf.nn.elu),
    keras.layers.Dense(64, activation=tf.nn.elu),
    keras.layers.Dense(64, activation=tf.nn.elu),
    keras.layers.Dense(64, activation=tf.nn.elu),
    keras.layers.Dense(64, activation=tf.nn.elu),
    keras.layers.Dense(64, activation=tf.nn.elu),
    keras.layers.Dense(64, activation=tf.nn.elu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)])

model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss='mse',
              metrics=['mae'])

model.summary()

model.fit(train_data, train_labels, epochs=4)

val_loss, val_acc = model.evaluate(val_data, val_labels)
print('Val accuracy:', val_acc, val_acc*162)
val_pred = model.predict(val_data)
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc, test_acc*162)


#model.save('models/eval.h5')


