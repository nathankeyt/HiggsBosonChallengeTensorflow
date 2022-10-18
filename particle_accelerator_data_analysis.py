import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import csv
import pandas as pd
import random
import numpy as np

DATADIR = "C:/Users/nathankeyt/Documents/MachineLearningProject/"
CATEGORIES = ["s", "b"]

df = pd.read_csv('training.csv')

mappings = {
    's': 1,
    'b': 0
}

df['Label'] = df['Label'].apply(lambda x: mappings[x]).tolist()

df = df.where(df > -998.0, 0)

df = df.reindex(np.random.permutation(df.index))

headers = list(df.head(0))
headers = headers[1:(len(headers))]

print(headers)

x_train = []

for name in headers:
    x_train.append(df[name].tolist())

total_s = 0
total_b = 0

finalx = []

z = 0

while (z < 32):
    finalx.append([])
    z += 1
    
print(x_train[len(x_train) - 1][0])

for index, i in enumerate(x_train[len(x_train) - 1]):
    if x_train[len(x_train) - 1][index] == 1:
        z =0
        while (z < 32):
            finalx[z].append(x_train[z][index])
            z +=1
    elif (total_b <= 85667):
        total_b += 1
        z = 0
        while (z < 32):
            finalx[z].append(x_train[z][index])
            z += 1 

x_train = finalx

z = 0

while (z < 32):
    x_train[z] = np.asarray(x_train[z])
    z += 1


x_train = np.asarray(x_train)

np.random.shuffle(x_train.T)

y_train = x_train[len(x_train) - 1]
sample_weights = x_train[len(x_train) - 2]

x_train = np.delete(x_train, len(x_train) - 1, 0)
x_train = np.delete(x_train, len(x_train) - 2, 0)

z = 0

while (z < 30):
    print(x_train[z][1])
    z += 1

x_test = []
new_x_train = []


x_train = tf.keras.utils.normalize(x_train, axis=1)


z = 0

while (z < 30): 
    print(x_train[z][0])
    """ x_test.append(np.asarray(x_train[z][200001:250000]).astype('float32').reshape((-1,1)))
    new_x_train.append(np.asarray(x_train[z][:200000]).astype('float32').reshape((-1,1))) """
    x_test.append(np.asarray(x_train[z][121335:171334]).astype('float32').reshape((-1,1)))
    new_x_train.append(np.asarray(x_train[z][:121334]).astype('float32').reshape((-1,1)))
    z += 1
    
x_train = new_x_train


""" y_test = y_train[200001:250000]
y_train = y_train[:200000]
sample_weights = sample_weights[:200000] """
y_test = y_train[121335:171334]
y_train = y_train[:121334]
sample_weights = sample_weights[:121334]

y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
sample_weights = np.asarray(sample_weights).astype('float32').reshape((-1,1))

input1 = tf.keras.layers.Input(shape=(1,))
input2 = tf.keras.layers.Input(shape=(1,))
input3 = tf.keras.layers.Input(shape=(1,))
input4 = tf.keras.layers.Input(shape=(1,))
input5 = tf.keras.layers.Input(shape=(1,))
input6 = tf.keras.layers.Input(shape=(1,))
input7 = tf.keras.layers.Input(shape=(1,))
input8 = tf.keras.layers.Input(shape=(1,))
input9 = tf.keras.layers.Input(shape=(1,))
input10 = tf.keras.layers.Input(shape=(1,))
input11 = tf.keras.layers.Input(shape=(1,))
input12 = tf.keras.layers.Input(shape=(1,))
input13 = tf.keras.layers.Input(shape=(1,))
input14 = tf.keras.layers.Input(shape=(1,))
input15 = tf.keras.layers.Input(shape=(1,))
input16 = tf.keras.layers.Input(shape=(1,))
input17 = tf.keras.layers.Input(shape=(1,))
input18 = tf.keras.layers.Input(shape=(1,))
input19 = tf.keras.layers.Input(shape=(1,))
input20 = tf.keras.layers.Input(shape=(1,))
input21 = tf.keras.layers.Input(shape=(1,))
input22 = tf.keras.layers.Input(shape=(1,))
input23 = tf.keras.layers.Input(shape=(1,))
input24 = tf.keras.layers.Input(shape=(1,))
input25 = tf.keras.layers.Input(shape=(1,))
input26 = tf.keras.layers.Input(shape=(1,))
input27 = tf.keras.layers.Input(shape=(1,))
input28 = tf.keras.layers.Input(shape=(1,))
input29 = tf.keras.layers.Input(shape=(1,))
input30 = tf.keras.layers.Input(shape=(1,))
merged = tf.keras.layers.Concatenate(axis=1)([input1, 
                                              input2, 
                                              input3, 
                                              input4, 
                                              input5, 
                                              input6, 
                                              input7, 
                                              input8, 
                                              input9,
                                              input10,
                                              input11,
                                              input12,
                                              input13,
                                              input14,
                                              input15,
                                              input16,
                                              input17,
                                              input18,
                                              input19,
                                              input20,
                                              input21,
                                              input22,
                                              input23,
                                              input24,
                                              input25,
                                              input26,
                                              input27,
                                              input28,
                                              input29,
                                              input30,])
y = tf.keras.layers.Dense(30, input_dim=30, activation=tf.keras.activations.sigmoid, use_bias=True)(merged)
y = tf.keras.layers.Dense(32, activation='relu')(y)
y = tf.keras.layers.Dense(32, activation='relu')(y)
y = tf.keras.layers.Dense(32, activation='relu')(y)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(y)
model = tf.keras.models.Model(inputs=[input1, 
                                              input2, 
                                              input3, 
                                              input4, 
                                              input5, 
                                              input6, 
                                              input7, 
                                              input8, 
                                              input9,
                                              input10,
                                              input11,
                                              input12,
                                              input13,
                                              input14,
                                              input15,
                                              input16,
                                              input17,
                                              input18,
                                              input19,
                                              input20,
                                              input21,
                                              input22,
                                              input23,
                                              input24,
                                              input25,
                                              input26,
                                              input27,
                                              input28,
                                              input29,
                                              input30,], outputs=outputs)
model.summary()

with tf.device('/cpu:0'):
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=20)

    val_loss, val_acc = model.evaluate(x_test, y_test)

    print(val_loss, val_acc)
    