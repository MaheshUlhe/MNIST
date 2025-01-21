# MNIST

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import random

random.seed(25)
np.random.seed(25)
tf.random.set_seed(25)
     

from google.colab import drive
drive.mount('/content/drive')
     
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).

train_df = pd.read_csv('/content/drive/MyDrive/mnist dataset/train.csv')
train_df

label	pixel0	pixel1	pixel2	pixel3	pixel4	pixel5	pixel6	pixel7	pixel8	...	pixel774	pixel775	pixel776	pixel777	pixel778	pixel779	pixel780	pixel781	pixel782	pixel783
0	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	4	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
41995	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
41996	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
41997	7	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
41998	6	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
41999	9	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
42000 rows × 785 columns

test_df = pd.read_csv('/content/drive/MyDrive/mnist dataset/test.csv')
test_df
     
pixel0	pixel1	pixel2	pixel3	pixel4	pixel5	pixel6	pixel7	pixel8	pixel9	...	pixel774	pixel775	pixel776	pixel777	pixel778	pixel779	pixel780	pixel781	pixel782	pixel783
0	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
27995	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
27996	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
27997	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
27998	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
27999	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
28000 rows × 784 columns

print("Shape of X_train:", train_df.shape)
print("Shape of Y_train:", test_df.shape)
     
Shape of X_train: (42000, 785)
Shape of Y_train: (28000, 784)

X_train = train_df.drop(labels = ['label'], axis = 1)
y_train = train_df['label']
     

X_train

pixel0	pixel1	pixel2	pixel3	pixel4	pixel5	pixel6	pixel7	pixel8	pixel9	...	pixel774	pixel775	pixel776	pixel777	pixel778	pixel779	pixel780	pixel781	pixel782	pixel783
0	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
41995	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
41996	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
41997	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
41998	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
41999	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
42000 rows × 784 columns

y_train
     
label
0	1
1	0
2	1
3	4
4	0
...	...
41995	0
41996	1
41997	7
41998	6
41999	9
42000 rows × 1 columns

dtype: int64

X_train = X_train / 255.0
X_test  = test_df / 255.0
     

X_train = X_train.values.reshape(-1,28,28,1).astype('float32')
X_test  = X_test.values.reshape(-1,28,28,1).astype('float32')
     

print("Shape of X_train:", X_train.shape)
print("Shape of Y_train:", X_test.shape)
     
Shape of X_train: (42000, 28, 28, 1)
Shape of Y_train: (28000, 28, 28, 1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
     

from sklearn.model_selection import train_test_split
X_trains, X_val, y_trains, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
     

print("X_train shape:", X_trains.shape)
print("y_train shape:", y_trains.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

X_train shape: (33600, 28, 28, 1)
y_train shape: (33600, 10)
X_val shape: (8400, 28, 28, 1)
y_val shape: (8400, 10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
     


model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

     /usr/local/lib/python3.11/dist-packages/keras/src/layers/convolutional/base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)

  # Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 26, 26, 32)          │             320 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 13, 13, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 11, 11, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 5, 5, 64)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 1600)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 128)                 │         204,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 10)                  │           1,290 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 225,034 (879.04 KB)
 Trainable params: 225,034 (879.04 KB)
 Non-trainable params: 0 (0.00 B)


 callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00001,
    patience=10,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False
)
     

from tensorflow.keras.callbacks import EarlyStopping
     

history = model.fit(X_trains, y_trains,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=128,
                    verbose=1,callbacks=callback)

  Epoch 1/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 34s 120ms/step - accuracy: 0.7420 - loss: 0.8161 - val_accuracy: 0.9686 - val_loss: 0.0987
Epoch 2/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 40s 117ms/step - accuracy: 0.9604 - loss: 0.1301 - val_accuracy: 0.9782 - val_loss: 0.0662
Epoch 3/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 32s 120ms/step - accuracy: 0.9715 - loss: 0.0945 - val_accuracy: 0.9837 - val_loss: 0.0517
Epoch 4/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 35s 134ms/step - accuracy: 0.9785 - loss: 0.0696 - val_accuracy: 0.9857 - val_loss: 0.0448
Epoch 5/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 30s 115ms/step - accuracy: 0.9812 - loss: 0.0599 - val_accuracy: 0.9868 - val_loss: 0.0443
Epoch 6/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 41s 115ms/step - accuracy: 0.9827 - loss: 0.0546 - val_accuracy: 0.9877 - val_loss: 0.0415
Epoch 7/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 31s 119ms/step - accuracy: 0.9846 - loss: 0.0491 - val_accuracy: 0.9883 - val_loss: 0.0380
Epoch 8/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 41s 120ms/step - accuracy: 0.9876 - loss: 0.0410 - val_accuracy: 0.9879 - val_loss: 0.0378
Epoch 9/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 41s 119ms/step - accuracy: 0.9869 - loss: 0.0370 - val_accuracy: 0.9873 - val_loss: 0.0407
Epoch 10/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 31s 117ms/step - accuracy: 0.9888 - loss: 0.0348 - val_accuracy: 0.9898 - val_loss: 0.0365
Epoch 11/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 41s 119ms/step - accuracy: 0.9878 - loss: 0.0349 - val_accuracy: 0.9906 - val_loss: 0.0342
Epoch 12/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 31s 118ms/step - accuracy: 0.9905 - loss: 0.0279 - val_accuracy: 0.9899 - val_loss: 0.0332
Epoch 13/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 31s 118ms/step - accuracy: 0.9912 - loss: 0.0252 - val_accuracy: 0.9900 - val_loss: 0.0346
Epoch 14/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 31s 118ms/step - accuracy: 0.9921 - loss: 0.0233 - val_accuracy: 0.9902 - val_loss: 0.0322
Epoch 15/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 30s 116ms/step - accuracy: 0.9927 - loss: 0.0226 - val_accuracy: 0.9904 - val_loss: 0.0340
Epoch 16/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 43s 122ms/step - accuracy: 0.9934 - loss: 0.0207 - val_accuracy: 0.9913 - val_loss: 0.0327
Epoch 17/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 40s 117ms/step - accuracy: 0.9931 - loss: 0.0204 - val_accuracy: 0.9917 - val_loss: 0.0321
Epoch 18/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 41s 119ms/step - accuracy: 0.9938 - loss: 0.0196 - val_accuracy: 0.9908 - val_loss: 0.0360
Epoch 19/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 31s 116ms/step - accuracy: 0.9952 - loss: 0.0140 - val_accuracy: 0.9920 - val_loss: 0.0344
Epoch 20/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 42s 120ms/step - accuracy: 0.9946 - loss: 0.0153 - val_accuracy: 0.9905 - val_loss: 0.0424
Epoch 21/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 31s 117ms/step - accuracy: 0.9940 - loss: 0.0165 - val_accuracy: 0.9912 - val_loss: 0.0356
Epoch 22/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 41s 118ms/step - accuracy: 0.9952 - loss: 0.0142 - val_accuracy: 0.9927 - val_loss: 0.0318
Epoch 23/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 30s 114ms/step - accuracy: 0.9953 - loss: 0.0130 - val_accuracy: 0.9914 - val_loss: 0.0371
Epoch 24/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 42s 118ms/step - accuracy: 0.9954 - loss: 0.0131 - val_accuracy: 0.9912 - val_loss: 0.0347
Epoch 25/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 31s 117ms/step - accuracy: 0.9958 - loss: 0.0120 - val_accuracy: 0.9921 - val_loss: 0.0324
Epoch 26/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 31s 118ms/step - accuracy: 0.9959 - loss: 0.0117 - val_accuracy: 0.9898 - val_loss: 0.0484
Epoch 27/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 30s 115ms/step - accuracy: 0.9955 - loss: 0.0121 - val_accuracy: 0.9907 - val_loss: 0.0395
Epoch 28/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 44s 125ms/step - accuracy: 0.9952 - loss: 0.0129 - val_accuracy: 0.9925 - val_loss: 0.0355
Epoch 29/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 41s 127ms/step - accuracy: 0.9965 - loss: 0.0098 - val_accuracy: 0.9921 - val_loss: 0.0396
Epoch 30/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 32s 122ms/step - accuracy: 0.9966 - loss: 0.0109 - val_accuracy: 0.9923 - val_loss: 0.0388
Epoch 31/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 41s 120ms/step - accuracy: 0.9964 - loss: 0.0100 - val_accuracy: 0.9924 - val_loss: 0.0395
Epoch 32/100
263/263 ━━━━━━━━━━━━━━━━━━━━ 40s 117ms/step - accuracy: 0.9967 - loss: 0.0098 - val_accuracy: 0.9927 - val_loss: 0.0398
Epoch 32: early stopping

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_val, y_val, verbose=1)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
     
263/263 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9929 - loss: 0.0444
Test Loss: 0.0398489348590374
Test Accuracy: 0.9927380681037903

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

[<matplotlib.lines.Line2D at 0x7bb9284f9d10>]

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
     
[<matplotlib.lines.Line2D at 0x7bb92844cf90>]

y_probs = model.predict(X_val)
     
263/263 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step


pred_labels = y_probs.argmax(axis=1)
pred_labels

     
array([8, 1, 9, ..., 3, 0, 9])

# Predicting results and selecting the highest probability index
results = model.predict(X_test).argmax(axis=1)

# Creating the submission DataFrame
submission = pd.DataFrame({
    "ImageId": range(1, len(results) + 1),
    "Label": results
})

# Save to a CSV file for submission
submission.to_csv('submission.csv', index=False)

print(submission)

875/875 ━━━━━━━━━━━━━━━━━━━━ 7s 8ms/step
       ImageId  Label
0            1      2
1            2      0
2            3      9
3            4      9
4            5      3
...        ...    ...
27995    27996      9
27996    27997      7
27997    27998      3
27998    27999      9
27999    28000      2

[28000 rows x 2 columns]
