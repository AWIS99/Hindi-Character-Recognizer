import numpy as np
from keras import layers
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D
from keras.layers import AveragePooling2D,MaxPooling2D,Dropout,GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.utils import np_utils,print_summary
import pandas as pd
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K

data=pd.read_csv("/home/awis/Desktop/ML/data_sets/data.csv")
dataset=np.array(data)
np.random.shuffle(dataset)
x=dataset
y=dataset
x=x[:, 0:1024]
y=y[:, 1024]

x_train=x[0:70000, :]
x_train=x_train/255.
x_test=x[70000:72001, :]
x_test=x_test/255.
print(x_train.shape)
# Reshape
y=y.reshape(y.shape[0],1)
y_train=y[0:70000, :]
y_train=y_train.T
y_test=y[70000:72001, :]
y_test=y_test.T

#print(y.shape)
#print(y_train.shape)
#print(y_test.shape)

print("Number of training examples = "+str(x_train.shape[0]))
print("Number of test samples = " + str(x_test.shape[0]))
print("x_train shape: "+ str(x_train.shape))
print("y_train shape: "+ str(y_train.shape))
print("x_test shape: "+ str(x_test.shape))
print("y_test shape: "+ str(y_test.shape))

image_x=32
image_y=32

train_y=np_utils.to_categorical(y_train) # feature vector conversion
test_y=np_utils.to_categorical(y_test)   # feature vector conversion
#print(train_y)
#print(train_y.shape)
train_y=train_y.reshape(train_y.shape[1],train_y.shape[2])
test_y=test_y.reshape(test_y.shape[1],test_y.shape[2])
x_train=x_train.reshape(x_train.shape[0],image_x,image_y, 1)
x_test=x_test.reshape(x_test.shape[0],image_x,image_y, 1)

print("X_train shape: " + str(x_train.shape))
print("Y_train shape: " + str(train_y.shape))

def keras_model(image_x,image_y):
    num_of_classes = 37
    model = Sequential() #sequential model
    model.add(Conv2D(filters=32, kernel_size=(5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "devanagari.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]

    return model, callbacks_list

model, callbacks_list = keras_model(image_x, image_y)
model.fit(x_train, train_y, validation_data=(x_test, test_y), epochs=8, batch_size=64,callbacks=callbacks_list)
scores = model.evaluate(x_test, test_y, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
print_summary(model)
model.save('devanagari.h5')
