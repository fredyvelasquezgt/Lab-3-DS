from numpy import asarray
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import random
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
xTrain = train.fillna(0)
xTest = test.fillna(0)

yTrain = to_categorical(xTrain['label'])
del xTrain['label']

xTrain = xTrain.values.reshape(-1, 28, 28, 1)
xTest = xTest.values.reshape(-1, 28, 28, 1)

displayList = []

for x in range(9):
    displayList.append(random.randint(1, len(xTrain)+1))

for x in range(len(displayList)):
    plt.subplot(331 + x)
    plt.imshow(xTrain[displayList[x]][:, :, 0])

dataGenerator = ImageDataGenerator(
    zoom_range=.1, height_shift_range=.1, width_shift_range=.1)

model = Sequential()

model.add(Conv2D(20, kernel_size=(3, 3),
          activation='relu', input_shape=(28, 28, 1)))

model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(.25))
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(.25))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam', metrics=['accuracy'])

model.fit_generator(dataGenerator.flow(xTrain, yTrain, batch_size=16),
                    steps_per_epoch=500,
                    epochs=20)

yPred = model.predict(xTest)


image = Image.open('numero1.jpg')
new_image = image.resize((28, 28)).convert('L')

data = asarray(new_image)

imagen_array = data.reshape(-1, 28, 28, 1)


def myMax(list1):

    # Assume first number in list is largest
    # initially and assign it to variable "max"
    max = list1[0]
 # Now traverse through the list and compare
    # each number with "max" value. Whichever is
    # largest assign that value to "max'.
    for x in list1:
        if x > max:
            max = x

    # after complete traversing the list
    # return the "max" value
    return max


res = model.predict(imagen_array)
# array to list
res = res.tolist()
print(res)
for x in range(len(res)):
    res[x] = res[x].index(myMax(res[x]))
    print(res[x])

print('El numero a predicho es: ', res[x])
