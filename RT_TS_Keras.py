#Prepare to display sample from dataset

import numpy as np
import seaborn as sns
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import os


#Defining pic size

img_size = 48

#Defining Basepath

base_path = "D:\\TrafficSigns\\"
print (base_path)
train_dir = os.path.join(base_path, "Training\\")
test_dir = os.path.join(base_path, "Testing\\")
print (train_dir)

#Initialize figure

plt.figure(1, figsize=(12,20))
cpt = 0

#Creating figure

for images in os.listdir(train_dir):
    print (os.path.join(train_dir,images))
    for i in range (0,1):
        cpt = cpt + 1
        plt.subplot(7,10,cpt)
        img = load_img(train_dir + images + "\\" +os.listdir(train_dir + images)[i], target_size=(img_size, img_size))
        plt.axis("off")
        plt.imshow(img, cmap="gray")

plt.tight_layout()
plt.show()

#Count number of train images per signal

for signal in os.listdir(train_dir):
    print(str(len(os.listdir(train_dir + "\\" + signal))) + " " + str(int(signal)) + " images" )



#--------------------------------Setup Data Generator----------------------------------------------#


from keras.preprocessing.image import ImageDataGenerator

#number of images to feed into the NN for every batch

batch_size = 15

datagen_train = ImageDataGenerator()
datagen_test = ImageDataGenerator()

train_generator = datagen_train.flow_from_directory(train_dir,
                                                    target_size=(img_size, img_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode="categorical",
                                                    shuffle=True)

test_generator = datagen_test.flow_from_directory(test_dir,
                                                    target_size=(img_size, img_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode="categorical",
                                                    shuffle=False)

#--------------------------------Define CNN Architecture-------------------------------------------#

from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD

#Number of possible labels

nb_classes = 62

#Initialising the CNN

model = Sequential()

# 1st-Layers Convolution

model.add(Conv2D(64,(3,3), padding="same", input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 2nd-Layers Convolution layer

model.add(Conv2D(128,(5,5), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 3nd-Layers Convolution layer

model.add(Conv2D(512,(3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 4nd-Layers Convolution layer

model.add(Conv2D(512,(3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Flattening

model.add(Flatten())

# 1st-Fully Connected Layer

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.25))

# 2nd-Fully Connected Layer

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.25))

# Output Layer

model.add(Dense(nb_classes, activation="softmax"))

opt = Adam(lr = 0.0001)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

#------------------------------Training Model------------------------------------------------------#

#Number of epochs

epochs = 50

from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("model_weights.h5", monitor="val_acc", verbose=1, save_best_only=True, mode="max")
callbacks_list = [checkpoint]

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=train_generator.n//train_generator.batch_size,
                              epochs=epochs,
                              validation_data=test_generator,
                              validation_steps=test_generator.n//test_generator.batch_size,
                              callbacks=callbacks_list)

