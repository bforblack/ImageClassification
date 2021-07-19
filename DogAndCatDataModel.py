from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import logging

class DogCatOrignal:
    def __init__(self):
        logging.info('----Dog And Cat Normal Model----')

    def selfCreatedModel(self):
        model=Sequential()
        model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
        model.add(layers.MaxPool2D(2,2))
        model.add(layers.Conv2D(64,(3,3),activation='sigmoid'))
        model.add(layers.MaxPool2D(2,2))
        model.add(layers.Conv2D(124,(3,3),activation='relu'))
        model.add(layers.MaxPool2D(2,2))
        model.add(layers.Flatten())
        model.add(layers.Dense(512,activation='sigmoid'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(224,activation='relu'))
        model.add(layers.Dense(64,activation='sigmoid'))
        model.add(layers.Dense(24,activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',metrics=['acc'])
        logging.info(model.summary())
        return model



