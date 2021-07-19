from ImageDataGetter import ImageDataGetter as imageDataGetter
from DogAndCatDataModel import DogCatOrignal
import logging

class ImageDataGenerator:
    def __init__(self):
        logging.info('-----Inside ImageDataGenerator-----')


    def prepareTrainTestData(self):
       try:
        data_getter=imageDataGetter()._provideImageGeneratorForNonModel(True)
        __trainData=data_getter.flow_from_directory('C:/Users/ShivankarT/Pictures\dogCat/training_set/training_set',target_size=(150,150),class_mode='binary')
        __validationData =data_getter.flow_from_directory('C:/Users/ShivankarT/Pictures/dogCat/training_set/training_set',target_size=(150,150),class_mode='binary',subset='validation')
        __testData=imageDataGetter()._provideImageGeneratorForNonModel(False).flow_from_directory('C:/Users/ShivankarT/Pictures/dogCat/test_set/test_set',target_size=(150,150),class_mode='binary')
        logging.info('-----Image Data fetched properly----')
        return __trainData,__validationData,__testData
       except :
           logging.error('-----Exception Caught in Prepare TrainTestData------')




if __name__ == '__main__':
    train_data,validation_data,testData=ImageDataGenerator().prepareTrainTestData()
    model=DogCatOrignal().selfCreatedModel()
    history=model.fit_generator(train_data,
    steps_per_epoch = 295.4,
    epochs = 10,
    verbose = 1,
    callbacks = None,
    validation_data = validation_data,
    validation_steps = 104.85,
    validation_freq = 1,
    class_weight = None,
    max_queue_size = 10,
    workers = 1,
    use_multiprocessing = False,
    shuffle = True,
    initial_epoch = 0)
    model.save('DogCatOrignal.h5')
    testLoss,testaccuracy=model.evaluate_generator(testData,steps=101.15)
    logging.info('Testloss',testLoss)
    logging.info("testaccuracy",testaccuracy)



