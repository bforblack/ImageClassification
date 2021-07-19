import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
class ImageDataGetter():
    def __init__(self):
        logging.info('----Fetching Image Data-----')

    def _provideImageGeneratorForNonModel(self,testData):

            if testData == True:
                return ImageDataGenerator(rescale=1/255,featurewise_center = True,
                samplewise_center = True,
                featurewise_std_normalization = True,
                samplewise_std_normalization = True,
                zca_whitening = True,
                zca_epsilon = 1e-6,
                rotation_range = 40,
                width_shift_range = 0.4,
                height_shift_range = 0.4,

                shear_range = 0.4,
                zoom_range = 0.4,
                channel_shift_range = 0.4,
                fill_mode = 'nearest',
                cval = 0.4,
                horizontal_flip = True,
                vertical_flip = True,
                preprocessing_function = True,
                validation_split=0.3)
            else:
                return ImageDataGenerator(rescale=1/255,featurewise_center = True,
                samplewise_center = True,
                featurewise_std_normalization = True,
                samplewise_std_normalization = True,
                zca_whitening = True,
                zca_epsilon = 1e-6,
                rotation_range = 40,
                width_shift_range = 0.4,
                height_shift_range = 0.4,
                shear_range = 0.4,
                zoom_range = 0.4,
                channel_shift_range = 0.4,
                fill_mode = 'nearest',
                cval = 0.4,
                horizontal_flip = True,
                vertical_flip = True,
                preprocessing_function = True
               )
