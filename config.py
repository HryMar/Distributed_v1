import os

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """

    ###############################  DATASET  ############################
    '''
    Pipeline read dataset from specific directory structure
    There are two optins:
    1.Standart optin:

    DATASET_DIR/
        first_class/
            IMG_1
            IMG_2
            ...
        second_class/
            IMG_1
            IMG_2
            ...
        ...

    2.Option with custom validation:

     DATASET_DIR/
        train/
           first_class/
                IMG_1
                IMG_2
                ...
           second_class/
                IMG_1
                IMG_2
                ...
            ...
        valid/          #this one keep without changes (no augmentation)
                        #so, make balanced validation
           first_class/
                IMG_1
                IMG_2
                ...
           second_class/
                IMG_1
                IMG_2
                ...
            ...

    '''
    CUSTOM_VALIDATION = False

    DATASET_DIR = "/home/ubuntu/Mark/ARVI/DATA/light/"

    #make augmentation to balance classes?
    AUGMENTATION = True

    #if validation is not custom:
    VALIDATION_SIZE = 0.1

    #Image size to convert
    #now only 299 is available
    IMG_SIZE = 299

    ######################################################################