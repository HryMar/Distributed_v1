import os

class Config(object):

    CUSTOM_VALIDATION = False

    DATASET_DIR = "/Users/Markos/ARVI/DATA/light/"

    #make augmentation to balance classes?
    AUGMENTATION = True

    #if validation is not custom:
    VALIDATION_SIZE = 0.1

    #Image size to convert
    #now only 299 is available
    IMG_SIZE = 299

    ######################################################################