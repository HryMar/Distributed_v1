import os
import numpy as np
import shutil
from config import Config

dataset = Config.DATASET_DIR
if Config.CUSTOM_VALIDATION:
      dataset = os.path.join(dataset, "train")
# original_dir = "/home/ubuntu/Mark/ARVI/DATA"
# dataset = "Choose_model_dataset/train"
print(dataset)
img_size =Config.IMG_SIZE


#create paths
# augmented_dataset = os.path.join(original_dir,dataset.split("/")[0]) + "_aug"
# dataset = os.path.join(original_dir,dataset)
classes_names = [name for name in os.listdir(dataset) if os.path.isdir(os.path.join(dataset,name))]

#remove '.DS_Store'
try:
    classes_names.remove('.DS_Store')
    os.rmdir(os.path.join(dataset,'.DS_Store'))
except:
    pass

#initialize lists of images and shuffle them
images_of_class = []
for i in range(0,len(classes_names)):
    imgs = os.listdir(os.path.join(dataset,classes_names[i]))
    try:
        imgs.remove('.DS_Store')
        os.rmdir(os.path.join(dataset + "/" + classes_names[i],'.DS_Store'))
    except:
        pass
    np.random.shuffle(imgs)
    images_of_class.append(imgs)


#augmentation
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order


#let's balance our classes

#firstly, find maximum elements in class
m = 0
for i in range(0, len(images_of_class)):
    if (len(images_of_class[i]) > m):
        m = len(images_of_class[i])

#augment to balance
#and resize 

#flag
not_equal = True

#handle names repeat
zzz = 'z_'

while not_equal:
    not_equal = False

    #get lists of images in each class
    images_of_class = []
    for i in range(0,len(classes_names)):
        imgs = os.listdir(os.path.join(dataset,classes_names[i]))
        np.random.shuffle(imgs)
        images_of_class.append(imgs)

    for class_n in range(0,len(classes_names)):
        #calculate number of images to augment
        n = len(os.listdir(os.path.join(dataset,classes_names[class_n])))
        if (m-n<n):
            n = m-n
        print("class number {}:{}".format(class_n, n))

        #update flag if classes are unequal
        if (n > 0):
            not_equal = True

        #create list of images
        images = np.empty((n,img_size,img_size,3))
        for i in range(0, n):
            images[i] = Image.open(os.path.join(dataset,classes_names[class_n],
                                   images_of_class[class_n][i])).resize((img_size,img_size))

        #augmentated images
        images_aug = seq.augment_images(images.astype('uint8'))

        for i in range(0, n):
            im = Image.fromarray(images_aug[i])
            im.save(os.path.join(dataset,classes_names[class_n],
                                 zzz + 'augms' + images_of_class[class_n][i]))
    zzz = 'z' + zzz