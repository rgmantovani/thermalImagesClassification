# ------------------------------------------------------------
# ------------------------------------------------------------

# For data augmentation
import albumentations as A
import numpy as np

# ------------------------------------------------------------
# ------------------------------------------------------------

def apply_aug(image, transform):
    augmented = transform(image=image)
    return augmented["image"]

# ------------------------------------------------------------
# ------------------------------------------------------------


def dataAugmentation(X_train, y_train):

    #link com outras transformações caso queria dar uma olhada: https://explore.albumentations.ai/
    augmentations = [ 
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.AdditiveNoise(p=1.0)
    ]

    X_min = X_train[y_train == 1]
    y_min = y_train[y_train == 1]

    augmented_images = []
    augmented_labels = []

    for img in X_min:
        for transform in augmentations:
            aug_img = apply_aug(img, transform)
            augmented_images.append(aug_img)
            augmented_labels.append(1)

    X_augmented = np.array(augmented_images)
    y_augmented = np.array(augmented_labels)

    X_train_balanced = np.concatenate((X_train, X_augmented), axis=0)
    y_train_balanced = np.concatenate((y_train, y_augmented), axis=0)

    return(X_train_balanced, y_train_balanced)

# ------------------------------------------------------------
# ------------------------------------------------------------