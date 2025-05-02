# -------------------------------------------------------
# -------------------------------------------------------

import glob 
import os
import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.loadFiles import load_images_from_csv_files
from src.config import *

from src.plotImages import normalize, thermal_to_rgb_image

from src.deepModels import get_CNN_model, get_VGG19_model_Keras, get_LW_CNN_model_Taspinar, get_ResNet50_model_Keras

# -------------------------------------------------------
# -------------------------------------------------------

if __name__ == "__main__":

    TYPE_OF_IMAGE = "rgb" #raw
    MODEL = "cnn" #"vgg19" "lwcnn" "resnet"

    DATA_AUGMENTATION = False # False

    # ----------------------------
    # Set the seed using keras.utils.set_random_seed. This will set:
    # 1) `numpy` seed
    # 2) `tensorflow` random seed
    # 3) `python` random seed
    # ----------------------------

    SEEDS = [404, 666, 42, 171, 51]
    current_seed = SEEDS[1]

    print(" ========================= ")
    print("* Runnin Experiments")
    print(" ========================= ")
    print("* Type of image: ", TYPE_OF_IMAGE)
    print("* Model: ", MODEL)
    print("* Data Augmentation: ", DATA_AUGMENTATION)
    print("* Seed: ", current_seed)
    print(" ========================= ")

    keras.utils.set_random_seed(current_seed)
    
    health_dir = "data/saudaveis"
    osteo_dir  = "data/diagnosticos"
    
    # loading filepaths
    csv_files_health = glob.glob(os.path.join(health_dir, "**","*.csv"),recursive=True)
    csv_files_osteo  = glob.glob(os.path.join(osteo_dir, "**", "*.csv"),recursive=True)
    
    print(" -> Health files = ", len(csv_files_health)) # 560 files
    print(" -> Osteo files = ", len(csv_files_osteo))   # 171 files
    
    # ----------------------------
    # Creating labels
    # ----------------------------

    Y_healthy = np.zeros(len(csv_files_health))
    Y_osteo   = np.ones(len(csv_files_osteo))
    Y = np.concatenate((Y_healthy, Y_osteo))

    # ----------------------------
    # Splitting files into training and testing folds
    # ----------------------------
    all_files = csv_files_health + csv_files_osteo

    x_train_files, x_test_files, y_train, y_test = train_test_split(all_files,
        Y, test_size=0.3, random_state=current_seed, stratify=Y)

    # print("Tamanho do X_train:", len(x_train_files))
    # print("Tamanho do X_test:", len(x_test_files))
    # print("Tamanho do y_train:", y_train.shape)
    # print("Tamanho do y_test:", y_test.shape)

    # --------------------------------------------------------------
    # reading images from files
    # --------------------------------------------------------------

    print(" @ Reading images from files ")

    x_train_images = load_images_from_csv_files(csv_files=x_train_files)
    print(x_train_images[0].shape)

    x_test_images = load_images_from_csv_files(csv_files=x_test_files)
    print(x_test_images[0].shape)


    if(TYPE_OF_IMAGE == "rgb"):
        print(" @ Converting thermal images to RGB")
        new_x_train_images = [thermal_to_rgb_image(x) for x in x_train_images]
        new_x_test_images  = [thermal_to_rgb_image(x) for x in x_test_images]
        input_shape = (239, 320, 3)
    else:
        print (" @ Using raw images - normalized between [0, 1]")
        new_x_train_images = [normalize(x) for x in x_train_images]
        new_x_test_images  = [normalize(x) for x in x_test_images]
        input_shape = (239, 320, 1)

    # -------------------------------------------
    # Creating Training and Testing Folds
    # -------------------------------------------
   
    X_train = np.array(new_x_train_images)
    X_test  = np.array(new_x_test_images)

    # convert values to the inverval [0, 1]
    if(TYPE_OF_IMAGE != "raw"):
        X_train = X_train.astype("float32") / 255
        X_test = X_test.astype("float32") / 255

    # -------------------------------------------
    # Defining DL model
    # -------------------------------------------
    
    match MODEL:
        case "cnn":
            model = get_CNN_model(input_shape=input_shape)
        case "vgg19":
            model = get_VGG19_model_Keras(input_shape=input_shape)
        case "lwcnn":
            model = get_LW_CNN_model_Taspinar(input_shape=input_shape)
        case "resnet":
            model = get_ResNet50_model_Keras(input_shape=input_shape)

    print(" ----------------------------")
    print(" @ Model summary")
    print(" ----------------------------")

    model.summary()

    print(" ----------------------------")
    
    # ----------------------------
    # Traninig the algorithm
    # ----------------------------

    model.compile(optimizer='adam',
                  loss=BinaryCrossentropy(), 
                  metrics=['binary_accuracy', 'accuracy', 'precision', 'recall', 'AUC'])

    # Callbacks
    early_stopper = EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=1)
    csv_logger    = CSVLogger(f"output/log_history_{MODEL}_{TYPE_OF_IMAGE}_seed_{current_seed}.csv", separator=",", append=False)

    print(f" @ Training {MODEL}\n")

    history  = model.fit(X_train, y_train, epochs=100, 
                         validation_split=0.3, batch_size=8, 
                         callbacks=[early_stopper, csv_logger])

    # ----------------------------
    # Evaluating predictions
    # ----------------------------
    print(" @ Evaluating DL model")

    predictions = model.predict(X_test)
    rounded_predictions = np.round(predictions)

    acc = accuracy_score(y_test, rounded_predictions)
    bac = balanced_accuracy_score(y_test, rounded_predictions)
    f1s = f1_score(y_test, rounded_predictions)
    print("----------------------------")
    print("acc = ", acc)
    print("bac = ", bac)
    print("f1c = ", f1s)
    print("----------------------------")

    print(" @ Saving models and performance values")

    performances = ([acc, bac, f1s, current_seed, TYPE_OF_IMAGE, MODEL])
    df_performances = pd.DataFrame(performances).transpose()
    df_performances.columns = ['accuracy', 'balanced_accuracy', 'fscore', 'seed', 'type_of_image', 'model']
    df_performances.to_csv(f"output/performances_{MODEL}_{TYPE_OF_IMAGE}_seed_{current_seed}.csv", index = False)

    # -----------------------------------------------------------
    # adding predictions to a data frame
    # -----------------------------------------------------------

    df_x_test_files = pd.DataFrame(x_test_files)
    df_pred  = pd.DataFrame(rounded_predictions)
    df_label = pd.DataFrame(y_test)
    df_merged = pd.concat([df_x_test_files, df_pred, df_label], axis = 1)
    df_merged.columns = ['filepath', 'predictions', 'labels']
    df_merged.to_csv(f"output/predictions_{MODEL}_{TYPE_OF_IMAGE}_seed_{current_seed}.csv", index = False)
 
    print(" Finished !!! :) ")
    print(" ----------------------------")
    
# -------------------------------------------------------
# -------------------------------------------------------
