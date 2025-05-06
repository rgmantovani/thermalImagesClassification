# -------------------------------------------------------
# -------------------------------------------------------

import glob 
import os.path as file

import pandas as pd
import numpy as np

import tensorflow.keras as keras
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.loadFiles import load_images_from_csv_files
from src.config import *

from src.plotImages import normalize, thermal_to_rgb_image

from src.deepModels import get_CNN_model, get_VGG19_model_Keras, get_LW_CNN_model_Taspinar, get_ResNet50_model_Keras

from src.augmentation import apply_aug, dataAugmentation

# -------------------------------------------------------
# -------------------------------------------------------

def deepLearningExperiment(current_seed, dl_model, type_of_image, data_augmentation=False):
    
    output_file = f"output/performances_{dl_model}_{type_of_image}_seed_{current_seed}.csv"
    if(data_augmentation):
        output_file = f"output/performances_{dl_model}_DataAugmented_{type_of_image}_seed_{current_seed}.csv"
    
    if(file.exists(output_file)):
        print(" @ File already exists, skipping...")
        return 

    print(" @ Loading images from files")
    health_dir = "data/saudaveis"
    osteo_dir  = "data/diagnosticos"
    
    # loading filepaths
    csv_files_health = glob.glob(file.join(health_dir, "**","*.csv"),recursive=True)
    csv_files_osteo  = glob.glob(file.join(osteo_dir, "**", "*.csv"),recursive=True)
    
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

    # --------------------------------------------------------------
    # reading images from files
    # --------------------------------------------------------------

    print(" @ Reading images from files ")

    x_train_images = load_images_from_csv_files(csv_files=x_train_files)
    print(x_train_images[0].shape)

    x_test_images = load_images_from_csv_files(csv_files=x_test_files)
    print(x_test_images[0].shape)


    if(type_of_image == "rgb"):
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
    if(type_of_image != "raw"):
        X_train = X_train.astype("float32") / 255
        X_test = X_test.astype("float32") / 255

    # -------------------------------------------
    # Data Augmentation   
    # -------------------------------------------

    if DATA_AUGMENTATION:
        print(f'Antes do augmentation: {X_train.shape}')
        X_train, y_train = dataAugmentation(X_train=X_train, y_train=y_train)
        print(f'Depois do augmentation: {X_train.shape}')
        model_file   = f"output/best_model_{dl_model}_DA_{type_of_image}_seed_{current_seed}.keras"
        history_file = f"output/log_history_{dl_model}_DA_{type_of_image}_seed_{current_seed}.csv"
        predictions_file = f"output/predictions_{dl_model}_DA_{type_of_image}_seed_{current_seed}.csv"
    else:
        model_file   = f"output/best_model_{dl_model}_{type_of_image}_seed_{current_seed}.keras"
        history_file = f"output/log_history_{dl_model}_{type_of_image}_seed_{current_seed}.csv"
        predictions_file = f"output/predictions_{dl_model}_{type_of_image}_seed_{current_seed}.csv"

    # -------------------------------------------
    # Defining DL model
    # -------------------------------------------
    
    match dl_model:
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
    if (dl_model == "resnet"):
        model.compile(optimizer='adam', loss=CategoricalCrossentropy(),
            metrics=['binary_accuracy', 'accuracy', 'precision', 'recall', 'AUC'])
        y_train = keras.utils.to_categorical(y_train, 2)
        y_test  = keras.utils.to_categorical(y_test, 2)
    else: 
        model.compile(optimizer='adam',loss=BinaryCrossentropy(), 
                  metrics=['binary_accuracy', 'accuracy', 'precision', 'recall', 'AUC'])

    # Callbacks
    model_checkpoint = ModelCheckpoint(filepath=model_file,
                                monitor='val_loss',save_best_only=True,
                                mode='min', verbose=1)
    early_stopper = EarlyStopping(monitor="val_loss", 
                                mode="min", patience=10,
                                verbose=1, restore_best_weights=True)
    csv_logger    = CSVLogger(history_file, separator=",", append=False)

    print(f" @ Training {dl_model}\n")

    history  = model.fit(X_train, y_train, epochs=100, validation_split=0.3, batch_size=16, 
                         callbacks=[csv_logger, model_checkpoint, early_stopper], Shuffle = True)

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
    performances = ([acc, bac, f1s, current_seed, type_of_image, dl_model])
    df_performances = pd.DataFrame(performances).transpose()
    df_performances.columns = ['accuracy', 'balanced_accuracy', 'fscore', 'seed', 'type_of_image', 'model']
    df_performances.to_csv(output_file, index = False)

    # -----------------------------------------------------------
    # adding predictions to a data frame
    # -----------------------------------------------------------

    df_x_test_files = pd.DataFrame(x_test_files)
    df_pred  = pd.DataFrame(rounded_predictions)
    df_label = pd.DataFrame(y_test)
    df_merged = pd.concat([df_x_test_files, df_pred, df_label], axis = 1)
    df_merged.columns = ['filepath', 'predictions', 'labels']
    df_merged.to_csv(predictions_file, index = False)
 
    print(" Finished !!! :) ")
    print(" ----------------------------")
    
# -------------------------------------------------------
# -------------------------------------------------------
