# -------------------------------------------------------
# -------------------------------------------------------

import pandas as pd
import numpy as np
import itertools
from tensorflow import keras

from src.config import *
from src.DLExperiment import deepLearningExperiment

# -------------------------------------------------------
# -------------------------------------------------------

if __name__ == "__main__":

    # all combination of experiments
    all_jobs = list(itertools.product(SEEDS, AVAILABLE_TYPES_OF_IMAGE, AVAILABLE_MODELS))
    # print(all_jobs)
    for job in all_jobs:

        current_seed, type_of_image, dl_model = job
        keras.utils.set_random_seed(current_seed)

        print(" ========================= ")
        print("* Running DL Job")
        print(" ========================= ")
        print("* Type of image: ", type_of_image)
        print("* Model: ", dl_model)
        print("* Data Augmentation: ", False)
        print("* Seed: ", current_seed)
        print(" ========================= ")
        try:
            deepLearningExperiment(current_seed=current_seed,
                                   dl_model=dl_model,
                                   type_of_image=type_of_image,
                                   data_augmentation=False)
        except Exception as e:
            print("Error: ", e)
            continue # run the next loop
   
        print(" ========================= ")
        
# -------------------------------------------------------
# -------------------------------------------------------
