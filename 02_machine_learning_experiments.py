# -------------------------------------------------------
# -------------------------------------------------------

import random
import numpy as np
import itertools

from src.config import SEEDS, AVAILABLE_ML_ALGORITHMS
from src.MLExperiment import machineLearningExperiment

# -------------------------------------------------------
# -------------------------------------------------------

if __name__ == "__main__":

    # all combination of experiments
    all_jobs = list(itertools.product(SEEDS, AVAILABLE_ML_ALGORITHMS))

    for job in all_jobs:
        seed, algo = job
        np.random.seed(seed)
        random.seed(seed)
        print("Seed: ", seed)
        print("Algorithm: ", algo)
        machineLearningExperiment(current_seed=seed, algorithm=algo)
        print("---------------------------------------")
        print("---------------------------------------")

# -------------------------------------------------------
# -------------------------------------------------------
