# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

import glob 
import os
import pandas as pd

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def loadRawImages(subdir):

    csv_files = glob.glob(os.path.join(subdir, "**","*.csv"),recursive=True)

    if not csv_files:
        print("No file was found in the subdir!")
        return None

    arrays = [pd.read_csv(file).to_numpy() for file in csv_files]
    return csv_files, arrays

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------