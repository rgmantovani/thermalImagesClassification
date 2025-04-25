# --------------------------------------------------------------
# --------------------------------------------------------------

import pandas as pd

# --------------------------------------------------------------
# function to load thermal images from csv files
# --------------------------------------------------------------

def load_images_from_csv_files(csv_files):
  if not csv_files:
    raise Exception("No *.csv file was found.")

  raw_images = [pd.read_csv(file).to_numpy() for file in csv_files]
  return(raw_images)

# --------------------------------------------------------------
# --------------------------------------------------------------
