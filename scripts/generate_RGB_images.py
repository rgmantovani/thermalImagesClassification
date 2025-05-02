# -------------------------------------------------------
# -------------------------------------------------------

import os
import glob
import cv2

from src.loadFiles import load_images_from_csv_files
from src.plotImages import thermal_to_rgb_image

# -------------------------------------------------------
# -------------------------------------------------------

# This script converts thermal to RGB images and saves them in new directories.
# The images are read from CSV files and the RGB images are saved in the following directories:
# - ./data/dataset/healthy_rgb
# - ./data/dataset/diagnosis_rgb
#
# In a further step, it feeds the image-meta-feature-extractor.py script to extract features from the RGB images.
# https://github.com/gabrieljaguiar/image-meta-feature-extractor
#  >> python run.py ./example/healthy_rgb ./example/output/healthy_rgb.csv
#  >> python run.py ./example/diagnosis_rgb ./example/output/diagnosis_rgb.csv



# -------------------------------------------------------
# -------------------------------------------------------

if __name__ == "__main__":
    
    health_dir = "./data/saudaveis"
    osteo_dir  = "./data/diagnosticos"
    
    # loading filepaths
    csv_files_health = glob.glob(os.path.join(health_dir, "**","*.csv"),recursive=True)
    csv_files_osteo  = glob.glob(os.path.join(osteo_dir, "**", "*.csv"),recursive=True)
    
    print(" -> Health files = ", len(csv_files_health)) # 560 files
    print(" -> Osteo files = ", len(csv_files_osteo))   # 171 files

    # creating new directories for RGB images
    os.makedirs("./data/rgb_images/healthy_rgb", exist_ok=True)
    os.makedirs("./data/rgb_images/diagnosis_rgb", exist_ok=True)
    
    # reading images from files
    print(" @ Reading images from files ")
    health_images = load_images_from_csv_files(csv_files=csv_files_health)
    osteo_images = load_images_from_csv_files(csv_files=csv_files_osteo)
    print(osteo_images[0].shape)

    print(" @ Converting thermal images to RGB")
    rgb_heath_images = [thermal_to_rgb_image(x) for x in health_images]
    rgb_osteo_images = [thermal_to_rgb_image(x) for x in osteo_images]

    # saving images to new directories
    print(" @ Saving RGB images to new directories") 
    for i, img in enumerate(rgb_heath_images):
        cv2.imwrite(f"./data/rgb_images/healthy_rgb/image_{i}.png", img)
    for i, img in enumerate(rgb_osteo_images):  
        cv2.imwrite(f"./data/rgb_images/diagnosis_rgb/image_{i}.png", img)
    print(" @ Finished converting thermal images to RGB")

# -------------------------------------------------------
# -------------------------------------------------------
