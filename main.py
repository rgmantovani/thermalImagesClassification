# -------------------------------------------------------
# -------------------------------------------------------

import glob 
import os
import pandas as pd

# -------------------------------------------------------
# -------------------------------------------------------

if __name__ == "__main__":
    
    health_dir = "data/saudaveis"
    osteo_dir  = "data/diagnosticos"
    
    # loading filepaths
    csv_files_health = glob.glob(os.path.join(health_dir, "**","*.csv"),recursive=True)
    csv_files_osteo  = glob.glob(os.path.join(osteo_dir, "**", "*.csv"),recursive=True)
    
    print("Health files = ", len(csv_files_health)) # 560 files
    print("Osteo files = ", len(csv_files_osteo))   # 171 files
    
    data = pd.read_csv(csv_file, header=None)

    # Converte os dados para um array NumPy
    image_array = data.to_numpy()

    # Plota a imagem
    plt.imshow(image_array, cmap='inferno', interpolation='nearest', vmin = 25, vmax = 40)
    plt.colorbar()
    plt.title("Imagem a partir de CSV")
    plt.show()

    
    
# -------------------------------------------------------
# -------------------------------------------------------
