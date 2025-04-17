import pandas as pd 
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def plotRawImage(csv_file):

    data = pd.read_csv(csv_file, header=None)
    image_array = data.to_numpy()
    # image_array = cv2.resize(image_array, (100, 100), interpolation=cv2.INTER_LINEAR)
    plt.imshow(image_array, cmap='inferno', interpolation='nearest')
    plt.colorbar()
    plt.show()
    
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------