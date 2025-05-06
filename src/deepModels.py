# -------------------------------------------------------
# -------------------------------------------------------

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19, ResNet50

# -------------------------------------------------------
# Classical CNN architecture
# -------------------------------------------------------

def get_CNN_model(input_shape):

	CNNmodel = models.Sequential()
	CNNmodel.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
	CNNmodel.add(layers.MaxPooling2D((2, 2)))
	CNNmodel.add(layers.Dropout(0.25))
	CNNmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))
	CNNmodel.add(layers.MaxPooling2D((2, 2)))
	CNNmodel.add(layers.Dropout(0.25))
	CNNmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))
	CNNmodel.add(layers.Flatten())
	CNNmodel.add(layers.Dense(64, activation='relu'))
	CNNmodel.add(layers.Dropout(0.5))
	CNNmodel.add(layers.Dense(1, activation="sigmoid"))

	return(CNNmodel)

# -------------------------------------------------------
# -------------------------------------------------------

def get_VGG19_model_Keras(input_shape) :

    VGGmodel  = models.Sequential()
    baseModel = VGG19(input_shape=input_shape, weights='imagenet',include_top=False)
    baseModel.trainable = False
    VGGmodel.add(baseModel)
    VGGmodel.add(layers.Flatten())
    VGGmodel.add(layers.Dense(4096,activation = 'relu'))
    VGGmodel.add(layers.Dense(4096,activation = 'relu'))
    VGGmodel.add(layers.Dense(1,activation = 'sigmoid'))
    return(VGGmodel)

# -------------------------------------------------------
# Light weight convolutional neural network and low-dimensional images 
#   transformation approach for classification of thermal images
# https://www.sciencedirect.com/science/article/pii/S2214157X22009078
# -------------------------------------------------------

def get_LW_CNN_model_Taspinar(input_shape):

    LWCNN_model = models.Sequential()
    LWCNN_model.add(layers.Conv2D(6, (5, 5), input_shape=input_shape))
    LWCNN_model.add(layers.LeakyReLU(alpha=0.01))
    LWCNN_model.add(layers.MaxPooling2D((2, 2)))

    LWCNN_model.add(layers.Conv2D(16, (5, 5)))
    LWCNN_model.add(layers.LeakyReLU(alpha=0.01))
    LWCNN_model.add(layers.MaxPooling2D((2, 2)))

    LWCNN_model.add(layers.Conv2D(64, (3, 3)))
    LWCNN_model.add(layers.LeakyReLU(alpha=0.01))
    LWCNN_model.add(layers.MaxPooling2D((2, 2)))

    LWCNN_model.add(layers.Flatten())
    LWCNN_model.add(layers.Dense(128))
    LWCNN_model.add(layers.LeakyReLU(alpha=0.01))

    LWCNN_model.add(layers.Dropout(0.2))
    LWCNN_model.add(layers.Dense(1, activation="sigmoid"))

    return(LWCNN_model)

# -------------------------------------------------------
# -------------------------------------------------------

def get_ResNet50_model_Keras(input_shape) :

  ResNetmodel = models.Sequential()
  baseModel = ResNet50(input_shape=input_shape, weights='imagenet',include_top=False)
  baseModel.trainable = True
  ResNetmodel.add(baseModel)
  ResNetmodel.add(layers.GlobalAveragePooling2D())
  ResNetmodel.add(layers.Dropout(0.5))
  ResNetmodel.add(layers.Dense(128, activation='relu'))
  ResNetmodel.add(layers.Dense(2,activation = 'softmax'))

  return(ResNetmodel)

# -------------------------------------------------------
# -------------------------------------------------------


