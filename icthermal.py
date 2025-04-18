# -------------------------------------------------------
# -------------------------------------------------------

from colorspacious import cspace_converter
from matplotlib import colormaps
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import keras
import glob
import os
import cv2

# -------------------------------------------------------
# -------------------------------------------------------

"""Teste de plotar imagem a partir do arquivo .csv"""

from google.colab import drive
drive.mount('/content/drive')

def plot_csv_image(csv_file):
    # Lê o arquivo CSV
    data = pd.read_csv(csv_file, header=None)

    # Converte os dados para um array NumPy
    image_array = data.to_numpy()

    # image_array = cv2.resize(image_array, (100, 100), interpolation=cv2.INTER_LINEAR)

    # Plota a imagem
    plt.imshow(image_array, cmap='inferno', interpolation='nearest')
    plt.colorbar()
    plt.title("Imagem a partir de CSV")
    plt.show()

csv_file = "/content/drive/MyDrive/icThermal/data/diagnosticos/pseudoartrose/pseudoartrose_de_tibia/pessoa01/pessoa01_avaliacao02.csv"
plot_csv_image(csv_file)

# teste do mantova
csv_file = "/content/drive/MyDrive/Pesquisa/Iniciacao_Cientifica/Alunos_24-25/Matheus_Bonfim-Classificacao_de_Termografia/Dataset/data/diagnosticos/pseudoartrose/pseudoartrose_de_tibia/pessoa01/pessoa01_avaliacao01.csv"

data = pd.read_csv(csv_file, header=None)

# Converte os dados para um array NumPy
image_array = data.to_numpy()

# Plota a imagem
plt.imshow(image_array, cmap='inferno', interpolation='nearest', vmin = 25, vmax = 40)
plt.colorbar()
plt.title("Imagem a partir de CSV")
plt.show()

# df = pd.read_csv(path,sep=',',decimal=".")

#     ########## Código ##########

#     #Convert the dataframe to array
#     arr = df.to_numpy()

#     #Normalize the array
#     arr_norm = normalize(arr)
#     img = arr_norm
#     scale_percent = 300 # percent of original size
#     width = int(img.shape[1] * scale_percent / 100)
#     height = int(img.shape[0] * scale_percent / 100)
#     dim = (width, height)


# Exemplo de uso:
# pasta_saudaveis = "/content/drive/MyDrive/icThermal/data/saudaveis/"
# pasta_diagnosticos = "/content/drive/MyDrive/icThermal/data/diagnosticos/"

pasta_saudaveis = "data/saudaveis"
pasta_diagnosticos = "data/diagnosticos"

lista1 = carregar_csvs_para_array(pasta_saudaveis)
lista2 = carregar_csvs_para_array(pasta_diagnosticos)

print(len(lista1))
print(len(lista2))

# 560 saudaveis
# 171 diagnosticos

x_saudaveis = np.array(lista1)
x_diagnosticos = np.array(lista2)

y_saudaveis = np.ones(len(x_saudaveis))
y_diagnosticos = np.zeros(len(x_diagnosticos))

X = np.concatenate((x_saudaveis,x_diagnosticos))
Y = np.concatenate((y_saudaveis,y_diagnosticos))

num_classes = 2
input_shape = (239, 320, 1)
# input_shape = (239, 320, 3) # RGB

#redimensionando as imagens para ficar mais leve
# def redimensionar_imagens(imagem, novo_tamanho = (50,50,1)):
    # return cv2.resize(imagem, novo_tamanho, interpolation=cv2.INTER_LINEAR

input_shape = (50, 50, 1)

listaX = list(X)
X = np.array([cv2.resize(img, (50, 50), interpolation=cv2.INTER_LINEAR) for img in X])

# Amostra estratificada de acordo com os valors de Y (classe)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# TODO: balanceamento no conjunto de treinamento
# Data augmentation da classe minoritaria (quais operacoes)
# usar data augmentation -> https://albumentations.ai/

print("Tamanho do X_train:", x_train.shape)
print("Tamanho do X_test:", x_test.shape)
print("Tamanho do y_train:", y_train.shape)
print("Tamanho do y_test:", y_test.shape)

"""Modelo"""

model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

model.compile(
    # https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
    loss=keras.losses.SparseCategoricalCrossentropy(), # binary cross entropy (diagnostico ou saudavel)
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)

batch_size = 128
epochs = 20

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath="model_at_epoch_{epoch}.keras"),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=6),
]

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.15,
    callbacks=callbacks,
)
score = model.evaluate(x_test, y_test, verbose=0)

score

# Carregar dados
data = pd.read_csv("/content/drive/MyDrive/icThermal/data/diagnosticos/fraturas_osseas/fratura_de_tibia/pessoa01/pessoa01_avaliacao01.csv", header=None)


# data = pd.read_csv("/content/drive/MyDrive/icThermal/data/saudaveis/tibia_deitado/Deitado_69.csv", header=None)
image_array = data.to_numpy()

# Redimensionar
image_array = cv2.resize(image_array, (50, 50), interpolation=cv2.INTER_LINEAR)

# Adicionar dimensões extras para compatibilidade com o modelo
image_array = np.expand_dims(image_array, axis=-1)
image_array = np.expand_dims(image_array, axis=0)

# Plota a imagem
plt.imshow(image_array[0, :, :, 0], cmap='inferno', interpolation='nearest')
plt.colorbar()
plt.title("Imagem a partir de CSV")
plt.show()

print(image_array.shape)

# Fazer previsão
predictions = model.predict(image_array)
score = float(keras.activations.sigmoid(predictions[0][0]))

print(f"This image is {100 * (1 - score):.2f}% diagnóstico and {100 * score:.2f}% saudável.")