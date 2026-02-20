from pathlib import Path
import numpy as np
import pandas as pd

# pega a pasta onde ta o script pra achar os csv
pasta = Path(__file__).parent

# calcula a intensidade: soma todos os pixels e divide por 255
def calcular_intensidade(img):
    pixels = 0
    
    for pixelx in range(img.shape[0]):
        for pixely in range(img.shape[1]):
            pixels += img[pixelx, pixely]
    
    return (pixels / 255.0)
    #return img.sum() / 255.0

# simetria vertical: compara lado esquerdo com direito da imagem
def calcular_simetria_vertical(img):
    sv = 0
    
    for i in range(28):
        for j in range(14):
            sv += np.abs(img[i][j] - img[i][27-j])
    
    return (sv/255.0)

    #esquerda = img[:, :14]
    #direita = img[:, 14:][:, ::-1]
    #sv = np.abs(esquerda - direita).sum() / 255.0
    #return sv6

# simetria horizontal: compara metade de cima com metade de baixo
def calcular_simetria_horizontal(img):
    sh = 0
    
    for i in range(14):
        for j in range(28):
            sh += np.abs(img[i][j] - img[27-i][j])
    
    return (sh/255.0)
    #cima = img[:14, :]
    #baixo = img[14:, :][::-1, :]
    #sh = np.abs(cima - baixo).sum() / 255.0
    #return sh

# simetria total = vertical + horizontal
def calcular_simetria(img):
    return calcular_simetria_vertical(img) + calcular_simetria_horizontal(img)

# le o csv, calcula intensidade e simetria pra cada imagem e retorna um dataframe
def processar(caminho_csv):
    print("Lendo", caminho_csv)
    df = pd.read_csv(caminho_csv, sep=';')

    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values

    n = len(pixels)
    intensidades = np.zeros(n)
    simetrias = np.zeros(n)

    for i in range(n):
        img = pixels[i].reshape(28, 28).astype(np.float64)
        intensidades[i] = calcular_intensidade(img)
        simetrias[i] = calcular_simetria(img)

    resultado = pd.DataFrame({
        "label": labels,
        "intensidade": intensidades,
        "simetria": simetrias,
    })

    return resultado

if __name__ == "__main__":

    # treino
    df_train = processar(pasta / "train.csv")
    df_train.to_csv(pasta / "train_redu.csv", index=False)
    print("Train salvo!")
    print(df_train.head())

    # teste
    df_test = processar(pasta / "test.csv")
    df_test.to_csv(pasta / "test_redu.csv", index=False)
    print("Test salvo!")
    print(df_test.head())