import cv2 # Biblioteca de interpretação e modificação de imagens
import numpy as np # Biblioteca de computação cientifica
import glob # Biblioteca que permite associar todas as imagens
import math # Biblioteca para cálculos matemáticos
from scipy.fftpack import fft
# import pyodbc

def reduce_image(img, width): # Função para reduzir o tamanho de imagens
    new_size = (
        width,
        int(
            width * float(
                img.shape[0] / img.shape[1]
            )
        )
    )

    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

def reduce_domain(image, rect): # Função para reduzir o domínio de uma imagem
    # region = image[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]].copy()
    
    black_image = np.zeros(
        shape=image.shape,
        dtype=np.uint8
    )

    black_image[
        rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]
    ] = image[
        rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]
    ] # Cola pedaço de imagem

    return black_image # Retorna imagem preta com o domínio sobreposto

def findContour(Img): # Função para achar os contornos fundamentais da bombinha
    hie = cv2.findContours(Img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    greater = [] # Maior borda entre as encontradas
    n_greater = 0 # Posição do contorno principal

    for k, ctr in enumerate(hie[0]): # Loop para encontrar a maior borda
        n = len(ctr) # Tamanho da maior borda

        if n > len(greater): # Se for maior que a maior borda já encontrada
            greater = ctr # Armazena maior valor
            n_greater = k # Armazena posição do contorno Principal

    son_list = [] # Lista de Buracos
    d = 0 # Variável de contagem

    for k, cont in enumerate(hie[0]): # Loop para selecionar buracos
        if hie[1][0][k][3] == n_greater:
            son_list.append(cont) # Adiciona na lista de buracos
            d += 1 # Fazer contagem

    return greater, son_list

def importImages(Path): # Função para importar conjunto de imagens
    filenames = glob.glob(Path)
    filenames.sort()
    return filenames

def sumAreas(cont): # Função para pegar contornos
    areas = [] # Lista de áreas

    for k in range(len(cont)): # Percorre todos os contornos
        areas.append(
            cv2 \
            .contourArea(cont[k])
        )

    return sum(areas) # Retorna a soma de todas as áreas internas dos contornos

def linearRegression(ctr, Img, Way):
    [vx, vy, x, y] = cv2.fitLine(
        ctr,
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    )

    cols = Img.shape[1]

    lefty = int(
        ((-x) * vy / vx) + y
    )

    righty = int(
        ((cols - x) * vy / vx) + y
    )

    similar = []
    xx = len(ctr)

    for borda in enumerate(ctr): # Loop para determinar novos pontos do contorno
        ya = borda[1][0][1]
        xa = borda[1][0][0]
        xb = (ya - lefty) / (vy / vx)
        yNew = 0

        if Way == "Dir":
            yNew = (xb - xa) * math.sin(vy / vx)
        elif Way == "Esq":
            yNew = (xa - xb) * math.cos(vy / vx)

        similar.append((xx + 150, yNew + 250))
        xx -= 1

    npObj = np \
        .array(similar, dtype=object) \
        .reshape((-1, 1, 2)) \
        .astype(np.int32)
    
    return (
        (0, lefty),
        (cols - 1, righty),
        npObj
    )

def contourImg(ctr, wei, hey):
    blank_image = np.zeros(
        (hey, wei, 3),
        np.uint8
    )
    
    cv2.drawContours(
        blank_image,
        ctr,
        -1,
        (255, 255, 255),
        3
    ) # Desenha borda

    return blank_image

def rotate(blank_image, roi, theta, way):
    recorte = (
        blank_image[roi[0][1]:roi[1][1],
        roi[0][0]:roi[1][0]]
    )

    (alt, lar) = recorte.shape[:2] # captura altura e largura
    centro = (0, 0)
    
    if way == "esq":
        centro = (
            (lar // 2) + 170,
            (alt // 2) + 170
        ) # acha o centro

    elif way == "dir":
        centro = (
            lar // 2,
            alt // 2
        ) # acha o centro

    M = cv2.getRotationMatrix2D(
        centro,
        ((theta) * 180 / math.pi),
        1
    ) # Girar

    return cv2.warpAffine(
        recorte,
        M,
        (alt, lar)
    )

def fillWithZeros(ctr):
    newContour = []
    lenght = len(ctr)
    twoPot = 2

    while lenght > twoPot:
        twoPot = twoPot * 2

    for k in range(twoPot):
        if k < lenght:
            newContour.append(
                (twoPot - k, ctr[k][0][1])
            )
        else:
            newContour.append(
                (twoPot - k, 250)
            )

    return np \
        .array(newContour) \
        .reshape((-1, 1, 2)) \
        .astype(np.int32)

def unicLine(greater, roi):
    listOfContour = []
    thisContour = []

    for borda in enumerate(greater): # Adiciona apenas coordenadas relevantes
        if borda[1][0][1] != roi[1][1] - 1 and \
            borda[1][0][0] != roi[0][0] and \
            borda[1][0][0] != roi[1][0] - 1 and \
            borda[1][0][1] != roi[0][1] \
        :
            thisContour.append(borda[1][0])
        else:
            if thisContour: # Se não estiver vazio
                listOfContour.append(thisContour)
                thisContour = []

    largest = []

    for number in range(len(listOfContour)): # Loop para achar o maior contorno
        if len(listOfContour[number]) > len(largest):
            largest = listOfContour[number]

    return (
        largest,
        np \
        .array(largest) \
        .reshape((-1, 1, 2)) \
        .astype(np.int32)
    )

# def Cadastrar(DataBD,HoraBD, NotaBD):
# conBD = pyodbc.connect(
#   'Driver={ODBC Driver 11 for SQL Server};'
# 'Server=135.10.1.4;'
# 'Database=L2MILL;'
# 'UID=user_services;'
# 'PWD=gerdauacominas;')
# cursor = conBD.cursor()
# cursor.execute(
#   "INSERT INTO L2_FE_NOTA(DATA, HORA, NOTA_BORDA) VALUES (?,?,?)",
#   DataBD,
#   HoraBD,
#   NotaBD
# )
# conBD.commit()(