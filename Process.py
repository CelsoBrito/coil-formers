import Methods as m # Chamando o modulo métodos

import numpy as np

import copy # Biblioteca para permitir a tarefa de copiar uma lista
import math # Biblioteca para cálculos matemáticos
import cv2 # Biblioteca interpretadora de imagens

from datetime import date
from datetime import time
from datetime import datetime

purpleColor = (113, 89 , 193)
redColor = (197, 48, 48)

def processImage(r1, r2, AllImagens, total, k):
    roi_xld = (
        copy.deepcopy(r1),
        copy.deepcopy(r2)
    ) # Copia duas Rois em uma variável

    GrayImage = cv2.imread(AllImagens[k], 0)
    
    hist = float(
        max(
            cv2.calcHist(
                [GrayImage],
                [0],
                None,
                [256],
                [0, 256]
            )
        )
    )

    if hist > 1100000 and \
        hist < 1450000 \
    : # se tiver muitos pixels brancos
        # Tratamento de imagem: ==-==-==-==-==-==-==-==-==-==-==-==-==-==-==-==
        thres = cv2.threshold(
            GrayImage,
            180,
            255,
            cv2.THRESH_BINARY
        )[1]

        altM, larM = thres.shape[:2]

        blank_image = np.zeros(
            (altM, larM, 3),
            np.uint8
        )
        
        # Roi direita e Roi esquerda: ==-==-==-==-==-==-==-==-==-==-==-==-==-==
        Img = cv2.cvtColor(
            GrayImage,
            cv2.COLOR_GRAY2BGR
        )

        NBurEsq = 0; NBurDir = 0; AreasEsq = 0; AreasDir = 0
        sizeLeft = 0; sizeRight = 0
        contWantedLeft = []; contWantedRight = []
        
        for roi in roi_xld: # Repete o mesmo procedimento nas duas RoIs
            roi[0][1] = int(
                ((roi[1][1] - roi[0][1]) * 0.2) + roi[0][1]
            )

            reduced = m.reduce_domain(thres, roi) # Reduz domínio da imagem

            # Definindo valores: ==-==-==-==-==-==-==-==-==-==-==-==-==-==-==-==
            greater, son_list = m.findContour(reduced) # Contorno principal

            largest, ctr = m.unicLine(greater, roi) # Isola apenas uma linha

            if roi[0][0] == r1[0][0]: # Se a Roi em execução for a primeira
                NBurEsq = len(son_list) # Número de buracos na Roi esquerda
                AreasEsq = m.sumAreas(son_list) # Soma áreas de todos os buracos
            elif roi[0][0] == r2[0][0]: # Se a Roi em execução for a segunda
                NBurDir = len(son_list) # Número de buracos na Roi direita
                AreasDir = m.sumAreas(son_list) # Soma áreas de todos os buracos

            start = (
                largest[0][0],
                largest[0][1]
            )

            end = (
                largest[len(largest) - 1][0],
                largest[len(largest) - 1][1]
            )

            # Pintar imagem: ==-==-==-==-==-==-==-==-==-==-==-==-==-==-==-==\
            blank_image = cv2.drawContours(
                blank_image.copy(),
                ctr,
                -1,
                (255, 255, 255),
                3
            ) # Desenha borda

            Img_Contours = cv2.drawContours(
                Img.copy(),
                ctr,
                -1,
                (0, 255, 0),
                3
            )

            Fill_Contour = cv2.fillPoly(
                Img_Contours.copy(),
                son_list,
                purpleColor
            )

            Img = Fill_Contour.copy() # Img passa a ser a outra imagem editada

            if roi[0][0] == r1[0][0]: # Se a Roi em execução for a primeira
                sizeLeft = int(
                    math.sqrt(
                        ((end[0] - start[0]) ** 2) + ((end[1] - start[1]) ** 2)
                    )
                )

                start, end, contWantedLeft = m.linearRegression(
                    ctr,
                    Img,
                    "Dir"
                )
                contWantedLeft = m.fillWithZeros(contWantedLeft)

            elif roi[0][0] == r2[0][0]: # Se a Roi em execução for a primeira
                sizeRight = int(
                    math.sqrt(
                        ((end[0] - start[0]) ** 2) + ((end[1] - start[1]) ** 2)
                    )
                )

                start, end, contWantedRight = m.linearRegression(
                    ctr,
                    Img,
                    "Esq"
                )

                contWantedRight = m.fillWithZeros(contWantedRight)

            cv2.line(
                Img,
                start,
                end,
                redColor,
                2
            )
        
        if sizeLeft > 400 and \
            sizeRight > 400 \
        :
            # ------------------------------------------
            # --- ESTIMAÇÃO DAS NOTAS PELO CLASSIFICADOR
            # ------------------------------------------
            medNuBur = (NBurEsq + NBurDir) / 2 # Média do número de buracos
            pesoBur = math.exp(-(medNuBur / 15)) # Peso do número de buracos
            medGeral = (AreasEsq + AreasDir) / 2 # Média geral da soma das áreas
            medPond = medGeral * pesoBur # Média ponderada
            
            notaClaBur = (
                0.0000000004 * (medPond ** 2)
            ) - ( 0.0006 * medPond ) + 8.9401 # Polinômio para achar a nota

            notaClaBur = notaClaBur * 0.92

            # --------------------------------
            # --- SATURADOR NOTA CLASSIFICADOR
            # --------------------------------
            if notaClaBur > 10: # > Caso notaClabur > 0
                notaClaBur = 10
            elif notaClaBur < 0: # > Caso notaClaBur < 0
                notaClaBur = 0

            # ----------------------------------
            # --- CADASTRO BANCO DE DADOS GERDAU
            # ----------------------------------
            # Data = date.today() # > Coleta data
            # Hora = datetime.time(datetime.now()) # > Coleta Hora
            # m.Cadastrar(Data, Hora, notaClaBur) # > Cadastro no BD Gerdau

            # -------------------------------
            # --- APRESENTAR RESULTADO NA IHM
            # -------------------------------
            Img_Text = cv2.putText(
                Img.copy(),
                f'[RoI Esquerda] NBur: {NBurEsq} | Areas: {AreasEsq}',
                (17, 527),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                purpleColor,
                2,
                cv2.LINE_AA
            ) # Escreve na tela

            Img_Text = cv2.putText(
                Img_Text,
                f'[RoI Direita] NBur: {NBurDir} | Areas: {AreasDir}',
                (1280, 527),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                purpleColor,
                2,
                cv2.LINE_AA
            ) # Escreve na tela

            Img_Text = cv2.putText(
                Img_Text,
                f'NOTA: {round(notaClaBur, 9)}',
                (800, 500),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                purpleColor,
                2,
                cv2.LINE_AA
            ) # Escreve na tela

            Img_Text = cv2.rectangle(
                Img_Text,
                (r1[0][0],( int(r1[0][1] + ((r1[1][1] - r1[0][1]) * 0.2)))),
                (r1[1][0], r1[1][1]),
                purpleColor,
                4
            )

            Img_Text = cv2.rectangle(
                Img_Text,
                (r2[0][0],( int(r2[0][1] + ((r2[1][1] - r2[0][1]) * 0.2)))),
                (r2[1][0], r2[1][1]),
                purpleColor,
                4
            )

            # -------------------------------
            # --- MOSTRAR RESULTADOS NA TELA
            # -------------------------------
            img_cont = m.contourImg(
                contWantedLeft,
                r1[1][1],
                r1[1][0]
            )

            img_cont2 = m.contourImg(
                contWantedRight,
                r1[1][1],
                r1[1][0]
            )

            return [
                m.reduce_image(GrayImage, 1020),
                m.reduce_image(thres, 1020),
                m.reduce_image(Img, 1020),
                m.reduce_image(Img_Text, 1020),
                m.reduce_image(img_cont, 510),
                m.reduce_image(img_cont2, 510)
            ]

    return -1

# def nextImage():
#     global cont
#     cont += 1
#     listaImagens = processImage(cont)
