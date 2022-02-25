from platform import java_ver
import Methods as m # Chamando o modulo métodos

import numpy as np

import copy # Biblioteca para permitir a tarefa de copiar uma lista
import math # Biblioteca para cálculos matemáticos
import cv2 # Biblioteca interpretadora de imagens
# import matplotlib.pyplot as plt # Biblioteca para trabalhar com gráficos
#from scipy.fftpack import fft

from datetime import date
from datetime import time
from datetime import datetime

purpleColor = (113, 89 , 193)
redColor = (197, 48, 48)

def processImage(r1, r2, r3, AllImagens, k):

    # Copia duas Rois em uma variável
    roi_xld = (
        copy.deepcopy(r2),
        copy.deepcopy(r3)
    )

    grayImage = cv2.imread(AllImagens[k], 0)

    centerImage = grayImage[r1[0][1]:r1[1][1], r1[0][0]:r1[1][0]]

    roiTuplaHist = \
        cv2.calcHist(
            [centerImage],
            [0],
            None,
            [256],
            [0, 256]
        )

    histMax = max(roiTuplaHist)

    # Intensidade da luz branca na r1
    if histMax > 3750:
        # Tratamento de imagem: ==-==-==-==-==-==-==-==-==-==-==-==-==-==-==-==
        thres = cv2.threshold(
            grayImage,
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
            grayImage,
            cv2.COLOR_GRAY2BGR
        )

        ImgMedianPoint = Img.copy()

        NBurEsq = 0; NBurDir = 0; AreasEsq = 0; AreasDir = 0
        sizeLeft = 0; sizeRight = 0
        contWantedLeft = []; contWantedRight = []
        xLeft = 0
        xRight = 0


        for roi in roi_xld: # Repete o mesmo procedimento nas duas RoIs
            roi[0][1] = int(
                ((roi[1][1] - roi[0][1]) * 0.2) + roi[0][1]
            )

            reduced = m.reduce_domain(thres, roi) # Reduz domínio da imagem

            # Definindo valores: ==-==-==-==-==-==-==-==-==-==-==-==-==-==-==-==
            greater, son_list = m.findContour(reduced) # Contorno principal

            largest, ctr = m.unicLine(greater, roi) # Isola apenas uma linha

            if roi[0][0] == r2[0][0]: # Se a Roi em execução for a primeira
                NBurEsq = len(son_list) # Número de buracos na Roi esquerda
                AreasEsq = m.sumAreas(son_list) # Soma áreas de todos os buracos
            elif roi[0][0] == r3[0][0]: # Se a Roi em execução for a segunda
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

            #Pinta contornos
            Img_Contours = cv2.drawContours(
                Img.copy(),
                ctr,
                -1,
                (0, 255, 0),
                3
            )
            # Pinta buracos
            Fill_Contour = cv2.fillPoly(
                Img_Contours.copy(),
                son_list,
                purpleColor
            )

            Img = Fill_Contour.copy() # Img passa a ser a outra imagem editada

            if roi[0][0] == r2[0][0]: # Se a Roi em execução for a primeira
                sizeLeft = int(
                    math.sqrt(
                        ((end[0] - start[0]) ** 2) + ((end[1] - start[1]) ** 2)
                    )
                )
                # Coleta local de inicio e fim do contorno
                start, end, contWantedLeft = m.linearRegression(
                    ctr,
                    Img,
                    "Dir"
                )
                xLeft = ctr[0][0][0]
                contWantedLeft = m.fillWithZeros(contWantedLeft)

            elif roi[0][0] == r3[0][0]: # Se a Roi em execução for a segunda
                sizeRight = int(
                    math.sqrt(
                        ((end[0] - start[0]) ** 2) + ((end[1] - start[1]) ** 2)
                    )
                )
                
                # Coleta local de inicio e fim do contorno
                start, end, contWantedRight = m.linearRegression(
                    ctr,
                    Img,
                    "Esq"
                )
                
                xRight = ctr[len(ctr) - 1][0][0]
                contWantedRight = m.fillWithZeros(contWantedRight)

            # printa linha de angulação entre pontos finais e iniciais dos contornos
            cv2.line(
                Img,
                start,
                end,
                redColor,
                2
            )
        # Se o a distância do inicio e do final dos contornos for maior que 400px
        if sizeLeft > 400 and \
            sizeRight > 400 \
        :
            # ------------------------------------------
            # --- ESTIMAÇÃO DAS NOTAS PELO CLASSIFICADOR
            # ------------------------------------------
            medNuBur = (NBurEsq + NBurDir) / 2 # Média do número de buracos
            pesoBur = math.exp(-(medNuBur / 90)) # Peso do número de buracos
            medGeral = (AreasEsq + AreasDir) / 2 # Média geral da soma das áreas
            medPond = medGeral * pesoBur # Média ponderada
            
            notaClaBur = (
                -(0.000000000009122 * medPond ** 3)
            ) + ( 0.0000001731 * medPond ** 2) - (0.0015 * medPond) + 9.759 # Polinômio para achar a nota

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
            Img_Text = Img.copy()       #Apagar#########################<<<<<<<<<<<<

            # ESCREVE INFORMAÇÕES NA TELA
            Img_Text = cv2.putText(
                Img.copy(),
                f'[RoI Esquerda] NBur: {NBurEsq} | Areas: {AreasEsq}',
                (17, 527),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                purpleColor,
                2,
                cv2.LINE_AA
            )
            Img_Text = cv2.putText(
                Img_Text,
                f'[RoI Direita] NBur: {NBurDir} | Areas: {AreasDir}',
                (1280, 527),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                purpleColor,
                2,
                cv2.LINE_AA
            )
            # Classificador de buracos
            Img_Text = cv2.putText(
                Img_Text,
                f'NOTA: {round(notaClaBur, 9)}',
                (800, 700),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                purpleColor,
                2,
                cv2.LINE_AA
            )

            # Classificador espectral
            # Img_Text = cv2.putText(
            #     Img_Text,
            #     f'NOTA: {round(notaClaBur, 9)}',
            #     (800, 740),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     1,
            #     purpleColor,
            #     2,
            #     cv2.LINE_AA
            # )

            Img_Text = cv2.rectangle(
                Img_Text,
                (r1[0][0],( int(r1[0][1] + ((r1[1][1] - r1[0][1]) * 0.2)))),
                (r1[1][0], r1[1][1]),
                purpleColor,
                8
            )

            Img_Text = cv2.rectangle(
                Img_Text,
                (r2[0][0],( int(r2[0][1] + ((r2[1][1] - r2[0][1]) * 0.2)))),
                (r2[1][0], r2[1][1]),
                purpleColor,
                8
            )

            Img_Text = cv2.rectangle(
                Img_Text,
                (r3[0][0],( int(r3[0][1] + ((r3[1][1] - r3[0][1]) * 0.2)))),
                (r3[1][0], r3[1][1]),
                purpleColor,
                8
            )

            # -------------------------------
            # --- MOSTRAR RESULTADOS NA TELA
            # -------------------------------
            img_cont = m.contourImg(
                contWantedLeft,
                r2[1][1],
                r2[1][0]
            )

            # INDICA DISTÂNCIA DO CENTRO DA IMAGEM PARA O CENTRO DA BOBINA
            larg = int(ImgMedianPoint.shape[1] / 2)
            ImgMedianPoint = cv2.circle(ImgMedianPoint, (larg, 550), radius=8, color=(0, 0, 255), thickness=-1)

            ImgMedianPoint = cv2.line(ImgMedianPoint, (xLeft, 550), (xRight, 550), color=(255, 0, 255), thickness=1)
            ImgMedianPoint = cv2.circle(ImgMedianPoint, (xLeft, 550), radius=8, color=(255, 0, 255), thickness=-1)
            ImgMedianPoint = cv2.circle(ImgMedianPoint, (xRight, 550), radius=8, color=(255, 0, 255), thickness=-1)

            medianDot = int(((xRight - xLeft)/ 2) + xLeft)
            ImgMedianPoint = cv2.line(ImgMedianPoint, (medianDot, 550), (larg, 550), color=(0, 0, 255), thickness=4)
            ImgMedianPoint = cv2.circle(ImgMedianPoint, (medianDot, 550), radius=8, color=(255, 0, 255), thickness=-1)


             # TEXTO DA DISTÂNCIA DOS PONTOS
            distDots = medianDot - larg
            ImgMedianPoint = cv2.putText(
                ImgMedianPoint,
                f'Distancia: {distDots}',
                (larg - 100, 500),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2,
                cv2.LINE_AA
            )


            # INTRODUÇÃO A TRANSFORMADA DE FOURIER
            # xx = []
            # yy = []

            # for vet in contWantedLeft:
            #     xx.append(vet[0][0])
            #     yy.append(vet[0][1])
            #     # print("vet: ", vet)
            #     # print("x: ",vet[0][0], "y: ", vet[0][1])
            
            # fig, ax = plt.subplots()
            # N = len(yy)
            # T = 33/1000
            

            # yf = fft(yy, N)
            # wf = np.linspace(0.0, 1.0//(2.0*T), N//2)
            # ynf = 2.0/N * np.abs(yf[:N//2])

            # ax.plot(wf, ynf)
            # plt.show()

            img_cont2 = m.contourImg(
                contWantedRight,
                r2[1][1],
                r2[1][0]
            )
            return [
                m.reduce_image(grayImage, 1020),
                m.reduce_image(thres, 1020),
                m.reduce_image(ImgMedianPoint, 1020),
                m.reduce_image(Img_Text, 1020),
                m.reduce_image(img_cont, 510),
                m.reduce_image(img_cont2, 510),
                m.reduce_image(ImgMedianPoint, 1020)
            ]

    return -1
