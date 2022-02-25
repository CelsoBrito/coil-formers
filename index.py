import Process as pro
import Methods as m
# import cv2 #Usar quando for capiturar imagens pela camera

from PIL import (Image, ImageTk)
from tkinter import (
    Tk,
    Label,
    IntVar,
    TOP,
    Radiobutton,
    BOTTOM,
    LEFT,
    RIGHT,
    mainloop,
    font as tkFont
)

colorDarkPurple='#191622'
colorPurple='#7159c1'
colorLightPurple='#E1E1E6'
colorLightYellow='#f8f8f2'
colorGreen='#50fa7b'

print(f'NBE: Nº Buracos Esquerda')
print(f'NBD: Nº Buracos Direita')
print(f'AE: Áreas Esquerda')
print(f'AD: Áreas Direita')
print(f'N: NOTA')

imgtk = 0; panelA = 0; button = 0; cont = 0

def processa():
    return pro.processImage(
        r1,
        r2,
        r3,
        AllImagens,
        cont
    )

def makeTK():
    r = Tk() # Janela Gráfica
    r.configure(bg=colorDarkPurple)
    r.title(f'{AllImagens[cont]} [{cont + 1}/{total}]')
    return r

def tela(r, img):
    global imgtk, panelA, button

    im = Image.fromarray(img[0])
    imgtk = ImageTk.PhotoImage(image=im)
    
    panelA = Label(
        image=imgtk,
        highlightthickness = 2,
        highlightbackground=colorPurple,
        bd= 0
    )

    panelA.pack(
        side=TOP,
        padx=10,
        pady=10
    )

    intVar = IntVar()

    helv11 = tkFont.Font(family='Helvetica', size=11, weight=tkFont.BOLD)
    
    Radiobutton(
        r,
        text='Manual',
        variable=intVar,
        value=0,
        selectcolor=colorDarkPurple,
        fg=colorLightPurple,
        activeforeground=colorPurple,
        bg=colorDarkPurple,
        activebackground=colorDarkPurple,
        borderwidth = 0,
        highlightthickness = 0,
        font=helv11,
    ).pack(side=BOTTOM, padx=5)

    Radiobutton(
        r,
        text='Automático',
        variable=intVar,
        value=1,
        selectcolor=colorDarkPurple,
        fg=colorLightPurple,
        activeforeground=colorPurple,
        bg=colorDarkPurple,
        activebackground=colorDarkPurple,
        borderwidth = 0,
        highlightthickness = 0,
        font=helv11,
    ).pack(side=BOTTOM, padx=5)

    imLEFT = Image.fromarray(img[4])
    imgtkLEFT = ImageTk.PhotoImage(image=imLEFT)
    
    panelLEFT = Label(
        image=imgtkLEFT,
        highlightthickness = 2,
        highlightbackground=colorPurple,
        bd= 0
    )
    panelLEFT.pack(
        side=LEFT,
        padx=10,
        pady=10
    )

    imLEFT2 = Image.fromarray(img[5])
    imgtkLEFT2 = ImageTk.PhotoImage(image=imLEFT2)
    
    panelLEFT2 = Label(
        image=imgtkLEFT2,
        highlightthickness = 2,
        highlightbackground=colorPurple,
        bd= 0
    )
    panelLEFT2.pack(
        side=LEFT,
        padx=10,
        pady=10
    )

    def getScript():
        im = Image.fromarray(
            img[var.get()]
        )

        imgtk2 = ImageTk.PhotoImage(image=im)
        
        panelA.configure(image=imgtk2)
        panelA.image = imgtk2

    def muda(img2):
        im = Image.fromarray(
            img2[var.get()]
        )

        imgtk2 = ImageTk.PhotoImage(image=im)
        
        panelA.configure(image=imgtk2)
        panelA.image = imgtk2
        
        imLEFT = Image.fromarray(img2[4])
        imgtkLEFT = ImageTk.PhotoImage(image=imLEFT)
        
        panelLEFT.configure(image=imgtkLEFT)
        panelLEFT.image = imgtkLEFT
        
        imLEFT2 = Image.fromarray(img2[5])
        imgtkLEFT2 = ImageTk.PhotoImage(image=imLEFT2)
        
        panelLEFT2.configure(image=imgtkLEFT2)
        panelLEFT2.image = imgtkLEFT2

    def nextImage():
        global cont

        cont += 1
        img2 = processa()
        
        while(img2 == -1):
            cont += 1
            img2 = processa()
        muda(img2)
        
    def proxima():
        if (intVar.get() == 1 and cont + 1 < total):
            nextImage()
            r.title(f'{AllImagens[cont]} [{cont + 1}/{total}]')
            r.after(10, proxima) # Atualiza a tela a cada 0.01 segundo

    def key(event):
        if(intVar.get() == 0 and cont + 1 < total):
            nextImage()

            r.title(f'{AllImagens[cont]} [{cont + 1}/{total}]')
        elif(intVar.get() == 1):
            proxima()

    r.bind("<Return>", key)

    var = IntVar()
    
    Radiobutton(
        r,
        text = 'Original',
        variable = var,
        value = 0,
        selectcolor=colorDarkPurple,
        fg=colorLightPurple,
        activeforeground=colorPurple,
        bg=colorDarkPurple,
        activebackground=colorDarkPurple,
        borderwidth = 0,
        highlightthickness = 0,
        font=helv11,
        command=lambda: getScript()
    ).pack(side=RIGHT, padx=5)

    Radiobutton(
        r,
        text='Limiar',
        variable=var,
        value=1,
        selectcolor=colorDarkPurple,
        fg=colorLightPurple,
        activeforeground=colorPurple,
        bg=colorDarkPurple,
        activebackground=colorDarkPurple,
        borderwidth = 0,
        highlightthickness = 0,
        font=helv11,
        command=lambda:getScript()
    ).pack(side=RIGHT, padx=5)

    Radiobutton(
        r,
        text='Contornos',
        variable=var,
        value=2,
        selectcolor=colorDarkPurple,
        fg=colorLightPurple,
        activeforeground=colorPurple,
        bg=colorDarkPurple,
        activebackground=colorDarkPurple,
        borderwidth = 0,
        highlightthickness = 0,
        font=helv11,
        command=lambda:getScript()
    ).pack(side=RIGHT, padx=5)

    Radiobutton(
        r,
        text='Dados',
        variable=var,
        value=3,
        selectcolor=colorDarkPurple,
        fg=colorLightPurple,
        activeforeground=colorPurple,
        bg=colorDarkPurple,
        activebackground=colorDarkPurple,
        borderwidth = 0,
        highlightthickness = 0,
        font=helv11,
        command=lambda:getScript()
    ).pack(side=RIGHT, padx=5)
    mainloop()

# -------------------------------------
# --- ORIGEM DA IMAGEM
# -------------------------------------
# > CAMERA:
# ImgCam = cv2.VideoCapture('rtsp://admin:admin1234@10.45.106.51')
# ImgWth = int(ImgCam.get(3))
# ImgHht = int(ImgCam.get(4))

# > BANCO DE DADOS DE IMAGENS:
AddBDImg = 'ImagensFM/*.png'
AllImagens = m.importImages(AddBDImg) # Identifica conjunto de imagens em um diretório
total = len(AllImagens) # Quantidade de imagens achadas
# -------------------------------------
# --- DEFININDO AS REGIÕES DE INTERESSE
# -------------------------------------
# Variáveis com dados de coordenadas [(xi, yi), (xf, yf)]:
# xi = (x inicial); xf = (x final); yi = (y inicial); yf = (y final)
r1 = [[850, 150],[1250, 400]]   # Coordenads ROI Central
r2 = [[17, 390], [527, 1188]] # Coordenadas ROI Esq
r3 = [[1337, 390], [1917, 1188]] # Coordenadas ROI Dir

imgs = -1
while (cont < total - 1):
    imgs = processa()
    if(imgs != -1):
        break

    cont += 1

if( imgs != -1 ):
    root = makeTK()
    tela(root, imgs)
else:
    print('Não há imagens bem padronizadas.')
