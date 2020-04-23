# This algorithm was developed by Luis Navarrete
# Current student of Biomedical Engineering at Instituto Tecnológico de Mérida
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 20:36:17 2020

@author: DELL
"""
import tkinter

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
#from matplotlib.backend_bases import key_press_handler
#from matplotlib.figure import Figure
import skimage.io as io
from skimage.transform import resize
#import time
#from skimage.io import imread
#import os
#from skimage.transform import rotate
import numpy as np
import pandas as pd
#import re
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
#from keras.utils import np_utils
#from keras.preprocessing.image import ImageDataGenerator
#from keras.callbacks import ReduceLROnPlateau
#from keras.layers import AveragePooling2D
#from keras.optimizers import SGD
##################
#from keras.models import Model
##########################################
#leo el dataset
data = pd.read_csv('metadata_ordenado.csv')
etiquetas = data['dx']
image_id = data['image_id']
longitud = iter(np.arange(len(etiquetas)).tolist())
aleatorios = np.arange(len(etiquetas)).tolist()
np.random.shuffle(aleatorios)
alea = iter(aleatorios)
########################################
#### Defino la estructura de la red neuronal
def DevDay_LeNet5(num_classes):
    """CNN model based on LeNet-5."""
    
    # Create model:
    model = Sequential()
    # Add Conv2D -> 6, (5, 5), input_shape=(32, 32, 1), activation='relu', strides = (1,1)
    model.add(Conv2D(6, (3,3), input_shape = (32,32,3), activation='relu', strides = (1,1))) #stride (1,1)
    # Add MaxPooling2D -> pool_size=(2, 2)
    #model.add(AveragePooling2D(pool_size=(2,2), strides = (2,2)))
    model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
    
    # Add Conv2D -> 16, (3, 3), activation='relu'
    model.add(Conv2D(16, (3,3), activation='relu', strides = (1,1))) #stride (1,1)
    # Add MaxPooling2D -> pool_size=(2, 2)
    #model.add(AveragePooling2D(pool_size=(2,2), strides = (2,2)))
    model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
    
    
   
    #model.add(Flatten())
    #model.add(Dropout(0.25))  # Let's try to avoid overfitting... #Elimina el 20 porciento de la conexiones sinapticas para evitar el overfitting
    ## Add Dense -> 128, activation='relu'
    #model.add(Dense(120, activation='relu'))
    model.add(Conv2D(120,(5,5), strides=(1,1), activation = 'relu'))
    model.add(Flatten())
    # Add Dense -> 50, activation='relu'
    model.add(Dropout(0.2))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    #opt = SGD(lr=0.01, momentum=0.9)
    # Compile model:
    model.compile(loss='categorical_crossentropy', optimizer= 'adam', 
              metrics=['accuracy'])
    return model
num_classes = 7
# Build the model:
model = DevDay_LeNet5(num_classes)
model.load_weights('modelo_2_dic_2019.h5')

########################################################
lista = {"akiec":0, "bcc":1, "bkl":2, "df":3, "mel":4, "nv":5, "vasc":6}
########################################################
path = "C:/Users/DELL/Desktop/Curso machine learning python/datasets/imagenes HAM"
primera = "ISIC_0024306.jpg"
def imread_convert(f):
    original = io.imread(f)
    return resize(original, (32,32,3)) #correr en el ambiente tensorflow_gpuenv
def convert2(f):
    original2 = io.imread(f)
    return original2

ic = io.ImageCollection(path + '/*jpg', load_func=imread_convert)
ic_or = io.ImageCollection(path + '/*jpg', load_func=convert2)
#ic.files  #para obtener las direcciones
def load_next_img():
    #nn = next(longitud) #Si quiero sin forma aleatoria
    n_ran = next(alea)
    #imagen = ic[nn] #Esto debe de tener botones que cambien con el indicador
    imagen = ic[n_ran]
    #ima_or = ic_or[nn]
    pred_image = imagen.reshape(1,32,32,3)
    prediccion = model.predict(pred_image)
    argmax_pred = np.argmax(prediccion)
    final_pred = list(lista.keys())[list(lista.values()).index(argmax_pred)]
    probabilidad = prediccion.tolist()[0]
    ax.clear()
    #ax.imshow(ic[nn]), ax.grid(False)
    #ax.imshow(ic_or[nn]), ax.grid(False)
    ax.imshow(ic_or[n_ran]), ax.grid(False)
    #ax.set_title(image_id[nn]) #sin forma aleatoria
    ax.set_title(image_id[n_ran])
    ax.set_xticks([])
    ax.set_yticks([])
    PlotCanvas.draw()
    valor = probabilidad[argmax_pred]*100
    #Lc.config(text=etiquetas[nn]) #Sin forma aleatoris
    Lc.config(text=etiquetas[n_ran])
    Lc2.config(text=final_pred)
    Lc3.config(text = str(round(valor, 2))) # This algorithm was developed by Luis Navarrete                                  
    #print('Indicador', nn)
    print('Algorithm developed by Luis Navarrete')
    print('Biomedical Engineer student at Instituto Tecnológico de Mérida')
    #print('Real', etiquetas[nn]) #debe de ir con el número del indicador       # Current student of Biomedical Engineering at Instituto Tecnológico de Mérida
    #print('Predicción', final_pred)
    #print('Probabilidad', probabilidad[argmax_pred]*100)
    #fig = Figure(figsize=(5, 4), dpi=100)
    #fig.add_subplot(111).imshow(ic[next(longitud)])
    #return ic[next(longitud)
    #plt.show()
# imagen = ic[20] #Esto debe de tener botones que cambien con el indicador
# pred_image = imagen.reshape(1,32,32,3)
# prediccion = model.predict(pred_image)
# argmax_pred = np.argmax(prediccion)
# final_pred = list(lista.keys())[list(lista.values()).index(argmax_pred)]
# probabilidad = prediccion.tolist()[0]
# print('Real', etiquetas[20]) #debe de ir con el número del indicador
# print('Predicción', final_pred)
# print('Probabilidad', probabilidad[argmax_pred]*100)
root = tkinter.Tk()
root.geometry("1080x720")
root.wm_title("PigSkin v. 0.1 by LN")
#def load_next_img: 
#    imagen = ic[next(len(data))]
figure = plt.Figure(figsize=(5, 4), dpi=100)
ax = figure.add_subplot(111)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
#fig.add_subplot(111).imshow(ic[load_next_img()])
#fig, ax = plt.figure(kwargs)(111)

#fig.add_subplot(111).imshow(imagen)
#### frames ################33
Plot_Frame = tkinter.Frame(root, bg='white', relief=tkinter.SUNKEN, bd=3)
Widgets_Frame = tkinter.Frame(root)
botones = tkinter.Frame(root)
##################
PlotCanvas = FigureCanvasTkAgg(figure, Plot_Frame)  # A tk.DrawingArea.
PlotCanvas.get_tk_widget().pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=1)


#toolbar = NavigationToolbar2Tk(PlotCanvas, root)
#toolbar.update()
#PlotCanvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
# Lab = tkinter.Label(salidas,text='Predicción: ')
#Lab.grid(row=0,column=0)
#Lc = tkinter.Label(salidas,text=' ',bg='white',relief=tkinter.SUNKEN)
#Lc.grid(row=0,column=1,sticky="ew")

# load_next_img()

#def on_key_press(event):
#    print("you pressed {}".format(event.key))
#    key_press_handler(event, PlotCanvas, toolbar)


#PlotCanvas.mpl_connect("key_press_event", on_key_press)


def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

# Lab = tkinter.Label(salidas,text='Predicción: ')
# Lab.grid(row=0, column=0)
# Lc = tkinter.Label(salidas,text=' ',bg='white',relief=tkinter.SUNKEN)
# Lc.grid(row=0,column=0,sticky="ew")
#Plot_Frame = tkinter.Frame(root, bg='white', relief=tkinter.SUNKEN, bd=3)
#load_next_img()
#Lab = tkinter.Label(Widgets_Frame,text='a + b = ')
#Lab.grid(row=3,column=0)
#Lc = tkinter.Label(Widgets_Frame,text='hola',bg='white',relief=tkinter.SUNKEN)
#Lc.grid(row=3,column=1,sticky="ew")
   
button = tkinter.Button(master=botones, text="Quit", command=_quit)
# button.pack(side=tkinter.BOTTOM)
#button.grid(row=1,column=0, columnspan=3, sticky='ew',pady=5)
nextbutton = tkinter.Button(master=botones, text="next", command=load_next_img)
#nextbutton.grid(row=1,column=1, columnspan=3, sticky='ew',pady=5)
#nextbutton.pack(side=tkinter.BOTTOM)  
Lab = tkinter.Label(Widgets_Frame,text='Valor real: ')
#Lab.grid(row=1,column=2)
Lc = tkinter.Label(Widgets_Frame,text=' ',bg='white',relief=tkinter.SUNKEN)
#Lc.grid(row=1,column=3,sticky="ew")
Lab2 = tkinter.Label(Widgets_Frame,text='Predicción: ')
Lc2 = tkinter.Label(Widgets_Frame,text=' ',bg='white',relief=tkinter.SUNKEN)
Lab3 = tkinter.Label(Widgets_Frame,text='Probabilidad: ')
Lc3 = tkinter.Label(Widgets_Frame,text=' ',bg='white',relief=tkinter.SUNKEN)

load_next_img()
    
RH = 0.90 # Altura Relativa
Plot_Frame.place(relx=0,relwidth=0.75, relheight=1)
# salidas.place(relx=0.35,rely = RH,relwidth=0.65, relheight=1-RH)
# Lc.pack(side=tkinter.RIGHT,fill=tkinter.BOTH,expand=1)
Widgets_Frame.place(relx=0.75,relwidth=0.25, relheight=0.90)
Lab.pack(side=tkinter.TOP,fill=tkinter.BOTH,expand=1)
Lc.pack(side=tkinter.TOP,fill=tkinter.BOTH,expand=1)
Lab2.pack(side=tkinter.TOP,fill=tkinter.BOTH,expand=1)
Lc2.pack(side=tkinter.TOP,fill=tkinter.BOTH,expand=1)
Lab3.pack(side=tkinter.TOP,fill=tkinter.BOTH,expand=1)
Lc3.pack(side=tkinter.TOP,fill=tkinter.BOTH,expand=1)
#button.pack(side=tkinter.BOTTOM,fill=tkinter.BOTH,expand=1)
#nextbutton.pack(side=tkinter.BOTTOM,fill=tkinter.BOTH,expand=1)

botones.place(rely=RH,relwidth=1, relheight=1-RH)
button.pack(side=tkinter.BOTTOM,fill=tkinter.BOTH,expand=1)
nextbutton.pack(side=tkinter.BOTTOM,fill=tkinter.BOTH,expand=1)

tkinter.mainloop()
# If you put root.destroy() here, it will cause an error if the window is
# closed with the window manager.

# This algorithm was developed by Luis Navarrete
# Current student of Biomedical Engineering at Instituto Tecnológico de Mérida