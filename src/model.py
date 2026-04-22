""" le modèl de l'IA """
import tensorflow as tf 
from tensorflow.keras import layers, models

def build_model(input_shape,num_classes):
    #création du réseau de neurones convolutif(CNN)
    model = models.Sequentiel([#Première couche de convolution pour capter les motifs fréquenciels 
        layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape),layers.MaxPooling2D((2,2)),
        #Deuxième couche pour les details plus fins 
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        #couche finale de convolution 
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        #sortie 1 neurone par classe avec probabilité(softmax)
            layers.Dense(4,activation='softmax')])    
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return model 
