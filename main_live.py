import tensorflow as tf
import numpy as np
import cv2
import librosa
from src.preprocess import load_bearing_data

# 1. Charger le modèle et définir les classes (doit être le même ordre que train_main.py)
model = tf.keras.models.load_model('bearing_model.h5')
classes = ['Normal', 'Panne_Bille', 'Panne_Bague'] # À adapter selon tes dossiers

def predict_realtime(file_path):
    # On prend juste le premier segment pour la démo
    signal, sr = load_bearing_data(file_path)
    chunk = signal[:4096] 
    
    mel_spec = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_db_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    
    input_img = cv2.resize(mel_db_norm, (128, 128)).reshape(1, 128, 128, 1)
    
    prediction = model.predict(input_img)
    print(f"Résultat : {classes[np.argmax(prediction)]} ({np.max(prediction)*100:.2f}%)")
    #il faut appeler la fonction grace au path du fichier 
