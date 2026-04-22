from src.model import build_model
#On importe la structure depuis ton fichier model.py 
import os 
import cv2
from src.preprocess import audio_to_melspectrogram_segments
from tensorflow.keras.utilis import to_categorical
from sklearn.model_selection import train_test_split 

# configurer les path 
DATA_PATH = "data/raw/"
IMG_SIZE = (128,128)

X,y = [],[]
classes = [ d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH,d))]

print(f"Classes détectées :{classes}")
for i, label in enumerate(classes):
    folder = os.path.join(DATA_PATH, label)
    for file in os.listdir(folder):
        if file.endswith(('.mat', '.wav')):
            file_path = os.path.join(folder, file)
            # On récupère tous les morceaux découpés
            segments = audio_to_melspectrogram_segments(file_path)
            for seg in segments:
                X.append(cv2.resize(seg, IMG_SIZE))
                y.append(i)

X = np.array(X).reshape(-1, 128, 128, 1)
y = to_categorical(np.array(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Construction et entraînement
model = build_model(input_shape=(128, 128, 1), num_classes=len(classes))
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# SAUVEGARDE FINALE
model.save('bearing_model.h5')
print("Modèle 'bearing_model.h5' généré avec succès !")
