"""ce fichier sert à transformer le son en spectrogramme de Mel, on répresente la fréquence en fonction du temps """
#importation des bibliothèques 
import librosa
import numpy as np
import scipy.io
import os

def load_bearing_data(file_path):
    """
    Charge le signal brut peu importe le format (Matlab, Wav ou Texte IMS).
    Retourne : (signal_audio, frequence_echantillonnage)
    """
    # Cas 1 : Dataset IMS (Fichiers textes ou sans extension avec colonnes)
    if file_path.endswith('.txt') or '.' not in os.path.basename(file_path):
        try:
            # On charge le fichier (séparateur tabulation pour IMS)
            data = np.loadtxt(file_path, delimiter='\t')
            # On prend la 1ère colonne (Capteur 1) et fréquence 20kHz
            return data[:, 0], 20000 
        except Exception as e:
            print(f"Erreur lecture TXT/IMS : {e}")
            
    # Cas 2 : Dataset CWRU (Fichiers Matlab .mat)
    elif file_path.endswith('.mat'):
        data = scipy.io.loadmat(file_path)
        for key in data.keys():
            # On cherche la clé qui contient les données temporelles (ex: 'DE_time')
            if 'time' in key:
                return data[key].flatten(), 12000 # CWRU est en 12kHz
                
    # Cas 3 : Fichiers Audio classiques (.wav)
    elif file_path.endswith('.wav'):
        y, sr = librosa.load(file_path, sr=None)
        return y, sr

      return None, None

def audio_to_melspectrogram_segments(file_path, segment_length=4096, n_mels=128):
    """
    Découpe le signal et transforme chaque morceau en spectrogramme de Mel.
    """
    signal, sr = load_bearing_data(file_path)
    
    if signal is None:
        return []

    segments = []
    
    # On parcourt le signal long par fenêtres de 'segment_length'
    # Plus segment_length est petit, plus on a d'images pour l'IA
    for i in range(0, len(signal) - segment_length, segment_length):
        chunk = signal[i:i + segment_length]
        
        # 1. Calcul du spectrogramme de Mel
        mel_spec = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=n_mels)
        
        # 2. Conversion en décibels
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 3. Normalisation Min-Max (0 à 1) pour la stabilité du CNN
        # Le 1e-6 évite la division par zéro si le signal est plat
        mel_db_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        
        segments.append(mel_db_norm)
        
    return segments
