import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Charger le modèle pré-entraîné pour la détection de sentiments
model = load_model("model_optimal.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Dictionnaire pour mapper les indices de classes aux noms des sentiments
emotion_dict = {0: "En colère", 1: "Dégoûté", 2: "Peur", 3: "Joyeux", 4: "Triste", 5: "Surpris", 6: "Neutre"}

# Fonction pour détecter les visages et prédire les émotions
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        image_pixels = img_to_array(roi_gray)
        image_pixels = np.expand_dims(image_pixels, axis=0)
        image_pixels /= 255

        predictions = model.predict(image_pixels)
        max_index = np.argmax(predictions[0])

        emotion = emotion_dict[max_index]
        cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame

# Fonction pour mettre à jour l'image sur l'interface Tkinter
def update_image():
    ret, frame = cap.read()
    if not ret:
        print("Erreur: Impossible de lire la vidéo depuis la webcam.")
        return
    frame = cv2.flip(frame, 1)
    frame = detect_emotion(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, update_image)  # Mettre à jour toutes les 10 ms

# Créer une fenêtre Tkinter
root = tk.Tk()
root.title("Détection de sentiments en direct via la webcam")

# Démarrer la capture vidéo à partir de la webcam
cap = cv2.VideoCapture(1)

# Vérifier si la capture vidéo est réussie
if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la webcam.")
    exit()

# Créer une étiquette pour afficher l'image
label = tk.Label(root)
label.pack()

# Mettre à jour l'image sur l'étiquette
update_image()

# Lancer la boucle principale Tkinter
root.mainloop()

# Fermer la capture vidéo lorsque la fenêtre est fermée
cap.release()
cv2.destroyAllWindows()
