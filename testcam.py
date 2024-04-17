import cv2
# Test de la webcam à l'indice 0
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Webcam accessible")
    cap.release()
else:
    print("Erreur d'accès à la webcam")