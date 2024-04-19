# Projet "Détection des sentiments dans les vidéos par IA" :  Jonathan, Laurence, Pierre & Rémi

Objectif :  L'objectif principal de Emobot est de développer une compréhension approfondie des émotions humaines en analysant les expressions faciales à travers des images ou des vidéos. Ce système utilise des technologies avancées de vision par ordinateur et d'intelligence artificielle pour identifier et classer les émotions des individus de manière précise et en temps réel.

* **La Présentation** : https://docs.google.com/presentation/d/1kxaBjUozHUTgMbr8JZmA9hVGgcq8N6HN3E-68FjXzKM/edit#slide=id.p
* **Le notebook** : https://github.com/JonathDev/PiLoJoR-/blob/main/emotion-detector-fer-2013.ipynb
* **Documentation** : [https://github.com/leabizbille/PiLoJoR-/blob/main/projet_emotionV4.pdf](https://github.com/JonathDev/PiLoJoR-/blob/main/projet_emotionV4.pdf)

* **Détection des émotions sur un flux Webcam** :
    *  Le modele : https://github.com/JonathDev/PiLoJoR-/blob/main/Modele.py
    *  Entraînement d'un modèle de deep learning avec le dataset FER2013 : exemple https://github.com/JonathDev/PiLoJoR-/blob/main/15%20epochs.png
    *  Utilisation de la webcam comme outil de capture.
    *  Dataset :
      ![image](https://github.com/JonathDev/PiLoJoR-/assets/83597256/d0d133e6-fe4d-4bf5-9d33-60e66bcc75ef)

OUTILS UTILISÉS

**TensorFlow** :

TensorFlow est une bibliothèque logicielle gratuite et open-source pour le flux de données et la programmation différentiable sur une gamme de tâches. C'est une bibliothèque mathématique symbolique, et est également utilisée pour des applications d'apprentissage automatique telles que les réseaux neuronaux. 

**Keras**:

Keras est une bibliothèque de réseaux neuronaux open-source écrite en Python. Elle est capable de fonctionner sur TensorFlow. Conçue pour permettre une expérimentation rapide avec les réseaux neuronaux profonds, elle se concentre sur la convivialité, la modularité et l'extensibilité. 

**OpenCV**:

OpenCV (Open Source Computer Vision Library) est une bibliothèque de fonctions de programmation principalement destinée à la vision par ordinateur en temps réel. Elle se concentre principalement sur le traitement d'images, la capture vidéo et l'analyse, y compris des fonctionnalités telles que la détection de visage. 

MÉTHODOLOGIE

Le modèle CNN est conçu et entraîné en utilisant Keras. Nous utilisons OpenCV pour la détection de visages en utilisant son classifieur de détection de visages pour dessiner des boîtes englobantes autour des visages détectés automatiquement. Après avoir développé le modèle facial, le réseau est entraîné et sauvegardé. Ensuite, le modèle entraîné est déployé via une interface. Une fois que le modèle de reconnaissance des émotions est entraîné, nous exécutons le script principal Python qui charge le modèle entraîné et les poids sauvegardés par lesquels finalement le modèle est appliqué à un flux vidéo en temps réel via une webcam.

ENSEMBLE DE DONNÉES

L'ensemble de données se compose d'images en niveaux de gris de 48x48 pixels de visages. Les visages ont été automatiquement enregistrés de sorte que le visage soit plus ou moins centré et occupe environ la même quantité d'espace dans chaque image. Chaque visage est basé sur l'émotion montrée dans l'expression faciale dans l'une des sept catégories (Colère, Dégoût, Peur, Joie, Tristesse, Surprise, Neutre).L'ensemble d'entraînement se compose de 28 708 images et l'ensemble de test se compose de 7 178 images.

Cet ensemble de données a été préparé par Pierre-Luc Carrier et Aaron Courville pour le défi de reconnaissance des expressions faciales 2013 (Kaggle).
