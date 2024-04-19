# Projet "Détection des sentiments dans les vidéos par IA" :  Jonathan, Laurence, Pierre & Rémi

Objectif :  L'objectif principal de Emobot est de développer une compréhension approfondie des émotions humaines en analysant les expressions faciales à travers des images ou des vidéos. Ce système utilise des technologies avancées de vision par ordinateur et d'intelligence artificielle pour identifier et classer les émotions des individus de manière précise et en temps réel.

* **La Présentation** : https://docs.google.com/presentation/d/1kxaBjUozHUTgMbr8JZmA9hVGgcq8N6HN3E-68FjXzKM/edit#slide=id.p
* **Le notebook** : https://github.com/JonathDev/PiLoJoR-/blob/main/emotion-detector-fer-2013.ipynb
* **Documentation** : [https://github.com/leabizbille/PiLoJoR-/blob/main/projet_emotionV4.pdf](https://github.com/JonathDev/PiLoJoR-/blob/main/projet_emotionV4.pdf)
    *  Le modele : https://github.com/JonathDev/PiLoJoR-/blob/main/Modele.py
    *  Entraînement d'un modèle de deep learning avec le dataset FER2013 : exemple https://github.com/JonathDev/PiLoJoR-/blob/main/15%20epochs.png
    *  Dataset :
      ![image](https://github.com/JonathDev/PiLoJoR-/assets/83597256/d0d133e6-fe4d-4bf5-9d33-60e66bcc75ef)

***OUTILS UTILISÉS***

**-TensorFlow** :

TensorFlow est une bibliothèque logicielle gratuite et open-source pour le flux de données et la programmation différentiable sur une gamme de tâches. C'est une bibliothèque mathématique symbolique, et est également utilisée pour des applications d'apprentissage automatique telles que les réseaux neuronaux. 

**-Keras** :

Keras est une bibliothèque de réseaux neuronaux open-source écrite en Python. Elle est capable de fonctionner sur TensorFlow. Conçue pour permettre une expérimentation rapide avec les réseaux neuronaux profonds, elle se concentre sur la convivialité, la modularité et l'extensibilité. 

**-OpenCV** :

OpenCV (Open Source Computer Vision Library) est une bibliothèque de fonctions de programmation principalement destinée à la vision par ordinateur en temps réel. Elle se concentre principalement sur le traitement d'images, la capture vidéo et l'analyse, y compris des fonctionnalités telles que la détection de visage. 

***MÉTHODOLOGIE***

1- Importer les libraries.
2- Importer les test set et validation set.
3- Utiliser le notebook pour créer le modele. 

Le modèle CNN est conçu et entraîné en utilisant Keras. Nous utilisons OpenCV pour la détection de visages en utilisant son classifieur de détection de visages pour dessiner des boîtes englobantes autour des visages détectés automatiquement. Après avoir développé le modèle facial, le réseau est entraîné et sauvegardé. Ensuite, le modèle entraîné est déployé via une interface. Une fois que le modèle de reconnaissance des émotions est entraîné, nous exécutons le script principal Python qui charge le modèle entraîné et les poids sauvegardés par lesquels finalement le modèle est appliqué à un flux vidéo en temps réel via une webcam.

Un modèle CNN séquentiel est utilisé dans ce projet. L'entrée passe d'abord par 4 blocs de convolution. Le nombre de filtres est progressivement augmenté, ce qui est le flux de travail général de diverses architectures de convolution. Dans chaque bloc, une convolution, une normalisation par lots, une fonction d'activation RELU (non-linéarité), un regroupement maximal et une régularisation par abandon sont appliqués sur les données. À chaque bloc de convolution, le volume est réduit d'un facteur de 2 tandis que le nombre de canaux double presque. La sortie est aplatie après le quatrième bloc de convolution, puis passe aux deux couches entièrement connectées. Enfin, la couche dense avec une activation Softmax est utilisée pour prédire l'étiquette de sortie qui correspond à l'une des sept émotions. L'optimiseur Adam est utilisé avec un taux d'apprentissage de 0,0005, ce qui accélère l'entraînement à environ 9 minutes par époque. La fonction model.summary() est utilisée pour afficher tous les paramètres que le modèle devra apprendre (environ 3 millions dans ce cas).

Plus d'information dans le rapport. 

4- Entrainement du modele.
Tout d'abord, nous sélectionnons le nombre d'époques à 15. Le nombre d'époques est un hyperparamètre de la descente de gradient qui contrôle le nombre de passes complètes à travers l'ensemble de données d'entraînement. Chaque époque prend environ 9 à 10 minutes. La première époque prend le plus de temps car l'allocation de ressources doit être effectuée pour le GPU, diverses bibliothèques doivent être chargées et des fichiers pour l'optimisation doivent également être chargés. Le temps total pris est d'environ 2,5 heures.

5- Sauvegarde du modele. 
6- Utilisation du modele. 
Le script main.py est exécuté et utilise les prédictions du modèle via une interface web.
La classe de la caméra envoie le flux d'images au modèle CNN pré-entraîné, puis récupère les prédictions du modèle et ajoute des étiquettes aux trames vidéo, puis finalement renvoie l'image à l'interface web. Le modèle peut être appliqué à des vidéos enregistrées ou en temps réel via une webcam.

***ENSEMBLE DE DONNÉES***

L'ensemble de données se compose d'images en niveaux de gris de 48x48 pixels de visages. Les visages ont été automatiquement enregistrés de sorte que le visage soit plus ou moins centré et occupe environ la même quantité d'espace dans chaque image. Chaque visage est basé sur l'émotion montrée dans l'expression faciale dans l'une des sept catégories (Colère, Dégoût, Peur, Joie, Tristesse, Surprise, Neutre).L'ensemble d'entraînement se compose de 28 708 images et l'ensemble de test se compose de 7 178 images.

Cet ensemble de données a été préparé par Pierre-Luc Carrier et Aaron Courville pour le défi de reconnaissance des expressions faciales 2013 (Kaggle).
