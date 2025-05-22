import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Lista de sujetos (ajusta según tus carpetas)
subjects = ["", "Aaron", "Mike", "Joe", "Brad"]

# Función para detectar rostros (sin cambios)
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

# Función para dibujar rectángulo y texto (sin cambios)
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

# Función para predecir y mostrar imágenes (modificada para evaluación)
def predict_and_evaluate(test_data_path):
    true_labels = []
    predicted_labels = []
    test_images = []  # Almacenar imágenes para mostrar después

