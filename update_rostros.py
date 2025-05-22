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

   # Procesar imágenes de prueba
    for dir_name in os.listdir(test_data_path):
        if not dir_name.startswith("s"):
            continue
        true_label = int(dir_name.replace("s", ""))
        subject_dir_path = os.path.join(test_data_path, dir_name)
        
        for image_name in os.listdir(subject_dir_path):
            if image_name.startswith("."):
                continue
            image_path = os.path.join(subject_dir_path, image_name)
            test_img = cv2.imread(image_path)
            if test_img is None:
                continue
            
            # Detectar rostro y predecir
            face, rect = detect_face(test_img)
            if face is not None:
                predicted_label, confidence = face_recognizer.predict(face)
                
                # Dibujar rectángulo y texto en la imagen
                img_with_prediction = test_img.copy()
                draw_rectangle(img_with_prediction, rect)
                draw_text(img_with_prediction, subjects[predicted_label], rect[0], rect[1]-5)
                test_images.append(img_with_prediction)  # Guardar para mostrar
                
                true_labels.append(true_label)
                predicted_labels.append(predicted_label)

    # Mostrar imágenes con predicciones
    plt.figure(figsize=(15, 10))
    for i, img in enumerate(test_images[:min(6, len(test_images))]):  # Mostrar hasta 6 imágenes
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 3, i+1)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"Predicción: {subjects[predicted_labels[i]]}")
    plt.tight_layout()
    plt.show()
