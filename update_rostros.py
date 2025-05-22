import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Lista de sujet
subjects = ["", "Aaron", "Christopher", "Heath", "Gary"]


def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue
            
        label = int(dir_name.replace("s", ""))
        subject_dir_path = os.path.join(data_folder_path, dir_name)
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue
                
            image_path = os.path.join(subject_dir_path, image_name)
            image = cv2.imread(image_path)
            
            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)
                
    return faces, labels

# Función para detectar rostros
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

# Función para dibujar rectángulo y texto
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

# Función para predecir y evaluar
# Función para predecir y evaluar (versión modificada)
def predict_and_evaluate(test_data_path):
    IMAGENES_POR_SUJETO = 3
    true_labels = []
    predicted_labels = []
    test_images = []
    subjects_displayed = {}

    # Inicializar contadores
    for i in range(1, len(subjects)):
        subjects_displayed[i] = 0

    for dir_name in os.listdir(test_data_path):
        if not dir_name.startswith("s"):
            continue
            
        true_label = int(dir_name.replace("s", ""))
        if subjects_displayed.get(true_label, 0) >= IMAGENES_POR_SUJETO:  # Límite ajustable
            continue
            
        subject_dir_path = os.path.join(test_data_path, dir_name)
        
        for image_name in os.listdir(subject_dir_path):
            if subjects_displayed.get(true_label, 0) >= IMAGENES_POR_SUJETO:  # Límite aquí
                break
                
            if image_name.startswith("."):
                continue
                
            image_path = os.path.join(subject_dir_path, image_name)
            test_img = cv2.imread(image_path)
            if test_img is None:
                continue
                
            face, rect = detect_face(test_img)
            if face is not None:
                predicted_label, confidence = face_recognizer.predict(face)
                img_with_prediction = test_img.copy()
                draw_rectangle(img_with_prediction, rect)
                draw_text(img_with_prediction, subjects[predicted_label], rect[0], rect[1]-5)
                test_images.append(img_with_prediction)
                true_labels.append(true_label)
                predicted_labels.append(predicted_label)
                subjects_displayed[true_label] = subjects_displayed.get(true_label, 0) + 1

    # Ajustar diseño de la visualización
    plt.figure(figsize=(15, 5 * len(subjects[1:])))
    for i, subject_id in enumerate(sorted(subjects_displayed.keys())):
        subject_images = [img for idx, img in enumerate(test_images) if true_labels[idx] == subject_id]
        
        for j in range(min(IMAGENES_POR_SUJETO, len(subject_images))):  # Usar la variable aquí
            img_rgb = cv2.cvtColor(subject_images[j], cv2.COLOR_BGR2RGB)
            plt.subplot(len(subjects[1:]), IMAGENES_POR_SUJETO, i*IMAGENES_POR_SUJETO + j + 1)  # Ajuste en la cuadrícula
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.title(f"{subjects[subject_id]}: Pred {subjects[predicted_labels[i*IMAGENES_POR_SUJETO + j]]}")
    
    plt.tight_layout()
    plt.show()

    # Resto del código (métricas) permanece igual...

    # Resto del código de métricas (sin cambios)
    if len(true_labels) > 0:
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f_score = f1_score(true_labels, predicted_labels, average='weighted')
        accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels))
        
        cm = confusion_matrix(true_labels, predicted_labels, labels=range(1, len(subjects)))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=subjects[1:], yticklabels=subjects[1:])
        plt.xlabel('Predicho')
        plt.ylabel('Verdadero')
        plt.title('Matriz de Confusión')
        plt.show()
        
        print("\n--- Métricas de Evaluación ---")
        print(f"Exactitud (accuracy): {accuracy * 100:.2f}%")
        print(f"Precisión (precision): {precision:.2f}")
        print(f"Exhaustividad (recall): {recall:.2f}")
        print(f"F-score: {f_score:.2f}")
    else:
        print("No se detectaron rostros en las imágenes de prueba.")
# --- Entrenamiento del modelo ---
print("Preparando datos de entrenamiento...")
faces, labels = prepare_training_data("./traindata")
print(f"Datos preparados: {len(faces)} rostros, {len(labels)} etiquetas")

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

# --- Evaluar y mostrar resultados ---
print("\nEvaluando modelo y mostrando predicciones...")
predict_and_evaluate("./testdata")