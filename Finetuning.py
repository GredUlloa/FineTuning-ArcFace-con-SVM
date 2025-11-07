# %%
import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

## Implementando Resnet100 como modelo de extracción de características
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Ruta establecida del dataset
DATA_PATH = "dataset"
X, y = [], [] # Listas para características y etiquetas

## Extracción de embeddings
for person in os.listdir(DATA_PATH):
    person_path = os.path.join(DATA_PATH, person) # Establecer ruta de cada persona

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name) # Establecer ruta de cada imagen
        img = cv2.imread(img_path)
        faces = app.get(img)

        if len(faces) > 0:
            embedding = faces[0].embedding # Embedding del primer rostro detectado
            X.append(embedding) # Agregar embedding a la lista de características
            y.append(person)    # Agregar etiqueta correspondiente

X = np.array(X)
y = np.array(y)

# Imprime los datos extraídos (embeddings y etiquetas) / opcional
# print(X)
# print(y)

# Codificacion de etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)

## Entrenanmiento del clasificador SVM
clf = SVC(kernel='linear', probability=True)
clf.fit(X, y_encoded)

# Guardar modelo entrenado
joblib.dump(clf, "svm_insightface.pkl")
joblib.dump(le, "label_encoder.pkl")

# %%
## Test de predicción

def predict_image(img_path):
    img = cv2.imread(img_path)
    faces = app.get(img)
    if len(faces) == 0:
        return "No se detectó rostro", 0.0
    embedding = faces[0].embedding.reshape(1, -1)
    pred = clf.predict(embedding)[0] # Predicción del modelo SVM entrenado
    prob = clf.predict_proba(embedding).max()   # Probabilidad de la predicción

    return le.inverse_transform([pred])[0], prob

nombre, confianza = predict_image("dataset\FotoTest1.jpg")
print(f"Predicción: {nombre} (confianza: {confianza:.2f})")