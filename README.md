# Reconocimiento facial: Feature-extraction con ArcFace y Fine-tuning con SVM

*Feature extraction* consiste en transformar datos brutos en representaciones más compactas y relevantes que pueden ser utilizadas por algoritmos de inteligencia artificial para aprender, clasificar o reconocer patrones. 

En este proyecto se usa *ArcFace* para obtener los embeddings de un conjunto de imágenes; y se entrena un modelo clasificador (Suport Vector Machine - SVM) sobre estos embeddings para que realice predicciones sobre nuevas imágenes.

Tecnologías usadas en este proyecto:

* Insightface: Framework que usa *ArcFace* como función de pérdida para generar embeddings altamente discriminantes.
* opencv-python: Para cargar las imágenes del entorno de trabajo.
* numpy: Para operar las matrices de embeddings y etiquetas. 
* Scikit-learn: Para la importación de *Suport Vector Machine* (SVM) y codificación de etiquetas.
* onnxruntime: Motor de inferencia rápido para ejecutar modelos de deep learning en GPU o CPU.


## Instalación

Requiere Python 3.8+ y las siguientes bibliotecas indicadas en el archivo `requirementes.txt`:
```
numpy
opencv-python
scikit-learn
insightface
onnxruntime
```

## Configuración

1. Descarga o clona el proyecto en una terminal:

```sh
git clone https://github.com/GredUlloa/FineTuning-ArcFace-con-SVM
cd # copia la ruta del proyecto
``` 

2. Crea un entorno con la version 3.8 de python:

```cmd
py -3.8 -m venv venv
```

3. Activa el entorno creado:

```cmd
.\venv\Scripts\activate
# En macOS/Linux
# source venv/bin/activate
```

4. Instala las dependencias del proyecto:

```sh
pip install -r requirements.txt
```

## Modo de uso

### 1. Creacion de los datasets

En la ruta `\FaceRec\dataset` se encuentra por defecto los siguientes archivos que corresponden a fotos de una misma persona:

* `\Andres_C` (carpeta con 7 fotos)
* `Fototest2.jpg` (foto de prueba)

Para que el modelo SVM pueda realizar una predicción se necesitan al menos 2 datasets diferentes. Se debe crear otro dataset con el nombre de la otra persona y además una foto nueva que sirva como test de predicción:

* `\persona2` (carpeta con 7 o más fotos)
* `FotoTest1.jpg` (foto de prueba)

### 2. Ejecutar

En la terminal se ejecuta el código `Finetuning.py`:

```sh
python Finetuning.py
```

Finalmente en el entorno de trabajo se guardaran los siguientes archivos:

* `svm_insightface.pkl`: El modelo SVM entrenado con el dataset
* `label_encoder.pkl`: Las etiquetas codificadas para la predicción