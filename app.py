import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

st.title("Reconocimiento de Imágenes con Gestos")
image = Image.open("gestos.jpg")
st.image(image, width=360)

st.write(
    "En esta actividad podrás usar un modelo entrenado en **Teachable Machine** para reconocer gestos o movimientos. "
    "Captura una imagen con tu cámara y observa cómo el modelo identifica el gesto que realizas."
)

st.caption(f"Versión de Python: {platform.python_version()}")

with st.sidebar:
    st.subheader("Instrucciones")
    st.write(
        "1) Asegúrate de tener un modelo entrenado y guardado como `keras_model.h5`.\n"
        "2) Toma una foto con tu cámara.\n"
        "3) Observa la predicción del modelo y la probabilidad estimada."
    )

model = load_model("keras_model.h5")
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

img_file_buffer = st.camera_input("📸 Toma una foto para analizar")

if img_file_buffer is not None:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)
    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    st.subheader("Resultado de la predicción:")
    if prediction[0][0] > 0.5:
        st.success(f"✋ Gesto detectado: **Izquierda** (probabilidad: {prediction[0][0]:.3f})")
    elif prediction[0][1] > 0.5:
        st.success(f"👆 Gesto detectado: **Arriba** (probabilidad: {prediction[0][1]:.3f})")
    else:
        st.warning("No se detectó un gesto con suficiente confianza. Intenta nuevamente.")
