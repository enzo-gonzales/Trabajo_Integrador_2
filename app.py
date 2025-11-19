import streamlit as st
from PIL import Image
import io

from models.gfpgan import restore_face_gfpgan
from models.real_esrgan import enhance_with_realesrgan   # ← CORREGIDO
from utils.image_utils import compare_images

st.set_page_config(page_title="Mejora de Imágenes", layout="wide")
st.title("🖼️ Mejora de Imágenes con IA")
st.write("Sube una imagen y mejora su calidad mediante super-resolución y restauración facial.")

# ----------------------
# SIDEBAR
# ----------------------
with st.sidebar:
    st.header("Configuración")

    procesar_sr = st.checkbox("Super-resolución (Real-ESRGAN)", value=True)
    procesar_face = st.checkbox("Restauración facial (GFPGAN)", value=True)

    escala = st.selectbox("Escala de subida (Real-ESRGAN):", [1, 2, 3, 4], index=1)

st.write("---")

# ----------------------
# UPLOAD
# ----------------------
archivo = st.file_uploader("Sube tu imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])

if archivo:
    img_original = Image.open(archivo).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imagen original")
        st.image(img_original, use_column_width=True)

    # ----------------------
    # PROCESAMIENTO
    # ----------------------
    st.write("---")
    st.subheader("Procesando...")

    img_procesada = img_original.copy()

    # SUPER-RESOLUCIÓN
    if procesar_sr:
        with st.spinner("Aplicando Real-ESRGAN..."):
            try:
                img_procesada = enhance_with_realesrgan(img_procesada, scale=escala)
            except Exception as e:
                st.error(f"Error en Real-ESRGAN: {e}")

    # RESTAURACIÓN FACIAL
    if procesar_face:
        with st.spinner("Aplicando GFPGAN..."):
            try:
                img_procesada = restore_face_gfpgan(img_procesada)
            except Exception as e:
                st.error(f"Error en GFPGAN: {e}")

    # ----------------------
    # RESULTADOS
    # ----------------------
    with col2:
        st.subheader("Imagen procesada")
        st.image(img_procesada, use_column_width=True)

    # Comparación
    st.write("---")
    st.subheader("Comparación lado a lado")
    comp = compare_images(img_original, img_procesada)
    st.image(comp, use_column_width=True)

    # Descargar
    st.write("---")
    st.subheader("Descargar resultado")

    buffer = io.BytesIO()
    img_procesada.save(buffer, format="PNG")
    st.download_button(
        label="⬇️ Descargar imagen mejorada",
        data=buffer.getvalue(),
        file_name="imagen_mejorada.png",
        mime="image/png"
    )
##streamlit run project/app.py