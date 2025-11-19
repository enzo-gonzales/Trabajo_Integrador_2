{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "a1ea7642",
   "metadata": {},
   "source": [
    " ## TP_2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "d914c67f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Carpeta creada: project\n",
      "✅ Carpeta creada: project/models\n",
      "✅ Carpeta creada: project/utils\n",
      "✅ Carpeta creada: project/static\n",
      "🎯 Todas las carpetas han sido creadas\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "\n",
    "# Definir las rutas de las carpetas a crear\n",
    "carpetas = [\n",
    "    \"project\",\n",
    "    \"project/models\", \n",
    "    \"project/utils\",\n",
    "    \"project/static\"\n",
    "\n",
    "]\n",
    "\n",
    "# Crear cada carpeta si no existe\n",
    "for carpeta in carpetas:\n",
    "    try:\n",
    "        os.makedirs(carpeta, exist_ok=True)\n",
    "        print(f\"✅ Carpeta creada: {carpeta}\")\n",
    "    except Exception as e:\n",
    "        print(f\"❌ Error creando {carpeta}: {e}\")\n",
    "\n",
    "print(\"🎯 Todas las carpetas han sido creadas\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e01db3a5",
   "metadata": {},
   "source": [
    "## 1) Celda: instalar dependencias\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "1443b3a3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🔄 Actualizando pip, setuptools y wheel...\n",
      "📦 Instalando dependencias...\n",
      "Instalando streamlit...\n",
      "✅ streamlit instalado correctamente\n",
      "Instalando pyngrok...\n",
      "✅ pyngrok instalado correctamente\n",
      "Instalando pillow...\n",
      "✅ pillow instalado correctamente\n",
      "Instalando numpy...\n",
      "✅ numpy instalado correctamente\n",
      "Instalando opencv-python...\n",
      "✅ opencv-python instalado correctamente\n",
      "Instalando tqdm...\n",
      "✅ tqdm instalado correctamente\n",
      "Instalando realesrgan...\n",
      "✅ realesrgan instalado correctamente\n",
      "Instalando gfpgan...\n",
      "✅ gfpgan instalado correctamente\n",
      "Instalando basicsr...\n",
      "✅ basicsr instalado correctamente\n",
      "Instalando facexlib...\n",
      "✅ facexlib instalado correctamente\n",
      "\n",
      "🎯 ¡Todas las dependencias han sido instaladas!\n",
      "Puedes verificar la instalación con: pip list\n"
     ]
    }
   ],
   "source": [
    "import subprocess\n",
    "import sys\n",
    "import os\n",
    "\n",
    "def instalar_dependencias():\n",
    "    \"\"\"Instala todas las dependencias del proyecto\"\"\"\n",
    "    \n",
    "    # Lista de paquetes a instalar\n",
    "    paquetes = [\n",
    "        \"streamlit\",\n",
    "        \"pyngrok\", \n",
    "        \"pillow\",\n",
    "        \"numpy\",\n",
    "        \"opencv-python\",\n",
    "        \"tqdm\",\n",
    "        \"realesrgan\",\n",
    "        \"gfpgan\",\n",
    "        \"basicsr\",\n",
    "        \"facexlib\"\n",
    "    ]\n",
    "    \n",
    "    print(\"🔄 Actualizando pip, setuptools y wheel...\")\n",
    "    subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"--upgrade\", \"pip\", \"setuptools\", \"wheel\"])\n",
    "    \n",
    "    print(\"📦 Instalando dependencias...\")\n",
    "    for paquete in paquetes:\n",
    "        try:\n",
    "            print(f\"Instalando {paquete}...\")\n",
    "            subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", paquete])\n",
    "            print(f\"✅ {paquete} instalado correctamente\")\n",
    "        except subprocess.CalledProcessError as e:\n",
    "            print(f\"❌ Error instalando {paquete}: {e}\")\n",
    "    \n",
    "    print(\"\\n🎯 ¡Todas las dependencias han sido instaladas!\")\n",
    "    print(\"Puedes verificar la instalación con: pip list\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    instalar_dependencias()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ee00bd49",
   "metadata": {},
   "source": [
    " ## 2) Celda: crear estructura de proyecto y archivos (escribe todo a disco)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "37b7f262",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "from PIL import Image\n",
    "from io import BytesIO\n",
    "import os\n",
    "import sys"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "0a35e42f",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-11-19 15:28:21.957 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\Enzo\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python310\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "from PIL import Image\n",
    "from io import BytesIO\n",
    "\n",
    "# Funciones locales para evitar problemas de importación\n",
    "def pil_to_bytes(image):\n",
    "    buf = BytesIO()\n",
    "    image.save(buf, format=\"PNG\")\n",
    "    return buf.getvalue()\n",
    "\n",
    "def load_image_from_upload(uploaded_file):\n",
    "    image = Image.open(uploaded_file)\n",
    "    if image.mode in ('RGBA', 'LA', 'P'):\n",
    "        image = image.convert('RGB')\n",
    "    return image\n",
    "\n",
    "def enhance_with_realesrgan(image, scale=4):\n",
    "    # Placeholder para Real-ESRGAN\n",
    "    width, height = image.size\n",
    "    new_size = (width * scale, height * scale)\n",
    "    return image.resize(new_size, Image.Resampling.LANCZOS)\n",
    "\n",
    "def restore_face_gfpgan(image):\n",
    "    # Placeholder para GFPGAN\n",
    "    return image\n",
    "\n",
    "# Interfaz de Streamlit\n",
    "st.set_page_config(page_title=\"Mejorador de Fotos — Enzo\", layout=\"centered\")\n",
    "st.title(\"Mejorador de Fotos — Real-ESRGAN + GFPGAN\")\n",
    "st.write(\"Sube una imagen y elige la mejora. Hecho para entregar en clase.\")\n",
    "\n",
    "with st.sidebar:\n",
    "    st.header(\"Modo\")\n",
    "    use_realesrgan = st.checkbox(\"Real-ESRGAN (super-resolución)\", value=True)\n",
    "    use_gfpgan = st.checkbox(\"GFPGAN (restauración de rostros)\", value=True)\n",
    "    scale_choice = st.selectbox(\"Escala (Real-ESRGAN)\", [\"x2\", \"x4\"], index=1)\n",
    "    if scale_choice == \"x2\":\n",
    "        scale = 2\n",
    "    else:\n",
    "        scale = 4\n",
    "\n",
    "st.write(\"---\")\n",
    "uploaded = st.file_uploader(\"Subí tu imagen (jpg, png)\", type=[\"jpg\", \"jpeg\", \"png\"])\n",
    "\n",
    "if uploaded:\n",
    "    st.image(uploaded, caption=\"Original\", use_column_width=True)\n",
    "    img = load_image_from_upload(uploaded)\n",
    "\n",
    "    if st.button(\"Mejorar imagen\"):\n",
    "        result = img\n",
    "        with st.spinner(\"Procesando...\"):\n",
    "            if use_gfpgan:\n",
    "                result = restore_face_gfpgan(result)\n",
    "            if use_realesrgan:\n",
    "                result = enhance_with_realesrgan(result, scale=scale)\n",
    "\n",
    "        st.success(\"Listo ✅\")\n",
    "        st.image(result, caption=\"Resultado\", use_column_width=True)\n",
    "\n",
    "        b = pil_to_bytes(result)\n",
    "        st.download_button(label=\"Descargar imagen mejorada\", data=b, file_name=\"imagen_mejorada.png\", mime=\"image/png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "d6bd4e09",
   "metadata": {},
   "outputs": [],
   "source": [
    "from PIL import Image\n",
    "from io import BytesIO\n",
    "\n",
    "def pil_to_bytes(image):\n",
    "    buf = BytesIO()\n",
    "    image.save(buf, format=\"PNG\")\n",
    "    return buf.getvalue()\n",
    "\n",
    "def load_image_from_upload(uploaded_file):\n",
    "    image = Image.open(uploaded_file)\n",
    "    if image.mode in ('RGBA', 'LA', 'P'):\n",
    "        image = image.convert('RGB')\n",
    "    return image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "fa5ac61e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from PIL import Image\n",
    "\n",
    "def restore_face_gfpgan(image):\n",
    "    try:\n",
    "        from gfpgan import GFPGANer\n",
    "        return image\n",
    "    except ImportError:\n",
    "        return image"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8732f874",
   "metadata": {},
   "source": [
    " ## 3) Celda: módulo models/real_esrgan.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "827b1ebe",
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "Wrapper sencillo para Real-ESRGAN.\n",
    "Si la librería está disponible, usa el modelo real.\n",
    "Si no, usa un placeholder (redimensionamiento).\n",
    "\"\"\"\n",
    "from PIL import Image\n",
    "import numpy as np\n",
    "\n",
    "def enhance_with_realesrgan(img: Image.Image, scale: int = 4) -> Image.Image:\n",
    "    \"\"\"\n",
    "    Recibe PIL.Image, devuelve PIL.Image mejorada.\n",
    "    scale: 2 o 4\n",
    "    \"\"\"\n",
    "    try:\n",
    "        # Intentar importar Real-ESRGAN\n",
    "        from realesrgan import RealESRGANer\n",
    "        from basicsr.archs.rrdbnet_arch import RRDBNet\n",
    "        import torch\n",
    "        \n",
    "        # Convertir a RGB si no lo es\n",
    "        if img.mode != \"RGB\":\n",
    "            img = img.convert(\"RGB\")\n",
    "            \n",
    "        # Convertir PIL a numpy array\n",
    "        img_array = np.array(img)\n",
    "        \n",
    "        # Determinar dispositivo\n",
    "        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "        \n",
    "        # Modelo para escala 4\n",
    "        if scale == 4:\n",
    "            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)\n",
    "            model_path = 'RealESRGAN_x4plus.pth'\n",
    "        else:  # scale == 2\n",
    "            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)\n",
    "            model_path = 'RealESRGAN_x2plus.pth'\n",
    "            \n",
    "        # Inicializar upsampler\n",
    "        upsampler = RealESRGANer(\n",
    "            scale=scale,\n",
    "            model_path=model_path,\n",
    "            model=model,\n",
    "            tile=400,\n",
    "            tile_pad=10,\n",
    "            pre_pad=0,\n",
    "            device=device\n",
    "        )\n",
    "        \n",
    "        # Mejorar imagen\n",
    "        output, _ = upsampler.enhance(img_array, outscale=scale)\n",
    "        \n",
    "        # Convertir numpy array de vuelta a PIL\n",
    "        result = Image.fromarray(output)\n",
    "        return result\n",
    "        \n",
    "    except ImportError:\n",
    "        # Si no está instalado, usar placeholder (redimensionamiento)\n",
    "        print(\"Real-ESRGAN no está instalado. Usando placeholder.\")\n",
    "        width, height = img.size\n",
    "        new_size = (width * scale, height * scale)\n",
    "        return img.resize(new_size, Image.Resampling.LANCZOS)\n",
    "        \n",
    "    except Exception as e:\n",
    "        # Si hay otro error, usar placeholder\n",
    "        print(f\"Error con Real-ESRGAN: {e}. Usando placeholder.\")\n",
    "        width, height = img.size\n",
    "        new_size = (width * scale, height * scale)\n",
    "        return img.resize(new_size, Image.Resampling.LANCZOS)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ba4c710a",
   "metadata": {},
   "source": [
    " ## 4) Celda: módulo models/gfpgan.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "f44367ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "Wrapper sencillo para GFPGAN.\n",
    "Usa la clase GFPGANer si está disponible.\n",
    "\"\"\"\n",
    "\n",
    "from PIL import Image\n",
    "import numpy as np\n",
    "\n",
    "def restore_face_gfpgan(img: Image.Image) -> Image.Image:\n",
    "    try:\n",
    "        from gfpgan import GFPGANer\n",
    "    except Exception as e:\n",
    "        raise RuntimeError(\n",
    "            \"No se pudo importar 'gfpgan'. Asegúrate de instalarlo. Error: \" + str(e)\n",
    "        )\n",
    "\n",
    "    # Convertir a numpy array (BGR) como GFPGAN espera\n",
    "    img_rgb = img.convert(\"RGB\")\n",
    "    arr = np.array(img_rgb)[:, :, ::-1]  # RGB → BGR\n",
    "\n",
    "    try:\n",
    "        # Inicializar GFPGANer\n",
    "        restorer = GFPGANer(\n",
    "            model_name=\"GFPGANv1\",\n",
    "            upscale=1,\n",
    "            arch=\"clean\",\n",
    "            channel_multiplier=2\n",
    "        )\n",
    "\n",
    "        cropped_faces, restored_faces, restored_img = restorer.enhance(\n",
    "            arr,\n",
    "            has_aligned=False,\n",
    "            only_center_face=False,\n",
    "            paste_back=True\n",
    "        )\n",
    "\n",
    "        # restored_img es un array BGR → convertir a PIL RGB\n",
    "        restored_img_rgb = restored_img[:, :, ::-1]\n",
    "        return Image.fromarray(restored_img_rgb)\n",
    "\n",
    "    except Exception as e:\n",
    "        raise RuntimeError(\"Ocurrió un error al ejecutar GFPGAN: \" + str(e))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "575e521c",
   "metadata": {},
   "source": [
    "## 5) Celda: utils/image_utils.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "0fd5cee1",
   "metadata": {},
   "outputs": [],
   "source": [
    "from PIL import Image\n",
    "from io import BytesIO\n",
    "\n",
    "def load_image_from_upload(uploaded_file):\n",
    "    # uploaded_file es un objeto tipo UploadedFile de Streamlit / file-like\n",
    "    try:\n",
    "        img = Image.open(uploaded_file)\n",
    "        return img.convert(\"RGB\")\n",
    "    except Exception as e:\n",
    "        raise RuntimeError(\"No se pudo leer la imagen subida: \" + str(e))\n",
    "\n",
    "def pil_to_bytes(img: Image.Image, format=\"PNG\"):\n",
    "    buf = BytesIO()\n",
    "    img.save(buf, format=format)\n",
    "    buf.seek(0)\n",
    "    return buf.getvalue()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "07a5e766",
   "metadata": {},
   "source": [
    "## 6) Celda: requirements.txt y README.md"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3d3d1062",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "e0ea8d04",
   "metadata": {},
   "source": [
    "## 7) Celda: arrancar Streamlit con ngrok (pegar tu token)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "95ad8428",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (507122745.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[20], line 1\u001b[1;36m\u001b[0m\n\u001b[1;33m    streamlit run app.py\u001b[0m\n\u001b[1;37m              ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "streamlit run app.py\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
