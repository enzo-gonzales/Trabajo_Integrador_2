"""
Wrapper sencillo para GFPGAN.
Usa la clase GFPGANer si está disponible.
"""

from PIL import Image
import numpy as np

def restore_face_gfpgan(img: Image.Image) -> Image.Image:
    try:
        from gfpgan import GFPGANer
    except Exception as e:
        raise RuntimeError(
            "No se pudo importar 'gfpgan'. Asegúrate de instalarlo. Error: " + str(e)
        )

    # Convertir a numpy array (BGR) como GFPGAN espera
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb)[:, :, ::-1]  # RGB → BGR

    try:
        # Inicializar GFPGANer
        restorer = GFPGANer(
            model_name="GFPGANv1",
            upscale=1,
            arch="clean",
            channel_multiplier=2
        )

        cropped_faces, restored_faces, restored_img = restorer.enhance(
            arr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

        # restored_img es un array BGR → convertir a PIL RGB
        restored_img_rgb = restored_img[:, :, ::-1]
        return Image.fromarray(restored_img_rgb)

    except Exception as e:
        raise RuntimeError("Ocurrió un error al ejecutar GFPGAN: " + str(e))