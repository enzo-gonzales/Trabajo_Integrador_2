"""
Wrapper sencillo para Real-ESRGAN.
Si la librería está disponible, usa el modelo real.
Si no, usa un placeholder (redimensionamiento).
"""
from PIL import Image
import numpy as np

def enhance_with_realesrgan(img: Image.Image, scale: int = 4) -> Image.Image:
    """
    Recibe PIL.Image, devuelve PIL.Image mejorada.
    scale: 2 o 4
    """
    try:
        # Intentar importar Real-ESRGAN
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        import torch
        
        # Convertir a RGB si no lo es
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        # Convertir PIL a numpy array
        img_array = np.array(img)
        
        # Determinar dispositivo
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Modelo para escala 4
        if scale == 4:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            model_path = 'RealESRGAN_x4plus.pth'
        else:  # scale == 2
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            model_path = 'RealESRGAN_x2plus.pth'
            
        # Inicializar upsampler
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            device=device
        )
        
        # Mejorar imagen
        output, _ = upsampler.enhance(img_array, outscale=scale)
        
        # Convertir numpy array de vuelta a PIL
        result = Image.fromarray(output)
        return result
        
    except ImportError:
        # Si no está instalado, usar placeholder (redimensionamiento)
        print("Real-ESRGAN no está instalado. Usando placeholder.")
        width, height = img.size
        new_size = (width * scale, height * scale)
        return img.resize(new_size, Image.Resampling.LANCZOS)
        
    except Exception as e:
        # Si hay otro error, usar placeholder
        print(f"Error con Real-ESRGAN: {e}. Usando placeholder.")
        width, height = img.size
        new_size = (width * scale, height * scale)
        return img.resize(new_size, Image.Resampling.LANCZOS)