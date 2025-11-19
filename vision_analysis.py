from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Cargar el modelo CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def analyze_image_clip(img: Image.Image):
    # Etiquetas de análisis
    labels = [
        "imagen restaurada de alta calidad",
        "imagen con artefactos",
        "fotografía borrosa",
        "rostro humano",
        "objeto",
        "escena al aire libre",
        "escena en interior",
    ]

    inputs = clip_processor(text=labels, images=img, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = clip_model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()[0]

    resultados = list(zip(labels, probs))
    resultados.sort(key=lambda x: x[1], reverse=True)

    # Devolver el top 3 más relevante
    analisis = "\n".join([
        f"- {label}: {prob:.2f}"
        for label, prob in resultados[:3]
    ])

    return analisis
