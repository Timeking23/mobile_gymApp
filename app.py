import timm
import torch
from fastapi import FastAPI, UploadFile
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import io

app = FastAPI()

# Load model once at startup
model = timm.create_model("mobilenetv3_large_100", pretrained=True)
model.eval()

config = resolve_data_config({}, model=model)
transform = create_transform(**config)

@app.post("/predict")
async def predict(file: UploadFile):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    tensor = transform(img).unsqueeze(0)

    with torch.inference_mode():
        preds = model(tensor)
        probs = torch.nn.functional.softmax(preds[0], dim=0)

    top5_prob, top5_catid = torch.topk(probs, 5)

    return {
        "top5": [
            {
                "class_id": int(top5_catid[i]),
                "probability": float(top5_prob[i])
            }
            for i in range(5)
        ]
    }
