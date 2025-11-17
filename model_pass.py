import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Load model
model = timm.create_model('mobilenetv3_large_100', pretrained=True)
model.eval()

# ---- USE A LOCAL IMAGE HERE ----
img_path = r"C:\Users\danie\Downloads\types-of-dumbbells-cover.jpg"
img = Image.open(img_path).convert("RGB")

# Preprocessing
config = resolve_data_config({}, model=model)
transform = create_transform(**config)
tensor = transform(img).unsqueeze(0)

# Inference
with torch.inference_mode():
    out = model(tensor)

probabilities = torch.nn.functional.softmax(out[0], dim=0)

# Load ImageNet class names
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Top-5 predictions
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(5):
    print(categories[top5_catid[i]], top5_prob[i].item())
