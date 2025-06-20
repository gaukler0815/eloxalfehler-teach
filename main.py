
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

labels = ['Flecken', 'Glasperlenstruktur', 'Kratzer', 'Säureflecken', 'Verfärbung']

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(labels))
model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.title("Eloxal Fehleranalyse – KI unterstützt")

uploaded_files = st.file_uploader("Lade 1–5 Bilder hoch", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            pred = torch.argmax(probs).item()
        st.write(f"📌 Fehler erkannt: **{labels[pred]}** ({probs[pred]*100:.2f}%)")
