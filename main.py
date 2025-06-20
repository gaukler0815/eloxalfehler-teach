
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# Anzahl der Fehlerklassen im Modell – HIER ANPASSEN!
NUM_CLASSES = 5

# Modell laden
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Bildvorverarbeitung
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Streamlit Oberfläche
st.title("Eloxal Fehleranalyse KI")

uploaded_files = st.file_uploader("Lade 1 bis 5 Bilder hoch", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=uploaded_file.name, use_column_width=True)

        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

        st.write(f"🧠 **Erkannte Klasse:** {predicted_class}")
        st.write(f"🔍 **Confidence:** {confidence:.2%}")
