
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Fehlerklassen und technische Erklärungen
CLASSES = [
    "Be injured by a collision",
    "Clean sample",
    "Coating cracking",
    "Convex powder",
    "Dirty spot",
    "Drain bottom",
    "non-conducting",
    "Orange peel",
    "pitting",
    "scuffing",
    "The transverse strip is dented"
]

EXPLANATIONS = {
    "Be injured by a collision": "Mechanische Beschädigung durch Stoß oder Transportfehler. Ursache: unsachgemäße Handhabung. Lösung: Kantenschutz und sichere Lagerung.",
    "Clean sample": "Fehlerfreies Referenzmuster. Wird für Kalibrierung und Vergleich verwendet.",
    "Coating cracking": "Spannungsrisse in der Schicht, oft durch harte Legierung oder zu hohe Schichtdicke. Lösung: Spannungsarmglühen oder Prozessanpassung.",
    "Convex powder": "Pulveranhaftung mit wulstigen Partikeln. Ursache: Verklumpung bei elektrostatischer Aufladung. Lösung: Filter kontrollieren.",
    "Dirty spot": "Punktuelle Verunreinigung durch Öl oder Fingerabdruck. Lösung: Reinigung oder Beize optimieren.",
    "Drain bottom": "Ablaufspuren durch schlechte Positionierung. Lösung: bessere Neigung oder Halterung verwenden.",
    "non-conducting": "Bereich ohne elektrische Leitfähigkeit. Ursache: schlechte Kontaktierung oder Beschichtung. Lösung: Kontakt prüfen.",
    "Orange peel": "Orangenhautstruktur. Ursache: ungleichmäßige Vorbehandlung. Lösung: Oberfläche glätten.",
    "pitting": "Lochkorrosion. Ursache: Chloride oder kontaminierte Beize. Lösung: Nachspülen mit VE-Wasser.",
    "scuffing": "Kratzspuren durch Reibung. Ursache: mechanischer Kontakt. Lösung: Schutzmaßnahmen am Gestell.",
    "The transverse strip is dented": "Eindellung quer zum Bauteil. Ursache: falsche Halterung oder Druckstelle. Lösung: Bauteilaufnahme verbessern."
}

# Modell laden
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(model.last_channel, len(CLASSES))
    )
    model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Bildtransformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Streamlit UI ---
st.set_page_config(page_title="Eloxal Fehleranalyse", layout="centered")

st.title("🧪 Eloxal Fehleranalyse – KI-gestützt")
st.markdown("Diese Anwendung erkennt typische Fehler auf eloxierten Aluminiumteilen anhand von hochgeladenen Bildern.")
st.info("🔒 **Datenschutzhinweis:** Die Bilder werden **nicht gespeichert** und ausschließlich temporär verarbeitet.")

uploaded_files = st.file_uploader("📷 Lade 1–5 Bilder hoch", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

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

        classname = CLASSES[predicted_class]
        explanation = EXPLANATIONS.get(classname, "Keine Beschreibung verfügbar.")

        st.success(f"**🧠 Erkannter Fehler:** {classname}")
        st.markdown(f"**📊 Sicherheit:** {confidence:.2%}")
        st.markdown(f"**📄 Erklärung:** {explanation}")
