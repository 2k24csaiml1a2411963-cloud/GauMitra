from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sqlite3
import os

app = Flask(__name__)
CORS(app)  # VERY IMPORTANT for Render + JS Fetch


# --------------------------
# Device Setup
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------
# Model & Classes
# --------------------------
MODEL_PATH = "final_resnet50_balanced.pth"

classes = [
    'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari',
    'Brown_Swiss', 'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar',
    'Hariana', 'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 'Kangayam',
    'Kankrej', 'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari',
    'Krishna_Valley', 'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori',
    'Nagpuri', 'Nili_Ravi', 'Nimari', 'Ongole', 'Pulikulam', 'Rathi',
    'Red_Dane', 'Red_Sindhi', 'Sahiwal', 'Surti', 'Tharparkar', 'Toda',
    'Umblachery', 'Vechur'
]


def load_model():
    print("Loading model...")
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully!")
    return model.to(device)


model = load_model()


# --------------------------
# Image Transform
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# --------------------------
# Database Access
# --------------------------
def get_breed_info(breed_name):
    conn = sqlite3.connect("cows.db")
    cursor = conn.cursor()
    cursor.execute("SELECT origin, color, milk_yield, characteristics FROM breeds WHERE name = ?", (breed_name,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "Origin": row[0],
            "Color": row[1],
            "Milk Yield": row[2],
            "Characteristics": row[3]
        }
    else:
        return {
            "Origin": "Unknown",
            "Color": "Unknown",
            "Milk Yield": "Unknown",
            "Characteristics": "No details found for this breed."
        }


# --------------------------
# Prediction Function
# --------------------------
def predict_breed(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)

    confidence = conf.item() * 100
    breed = classes[predicted.item()]

    if confidence < 35:
        return None, confidence

    return breed, round(confidence, 2)


# --------------------------
# Flask Routes
# --------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")


# 🚨 FIXED ROUTE — MUST BE DIRECTLY ABOVE FUNCTION
@app.route("/predict", methods=["POST"])
def upload_and_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded!"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty file!"}), 400

    os.makedirs("uploads", exist_ok=True)
    image_path = os.path.join("uploads", file.filename)
    file.save(image_path)

    breed, confidence = predict_breed(image_path)

    if breed is None:
        return jsonify({
            "prediction": None,
            "confidence": round(confidence, 2)
        })

    info = get_breed_info(breed)

    return jsonify({
        "prediction": breed,
        "confidence": confidence,
        "info": info
    })


@app.route("/uploads/<filename>")
def send_file(filename):
    return send_from_directory("uploads", filename)


# --------------------------
# Run App
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)
