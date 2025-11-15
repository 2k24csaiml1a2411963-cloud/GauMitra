from flask import Flask, render_template, request, send_from_directory, jsonify  
import torch
import torch.nn as nn  # 
from torchvision import models, transforms
from PIL import Image
import os
import sqlite3

app = Flask(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


MODEL_PATH = "final_resnet50_balanced.pth" #Load the model 
# Breeds name 
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

# loading the pretrained model 
def load_model():
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model.to(device)

print("Loading model.")
model = load_model()
print("Model loaded successfully!")


# transforming the images in a correct resoultion 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Connencted flask to our database
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

# model acces the inage where it is saved form the website 
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
    else:
        return breed, round(confidence, 2)

# connected the backend with front-end 
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])  # images goes to our model and retrun the result 

# this function checks that , uploaded thing is iamge or not and also save the photo to local folder 
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

    # if the photo is not of cow then prints , its is not that photo  
    if breed is None:
        return jsonify({
            "warning": "This image does not appear to be a cow.",
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

if __name__ == "__main__":
    app.run(debug=True)
