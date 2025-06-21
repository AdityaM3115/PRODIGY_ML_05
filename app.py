from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18
import os

app = Flask(__name__)

# Load the trained model
model = resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Adjust for regression
model.load_state_dict(torch.load("models/food_calorie_model.pth"))
model.eval()

# Define image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Define the main route for file upload and prediction
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    try:
        # Preprocess the uploaded image
        input_tensor = preprocess_image(file_path)

        # Predict the calories using the model
        with torch.no_grad():
            predicted_kcal = model(input_tensor).item()

        return jsonify({"predicted_kcal": round(predicted_kcal, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
