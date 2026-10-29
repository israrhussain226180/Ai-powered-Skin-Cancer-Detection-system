from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.model import get_model

app = Flask(__name__, 
            static_folder='../frontend', 
            template_folder='../frontend')

# --- Model Loading ---
class SkinCancerModel:
    def __init__(self, model_path, num_classes=7):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['akiec', 'bcc', 'bkl', 'df', 'melanoma', 'nevus', 'vasc']
        
        self.model = get_model(num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_bytes):
        image = Image.open(image_bytes).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        return {
            'class': self.class_names[predicted_idx.item()],
            'confidence': float(confidence.item())
        }

MODEL_PATH = os.path.join(project_root, 'models', 'best_efficientnet_b0.pth')
predictor = SkinCancerModel(MODEL_PATH)

# --- Routes ---
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/api/predict/', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    try:
        results = predictor.predict(io.BytesIO(file.read()))
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
