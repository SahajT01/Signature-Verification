from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import requests
import numpy as np
import base64
from io import BytesIO
from skimage import io, transform, filters, img_as_ubyte
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple
import matplotlib.pyplot as plt

load_dotenv()

app = Flask(__name__)
CORS(app)

def preprocess_signature(image: np.ndarray, canvas_dim: Tuple[int, int],
                         image_dim: Tuple[int, int] = (170, 242),
                         final_dim: Tuple[int, int] = (150, 220)) -> np.ndarray:
    image = image.astype(np.uint8)
    centered_image = adjust_image(image, canvas_dim)
    inverted_image = 255 - centered_image
    resized_image = scale_image(inverted_image, image_dim)

    if final_dim is not None and final_dim != image_dim:
        final_image = center_crop(resized_image, final_dim)
    else:
        final_image = resized_image

    return final_image

def adjust_image(image: np.ndarray, canvas_dim: Tuple[int, int] = (840, 1360)) -> np.ndarray:
    blur_amount = 2
    blurred_image = filters.gaussian(image, blur_amount, preserve_range=True)
    threshold_value = filters.threshold_otsu(image)

    # Calculate the center of mass (CoM)
    binary_image = blurred_image > threshold_value
    rows, cols = np.where(binary_image == 0)
    row_center = int(rows.mean() - rows.min())
    col_center = int(cols.mean() - cols.min())

    cropped_image = image[rows.min(): rows.max(), cols.min(): cols.max()]
    img_rows, img_cols = cropped_image.shape
    canvas_rows, canvas_cols = canvas_dim

    row_start = canvas_rows // 2 - row_center
    col_start = canvas_cols // 2 - col_center

    if img_rows > canvas_rows:
        row_start = 0
        excess_rows = img_rows - canvas_rows
        crop_start = excess_rows // 2
        cropped_image = cropped_image[crop_start:crop_start + canvas_rows, :]
        img_rows = canvas_rows
    else:
        extra_rows = (row_start + img_rows) - canvas_rows
        if extra_rows > 0:
            row_start -= extra_rows
        if row_start < 0:
            row_start = 0

    if img_cols > canvas_cols:
        col_start = 0
        excess_cols = img_cols - canvas_cols
        crop_start = excess_cols // 2
        cropped_image = cropped_image[:, crop_start:crop_start + canvas_cols]
        img_cols = canvas_cols
    else:
        extra_cols = (col_start + img_cols) - canvas_cols
        if extra_cols > 0:
            col_start -= extra_cols
        if col_start < 0:
            col_start = 0

    normalized_image = np.ones((canvas_rows, canvas_cols), dtype=np.uint8) * 255
    normalized_image[row_start:row_start + img_rows, col_start:col_start + img_cols] = cropped_image
    normalized_image[normalized_image > threshold_value] = 255

    return normalized_image

def scale_image(image: np.ndarray, dimensions: Tuple[int, int]) -> np.ndarray:
    target_height, target_width = dimensions
    height_ratio = float(image.shape[0]) / target_height
    width_ratio = float(image.shape[1]) / target_width
    if width_ratio > height_ratio:
        new_height = target_height
        new_width = int(round(image.shape[1] / height_ratio))
    else:
        new_width = target_width
        new_height = int(round(image.shape[0] / width_ratio))
    resized_image = transform.resize(image, (new_height, new_width),
                                     mode='constant', anti_aliasing=True, preserve_range=True)

    resized_image = resized_image.astype(np.uint8)
    if width_ratio > height_ratio:
        start_x = int(round((new_width - target_width) / 2.0))
        return resized_image[:, start_x:start_x + target_width]
    else:
        start_y = int(round((new_height - target_height) / 2.0))
        return resized_image[start_y:start_y + target_height, :]

def center_crop(image: np.ndarray, dimensions: Tuple[int, int]) -> np.ndarray:
    img_height, img_width = image.shape
    crop_height, crop_width = dimensions
    start_y = (img_height - crop_height) // 2
    start_x = (img_width - crop_width) // 2
    cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
    return cropped_image

def crop_center_multiple(images: np.ndarray, dimensions: Tuple[int, int]) -> np.ndarray:
    img_height, img_width = images.shape[2:]
    crop_height, crop_width = dimensions
    start_y = (img_height - crop_height) // 2
    start_x = (img_width - crop_width) // 2
    cropped_images = images[:, :, start_y:start_y + crop_height, start_x:start_x + crop_width]
    return cropped_images

def remove_image_background(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.uint8)
    threshold_value = filters.threshold_otsu(image)
    image[image > threshold_value] = 255
    return image


def decode_image(data_uri):
    """Decode a base64 string to a grayscale image."""
    header, encoded = data_uri.split(",", 1)
    binary_data = base64.b64decode(encoded)
    image = io.imread(BytesIO(binary_data), as_gray=True)
    return img_as_ubyte(image)


class SigNet(nn.Module):
    def __init__(self):
        super(SigNet, self).__init__()
        self.feature_space_size = 2048
        self.conv_layers = nn.Sequential(
            nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(1, 96, 11, stride=4, padding=0, bias=False)),
                ('bn', nn.BatchNorm2d(96)),
                ('mish', nn.Mish())
            ])),
            nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(96, 256, 5, stride=1, padding=2, bias=False)),
                ('bn', nn.BatchNorm2d(256)),
                ('mish', nn.Mish())
            ])),
            nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(256, 384, 3, stride=1, padding=1, bias=False)),
                ('bn', nn.BatchNorm2d(384)),
                ('mish', nn.Mish())
            ])),
            nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(384, 384, 3, stride=1, padding=1, bias=False)),
                ('bn', nn.BatchNorm2d(384)),
                ('mish', nn.Mish())
            ])),
            nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(384, 256, 3, stride=1, padding=1, bias=False)),
                ('bn', nn.BatchNorm2d(256)),
                ('mish', nn.Mish())
            ]))
        )
        self.fc_layers = nn.Sequential(
            nn.Sequential(OrderedDict([
                ('fc', nn.Linear(256 * 3 * 5, 2048, bias=False)),
                ('bn', nn.BatchNorm1d(2048)),
                ('mish', nn.Mish())
            ])),
            nn.Sequential(OrderedDict([
                ('fc', nn.Linear(2048, 2048, bias=False)),
                ('bn', nn.BatchNorm1d(2048)),
                ('mish', nn.Mish())
            ]))
        )

    def forward_once(self, img):
        x = self.conv_layers(img)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def forward(self, img1, img2):
        img1 = img1.view(-1, 1, 150, 220).float() / 255.0
        img2 = img2.view(-1, 1, 150, 220).float() / 255.0
        output1 = self.forward_once(img1)
        output2 = self.forward_once(img2)
        return output1, output2


class SiameseModel(nn.Module):
    def __init__(self):
        super(SiameseModel, self).__init__()

        self.model = SigNet()
        self.prob_layer = nn.Linear(4, 1)
        self.projection_layer = nn.Linear(self.model.feature_space_size, 2)

    def forward_once(self, img):
        return self.model.forward_once(img)

    def forward(self, img1, img2):
        img1 = img1.view(-1, 1, 150, 220).float() / 255.0
        img2 = img2.view(-1, 1, 150, 220).float() / 255.0

        embedding1 = self.forward_once(img1)
        embedding2 = self.forward_once(img2)

        embedding1 = self.projection_layer(embedding1)
        embedding2 = self.projection_layer(embedding2)

        combined_output = torch.cat([embedding1, embedding2], dim=1)
        similarity_score = self.prob_layer(combined_output)

        return embedding1, embedding2, similarity_score

# Load your trained model
model = SiameseModel()
model.load_state_dict(torch.load('../best_model_21.pt')['model'])
model.eval()
model.to('cpu')

@app.route('/create_user', methods=['POST'])
def create_user():
    # Parse JSON data sent from a client
    data = request.get_json()

    # Extract fields
    name = data.get('name')
    email = data.get('email')
    genuine_signature = data.get('genuineSignature')  # Base64-encoded image

    api_url = "https://us-west-2.aws.neurelo.com/rest/user_details/__one"
    headers = {
        "X-API-KEY": os.getenv('NEURELO_VALUE'),  # Get API key from environment variable
        "Content-Type": "application/json"
    }

    api_payload = {
        "name": name,
        "email": email,
        "signature_image": genuine_signature
    }

    response = requests.post(api_url, json=api_payload, headers=headers)

    # Check if the request to the external API was successful 
    if response.status_code == 201:
        # Parse the JSON response from the external API
        api_data = response.json()
        return jsonify({"status": "success", "message": "User created successfully!"})
    else:
        print(f"Failed to send data to external API. \n Status code: {response.status_code}")
        return jsonify({
            "status": "error",
            "message": "Failed to send data to external API.",
        })

@app.route('/get_users', methods=['GET'])
def get_users():
    api_url = "https://us-west-2.aws.neurelo.com/rest/user_details"
    
    headers = {
        "X-API-KEY": os.getenv('NEURELO_VALUE'),
        "Content-Type": "application/json"
    }

    response = requests.get(api_url, headers=headers)

    return response.json()

@app.route('/verify_signature', methods=['POST'])
def verify_signature():
    data = request.get_json()
    if not data or 'image1' not in data or 'image2' not in data:
        return jsonify({'error': 'Missing image data'}), 400

    try:
        img1_array = decode_image(data['image1'])
        img2_array = decode_image(data['image2'])
        img1_processed = preprocess_signature(img1_array, (952, 1360), (256, 256))
        img2_processed = preprocess_signature(img2_array, (952, 1360), (256, 256))

        img1_tensor = torch.tensor(img1_processed)
        img2_tensor = torch.tensor(img2_processed)

        with torch.no_grad():
            output1, output2, confidence = model(img1_tensor, img2_tensor)
            confidence = torch.sigmoid(confidence).item()  # Convert output to probability
            cos_sim = F.cosine_similarity(F.normalize(output1), F.normalize(output2)).item()
            # Ensure similarity is non-negative
            if cos_sim < 0:
                cos_sim *= -1

        # Define a threshold for classification
        threshold = 0.9  # Adjust based on model behavior and validation
        if cos_sim > threshold and confidence > 0.5:
            classification = 'Genuine'
        else:
            classification = 'Forged'
        #classification = 'Genuine' if cos_sim > threshold else 'Forged'
        return jsonify({'similarity': f"{cos_sim * 100:.2f}%", 'classification': classification, 'confidence': confidence})

    except Exception as e:
        return jsonify({'error': 'Failed to process images', 'message': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "Hello, testing..."

if __name__ == '__main__':
    app.run(debug=True)