from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import requests
import base64


load_dotenv()

app = Flask(__name__)
CORS(app)

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

@app.route('/', methods=['GET'])
def home():
    return "Hello, testing..."

if __name__ == '__main__':
    app.run(debug=True)