from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
import google.generativeai as genai
import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
import base64
import cv2
import numpy as np
import logging

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'supersecretkey')
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")

# MongoDB connection
try:
    client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017/'))
    db = client["health_forum"]
    questions_collection = db["questions"]
    alerts_collection = db['alerts']
    print("Connected to MongoDB")
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    questions_collection = None
    alerts_collection = None

# Gemini AI configuration
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-pro')
    print("Gemini configured successfully")
except Exception as e:
    print(f"Failed to configure Gemini: {e}")
    model = None

# Setup logging
logging.basicConfig(level=logging.INFO)

# Dataset configuration for eye disease detection
CLASSES = ['bulging_eyes', 'cataract', 'crossed_eye', 'glaucoma', 'uveitis']

# Initialize Roboflow inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="RBLxcPUj7Talpqvu3TYF"  # Replace with actual API key if needed
)

# SocketIO event handler for messages
@socketio.on('message')
def handle_message(data):
    try:
        if data.startswith('/help'):
            if model is None:
                emit('message', "Error: Gemini is not configured.", room=request.sid)
                return

            question = data[5:].strip()
            try:
                response = model.generate_content(question).text
                title = model.generate_content(f"Generate a concise title for this question: {question}").text
                emit('message', f"{response}", room=request.sid)
            except Exception as e:
                emit('message', "Error: Failed to generate a response.", room=request.sid)
                return

            if questions_collection:
                questions_collection.insert_one({
                    "question": question,
                    "answer": response,
                    "title": title,
                    "timestamp": datetime.now(),
                    "likes": 0,
                    "category": "AI Response",
                    "author": "Anonymous User"
                })

            emit('message', response, broadcast=True, include_self=True)
        else:
            emit('message', data, broadcast=True, include_self=True)
    except Exception as e:
        emit('message', "An error occurred while processing your request.", room=request.sid)

# Route to fetch previous questions
@app.route('/previous-questions', methods=['GET'])
def get_previous_questions():
    if questions_collection is None:
        return jsonify({"error": "MongoDB not connected"}), 500
    questions = list(questions_collection.find().sort("timestamp", -1).limit(50))
    for q in questions:
        q["_id"] = str(q["_id"])
    return jsonify(questions)

# Route to submit symptom reports
@app.route('/submit-report', methods=['POST'])
def submit_report():
    data = request.json
    if not data or 'location' not in data or 'description' not in data:
        return jsonify({"error": "Missing required fields: location and description"}), 400

    new_alert = {
        'title': f"Symptom Report - {data['location']}",
        'location': data['location'],
        'coordinates': {'lat': 10.8505, 'lng': 76.2711},
        'radius': 5,
        'severity': 'medium',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': data['description'],
        'preventiveMeasures': ['Seek medical attention', 'Avoid contact with others', 'Monitor symptoms']
    }
    if alerts_collection is None:
        return jsonify({"error": "MongoDB not connected"}), 500
    result = alerts_collection.insert_one(new_alert)
    new_alert['_id'] = str(result.inserted_id)
    return jsonify(new_alert), 201

# Route to fetch all alerts
@app.route('/alerts', methods=['GET'])
def get_alerts():
    if alerts_collection is None:
        return jsonify({"error": "MongoDB not connected"}), 500
    alerts = list(alerts_collection.find())
    return jsonify([{**alert, '_id': str(alert['_id'])} for alert in alerts]), 200

# Route for eye disease detection
@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        logging.error("Error decoding image: %s", e)
        return jsonify({"error": "Invalid image data"}), 400
    
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Could not decode image"}), 400
    
    try:
        result = CLIENT.infer(image, model_id="eye-disease-svz00/6")
        detections = [
            {
                'class': pred.get('class'),
                'confidence': pred.get('confidence'),
                'bbox': {
                    'x': pred.get('x'), 'y': pred.get('y'),
                    'width': pred.get('width'), 'height': pred.get('height')
                }
            }
            for pred in result.get('predictions', []) if pred.get('class') in CLASSES
        ]
        return jsonify({'detections': detections, 'count': len(detections)})
    except Exception as e:
        logging.error("Error during inference: %s", e)
        return jsonify({"error": "Inference failed"}), 500

# Run the app
if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)