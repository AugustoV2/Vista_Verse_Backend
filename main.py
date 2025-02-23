from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from pymongo import MongoClient
from datetime import datetime
import google.generativeai as genai
import os
from dotenv import load_dotenv
from flask_cors import CORS  # Import CORS

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'supersecretkey')

# Enable CORS for all routes
CORS(app)

# Initialize SocketIO
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
                title_prompt = f"Generate a concise title for this question: {question}"
                title = model.generate_content(title_prompt).text
                print(f"Question: {question}")
                print(f"Title: {title}")
                print(f"Response: {response}")
               
                emit('message', f"{response}", room=request.sid)
            except Exception as e:
                print(f"Error generating response from Gemini: {e}")
                emit('message', "Error: Failed to generate a response. Please try again.", room=request.sid)
                return

            if questions_collection is not None:
                try:
                    question_data = {
                        "question": question,
                        "answer": response,
                        "title": title,
                        "timestamp": datetime.now(),
                        "likes": 0,
                        "category": "AI Response",
                        "author": "Anonymous User"
                    }
                    questions_collection.insert_one(question_data)
                except Exception as e:
                    print(f"Error inserting question into MongoDB: {e}")
                    emit('message', "Error: Failed to save the question. Please try again.", room=request.sid)
                    return

            emit('message', response, broadcast=True, include_self=True)
        else:
            emit('message', data, broadcast=True, include_self=True)
    except Exception as e:
        print(f"Error handling message: {e}")
        emit('message', "An error occurred while processing your request.", room=request.sid)

# Route to fetch previous questions
@app.route('/previous-questions', methods=['GET'])
def get_previous_questions():
    try:
        if questions_collection is None:
            return jsonify({"error": "MongoDB not connected"}), 500

        questions = list(questions_collection.find().sort("timestamp", -1).limit(50))
        for q in questions:
            q["_id"] = str(q["_id"])
        return jsonify(questions)
    except Exception as e:
        print(f"Error fetching previous questions: {e}")
        return jsonify({"error": "Failed to fetch questions"}), 500

# Helper function to convert MongoDB alert to JSON
def alert_to_dict(alert):
    return {
        'id': str(alert['_id']),
        'title': alert['title'],
        'location': alert['location'],
        'coordinates': alert['coordinates'],
        'radius': alert['radius'],
        'severity': alert['severity'],
        'timestamp': alert['timestamp'],
        'description': alert['description'],
        'preventiveMeasures': alert['preventiveMeasures']
    }

# Route to submit symptom reports
@app.route('/submit-report', methods=['POST'])
def submit_report():
    try:
        data = request.json

        # Validate required fields
        if not data or 'location' not in data or 'description' not in data:
            return jsonify({"error": "Missing required fields: location and description"}), 400

        # Create a new alert
        new_alert = {
            'title': f"Symptom Report - {data['location']}",
            'location': data['location'],
            'coordinates': { 'lat': 10.8505, 'lng': 76.2711 },  # Default to Kerala center
            'radius': 5,  # Default radius
            'severity': 'medium',  # Default severity
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': data['description'],
            'preventiveMeasures': [
                'Seek medical attention',
                'Avoid contact with others',
                'Monitor symptoms'
            ]
        }

        # Insert the new alert into MongoDB
        if alerts_collection is None:
            return jsonify({"error": "MongoDB not connected"}), 500

        result = alerts_collection.insert_one(new_alert)
        new_alert['_id'] = result.inserted_id

        # Return the new alert as JSON
        return jsonify(alert_to_dict(new_alert)), 201
    except Exception as e:
        print(f"Error submitting report: {e}")
        return jsonify({"error": "Failed to submit report"}), 500

# Route to fetch all alerts
@app.route('/alerts', methods=['GET'])
def get_alerts():
    try:
        if alerts_collection is None:
            return jsonify({"error": "MongoDB not connected"}), 500

        alerts = list(alerts_collection.find())
        return jsonify([alert_to_dict(alert) for alert in alerts]), 200
    except Exception as e:
        print(f"Error fetching alerts: {e}")
        return jsonify({"error": "Failed to fetch alerts"}), 500

# Run the app
if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)