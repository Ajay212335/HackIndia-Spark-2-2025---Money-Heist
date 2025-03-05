from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from flask_cors import CORS
import requests
import json
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, request, render_template
from PIL import Image
from model import create_model

app = Flask(__name__)
CORS(app)  

UPLOAD_FOLDER = "static/uploads/"
MODEL_PATH = "rural_problem_model.pth"
CATEGORIES = ["Drainage", "Road", "Drought", "Electricity", "Others"]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
NUM_CLASSES = len(CATEGORIES)
model = create_model(NUM_CLASSES)

# Check if the model file exists before loading
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    print("Model loaded successfully!")
else:
    print(f"Error: Model file '{MODEL_PATH}' not found.")

model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


app.secret_key = 'secret_key'
app.config['UPLOAD_FOLDER'] = './static/uploads'

client = MongoClient("mongodb://localhost:27017/")
db = client['complaint_management']

users = db['users']
officers = db['officers']
complaints = db['complaints']
collection = db['messages']

API_KEY = "AIzaSyCvIOVnZUf7M-0d3IoHpVzy6zFfR5PTKW4"
chatbot_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

def get_gemini_response(user_query):
    """
    Handles rural area queries using the chatbot API.
    """
    rural_keywords = ["rural", "village", "farm", "agriculture", "countryside", "remote" ,"hi"]
    
    if any(keyword in user_query.lower() for keyword in rural_keywords):
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [{"text": user_query}]
                }
            ]
        }

        try:
            response = requests.post(f"{chatbot_url}?key={API_KEY}", headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()
            return parse_response(data)
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"
    else:
        return "This bot only handles rural area problems."

def parse_response(data):
    """
    Parses the response from the chatbot API.
    """
    try:
        candidates = data.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                return parts[0].get("text", "No response received.")
        return "Unexpected response format from Gemini AI."
    except Exception as e:
        return f"Error parsing response: {e}"

@app.route('/login_user', methods=['GET', 'POST'])
def login_user():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users.find_one({'username': username})
        
        if user:
            if user['password'] == password:
                session['username'] = username
                return redirect(url_for('chatbot'))
            else:
                flash('Invalid password! Please try again.', 'error')
        else:
            flash('Invalid username! Please try again.', 'error')
    
    return render_template('login_user.html')



@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_query = request.form.get('user_query')
        if user_query:
            response = get_gemini_response(user_query)
            return render_template('chatbot.html', response=response)
        else:
            flash("Please enter a query.")
            return render_template('chatbot.html')

    return render_template('chatbot.html')

@app.route('/')
def home():
    return redirect(url_for('index'))

@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/register_user', methods=['GET', 'POST'])
def register_user():
    if request.method == 'POST':
        data = {
            'name': request.form['name'],
            'mobile': request.form['mobile'],
            'age': request.form['age'],
            'address': request.form['address'],
            'username': request.form['username'],
            'password': request.form['password']
        }
        if request.form['password'] != request.form['confirm_password']:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('register_user'))

        if users.find_one({'username': data['username']}):
            flash('Username already exists! Please choose another.', 'error')
            return redirect(url_for('register_user'))

        users.insert_one(data)
        flash('User registered successfully! Please log in.', 'success')
        return redirect(url_for('login_user')) 

    return render_template('register_user.html')

@app.route('/register_officer', methods=['GET', 'POST'])
def register_officer():
    if request.method == 'POST':

        data = {
            'name': request.form['name'],
            'mobile': request.form['mobile'],
            'location': request.form['location'],
            'department': request.form['department'],
            'designation': request.form['designation'],
            'username': request.form['username'],
            'password': (request.form['password'])  
        }

        if request.form['password'] != request.form['confirm_password']:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('register_officer'))

        if officers.find_one({'username': data['username']}):
            flash('Username already exists! Please choose another.', 'error')
            return redirect(url_for('register_officer'))

        officers.insert_one(data)
        flash('Officer registered successfully!', 'success')
        return redirect(url_for('profile', username=data['username']))

    return render_template('register_officer.html')

@app.route('/officer_dashboard')
def officer_dashboard():

    if 'officer_username' not in session:
        return redirect(url_for('login_officer'))

    officer = officers.find_one({"username": session['officer_username']})

    officer_complaints = list(complaints.find())
    
    return render_template(
        'officer_dashboard.html',
        complaints=officer_complaints,
        officer=officer
    )

@app.route('/profile')
def profile():

    if 'officer_username' not in session:
        return redirect(url_for('login_officer'))

    officer = officers.find_one({"username": session['officer_username']})

    return render_template('profile.html', officer=officer)




@app.route('/login_officer', methods=['GET', 'POST'])
def login_officer():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        officer = officers.find_one({'username': username})
        
        if officer:

            if officer['password'] == password:
                session['officer_username'] = username
                return redirect(url_for('officer_dashboard'))
            else:
                flash('Invalid password! Please try again.', 'error')
        else:
            flash('Invalid username! Please try again.', 'error')
    
    return render_template('login_officer.html')


@app.route('/user_dashboard', methods=['GET', 'POST'])
def user_dashboard():
    if 'username' not in session:
        return redirect(url_for('login_user'))
    
    if request.method == 'POST':
        complaint_data = {
            'username': session['username'],
            'name': request.form['name'],
            'phone': request.form['phone'],
            'aadhar': request.form['aadhar'],
            'address': request.form['address'],
            'district': request.form['district'],
            'pincode': request.form['pincode'],
            'complaint': request.form['complaint'],
            'location': request.form['location']
        }

        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                filename = secure_filename(image.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(filepath)
                complaint_data['image'] = filename

        complaints.insert_one(complaint_data)
        flash('Complaint registered successfully!')

    return render_template('user_dashboard.html', username=session['username'])

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'officer_username' not in session:
        return redirect(url_for('login_officer'))

    if request.method == 'POST':
        selected_theme = request.form['theme']
        session['theme'] = selected_theme 
        flash('Theme updated successfully!', 'success')
    current_theme = session.get('theme', 'light')
    
    return render_template('settings.html', theme=current_theme)


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        full_name = request.form['fullName']
        phone = request.form['phone']
        email = request.form['email']
        message = request.form['message']

        contact_data = {
            'fullName': full_name,
            'phone': phone,
            'email': email,
            'message': message
        }
        collection.insert_one(contact_data)
        return redirect(url_for('index'))

    return render_template('index.html')


@app.route("/upload", methods=["POST"])  # Only allow POST requests
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Image processing
        image = Image.open(filepath).convert("RGB")
        image = transform(image).unsqueeze(0)  

        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1)[0]
            predictions = {CATEGORIES[i]: float(probabilities[i]) * 100 for i in range(len(CATEGORIES))}
            
            predicted_class = max(predictions, key=predictions.get)
            confidence = predictions[predicted_class]

        return jsonify({
            "filepath": filepath,
            "prediction": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "all_predictions": predictions
        })

    return jsonify({"error": "Invalid request"}), 400


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    if not os.path.exists('./static/uploads'):
        os.makedirs('./static/uploads')
    app.run(debug=True)
