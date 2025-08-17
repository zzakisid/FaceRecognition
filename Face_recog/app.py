from flask import Flask, render_template, redirect, url_for, request
import face_recognition_live
import face_recognition_video
import os
import webbrowser
from threading import Thread
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create uploads folder if not exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('Frontpage.html')

@app.route('/LiveFotage', methods=['POST'])
def run_live():
    Thread(target=face_recognition_live.run_live_recognition).start()
    return "Live camera recognition started!\nPress Esc to Stop"

@app.route('/VideoForm', methods=['GET'])
def video_form():
    return render_template('video.html')

@app.route('/RecordedVideo', methods=['POST'])
def run_video():
    if 'video_file' not in request.files:
        return "No file part in the request", 400

    video_file = request.files['video_file']

    if video_file.filename == "":
        return "No selected file", 400

    if not allowed_file(video_file.filename):
        return "Invalid file type", 400

    filename = secure_filename(video_file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(save_path)

    # Run recognition in background
    Thread(target=face_recognition_video.run_video_recognition, args=(save_path,)).start()
    
    return render_template("video_started.html", filename=filename)

def open_browser():
    """Open browser automatically when app starts"""
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    Thread(target=open_browser).start()
    app.run(debug=True)
    