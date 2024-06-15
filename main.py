from flask import Flask, render_template, request, send_file, redirect, url_for
import os
from FormDetection import mainFunction

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    if file:
        global filename
        file.filename = "formDetection.mp4"
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('download_file'))

@app.route('/download')
def download_file():
    mainFunction()
    result = 'output_video.mp4'
    return send_file(result, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
