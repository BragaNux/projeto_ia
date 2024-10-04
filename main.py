from flask import Flask, render_template, request, jsonify
from modules.audio_module import transcribe_audio_file
from modules.image_module import describe_image_with_clip
from modules.text_module import generate_text_with_ollama
from modules.video_module import process_video_file
import os

app = Flask(__name__)

# Página inicial com a interface
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint para transcrever áudio (via upload de arquivo)
@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "Arquivo de áudio não enviado"}), 400
    
    file = request.files['file']
    
    try:
        audio_path = os.path.join("/tmp", file.filename)
        file.save(audio_path)
        result = transcribe_audio_file(audio_path)
        return jsonify({"transcription": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint para descrever imagem (via upload de arquivo)
@app.route('/describe_image', methods=['POST'])
def describe_image():
    if 'image' not in request.files:
        return jsonify({"error": "Arquivo de imagem não enviado"}), 400
    
    file = request.files['image']
    
    try:
        image_path = os.path.join("/tmp", file.filename)
        file.save(image_path)
        description, similarity = describe_image_with_clip(image_path)
        similarity = float(similarity)
        return jsonify({"description": description, "similarity": similarity})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint para gerar texto
@app.route('/generate_text', methods=['POST'])
def generate_text():
    prompt = request.form.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt não fornecido"}), 400
    
    try:
        result = generate_text_with_ollama(prompt)
        return jsonify({"response": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint para processar vídeo (via upload de arquivo)
@app.route('/process_video', methods=['POST'])
def process_video_route():
    if 'video' not in request.files:
        return jsonify({"error": "Arquivo de vídeo não enviado"}), 400
    
    file = request.files['video']
    
    try:
        video_path = os.path.join("/tmp", file.filename)
        file.save(video_path)
        description = process_video_file(video_path)
        return jsonify({"description": description})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
