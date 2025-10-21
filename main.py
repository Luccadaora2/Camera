from flask import Flask, request, jsonify
from deepface import DeepFace
import requests
import os
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

app = Flask(__name__)

# Pasta onde ficam os rostos conhecidos
KNOWN_FACES_DIR = "rostos_salvos"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

@app.route('/')
def index():
    return "Servidor IA de reconhecimento facial do Lucca está rodando! 🚀"

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    image_url = data.get("imageUrl")

    if not image_url:
        return jsonify({"erro": "Nenhuma imagem recebida"}), 400

    try:
        # Baixa a imagem recebida da câmera
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        frame = np.array(img)

        # Salva a imagem temporariamente
        temp_path = "captura.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Compara com rostos salvos
        for rosto_file in os.listdir(KNOWN_FACES_DIR):
            caminho_rosto = os.path.join(KNOWN_FACES_DIR, rosto_file)
            try:
                result = DeepFace.verify(img1_path=temp_path, img2_path=caminho_rosto, model_name='VGG-Face', enforce_detection=False)
                if result['verified']:
                    print(f"✅ Rosto reconhecido: {rosto_file}")
                    return jsonify({"status": "Rosto conhecido", "pessoa": rosto_file}), 200
            except Exception as e:
                print("Erro na verificação:", e)

        print("⚠️ Rosto desconhecido detectado!")
        return jsonify({"status": "Rosto desconhecido"}), 200

    except Exception as e:
        print("Erro geral:", e)
        return jsonify({"erro": str(e)}), 500


if __name__ == '__main__':
    # Render usa porta pelo env PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
