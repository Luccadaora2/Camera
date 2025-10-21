from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Caminho onde ficam os rostos conhecidos
KNOWN_FACES_DIR = "rostos_salvos"

# Cria pasta se não existir
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Carrega os rostos conhecidos
def load_known_faces():
    known_faces = []
    known_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            known_faces.append(path)
            known_names.append(os.path.splitext(filename)[0])
    return known_faces, known_names

known_faces, known_names = load_known_faces()

@app.route("/")
def home():
    return "✅ Servidor IA ativo e pronto para reconhecimento facial!"

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json()
        image_url = data.get("imageUrl")

        if not image_url:
            return jsonify({"error": "Faltando campo 'imageUrl'"}), 400

        print(f"📸 Imagem recebida: {image_url}")

        # Salva imagem temporária
        image_path = "/tmp/temp.jpg"
        os.system(f"curl -s {image_url} -o {image_path}")

        # Verifica se o arquivo baixou
        if not os.path.exists(image_path):
            return jsonify({"error": "Falha ao baixar imagem"}), 500

        # Verifica se há rostos conhecidos
        if not known_faces:
            return jsonify({"message": "Nenhum rosto conhecido cadastrado"}), 200

        # Compara a imagem recebida com os rostos conhecidos
        result = DeepFace.find(
            img_path=image_path,
            db_path=KNOWN_FACES_DIR,
            model_name="VGG-Face",
            enforce_detection=False
        )

        # Analisa resultado
        if len(result) > 0 and not result[0].empty:
            matched_name = os.path.basename(result[0].iloc[0]["identity"])
            msg = f"✅ Rosto conhecido detectado: {matched_name}"
        else:
            msg = "🚨 Pessoa desconhecida detectada!"

        print(msg)
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        # Registra no log
        with open("logs.txt", "a") as log:
            log.write(f"[{now}] {msg}\n")

        return jsonify({"status": "ok", "message": msg}), 200

    except Exception as e:
        print("❌ Erro no processamento:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
