import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/detect", methods=["POST"])
def detect_person():
    image = request.files.get("image")
    if not image:
        return jsonify({"error": "Envie uma imagem no campo 'image'"}), 400

    # Envia para modelo público sem necessidade de token
    response = requests.post(
        "https://api-inference.huggingface.co/models/keremberke/yolov8m-person-detection",
        files={"file": image}
    )

    try:
        result = response.json()
    except Exception:
        result = {"error": "Erro ao interpretar resposta do modelo."}

    return jsonify(result)

@app.route("/", methods=["GET"])
def home():
    return "🟢 API de Detecção de Pessoas ativa (sem token Hugging Face)"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
