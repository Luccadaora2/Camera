import requests
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

HTML_TEST_PAGE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="UTF-8">
  <title>Teste IA - Detecção de Pessoas</title>
</head>
<body style="font-family:Arial; text-align:center; padding:50px;">
  <h2>🟢 Teste de Detecção de Pessoas</h2>
  <form id="form" enctype="multipart/form-data" method="POST" action="/detect">
    <input type="file" name="image" accept="image/*" required><br><br>
    <button type="submit">Enviar imagem</button>
  </form>
  <pre id="result" style="margin-top:20px; text-align:left;"></pre>

  <script>
    const form = document.getElementById('form');
    form.onsubmit = async (e) => {
      e.preventDefault();
      const data = new FormData(form);
      const res = await fetch('/detect', { method: 'POST', body: data });
      const json = await res.json();
      document.getElementById('result').textContent = JSON.stringify(json, null, 2);
    };
  </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return HTML_TEST_PAGE

@app.route("/detect", methods=["POST"])
def detect_person():
    image = request.files.get("image")
    if not image:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    # IA sem token (modelo público)
    response = requests.post(
        "https://api-inference.huggingface.co/models/keremberke/yolov8m-person-detection",
        files={"file": image}
    )

    try:
        return jsonify(response.json())
    except Exception:
        return jsonify({"error": "Erro ao processar resposta"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
