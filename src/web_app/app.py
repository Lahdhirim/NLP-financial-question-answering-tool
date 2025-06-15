from flask import Flask, request
from flask_cors import CORS
import subprocess
import os
import signal
from threading import Timer
import torch

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.toolbox import load_trained_model
from src.utils.schema import DataSchema

app = Flask(__name__)
CORS(app)

angular_process = None
def run_angular_ui():
    global angular_process
    angular_process = subprocess.Popen(
        ["ng", "serve", "--open"], cwd="src/web_app/angular_ui", shell=True)

TRAINED_MODELS_PATH = "trained_models/t5_model.pth"
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return {"error": "No input data provided"}, 400
        
        # Validate input data
        required_fields = [DataSchema.CONTEXT, DataSchema.QUESTION]
        if not all(field in data for field in required_fields):
            return {"error": "Missing required fields in input data"}, 400
        
        context = data[DataSchema.CONTEXT]
        question = data[DataSchema.QUESTION]

        # Load the trained model and tokenizer then generate the answer
        # [HIGH]: load the model only once at the beginning of the application
        model, tokenizer, max_input_length, max_answer_length = load_trained_model(save_path=TRAINED_MODELS_PATH)
        input_tokenized = tokenizer(question, context, max_length=max_input_length, padding="max_length", truncation=True)
        input_ids = torch.tensor(input_tokenized["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(input_tokenized["attention_mask"], dtype=torch.long)
        outputs = model.generate(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), max_length=max_answer_length)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Answer:", answer)
        return {"answer": answer}, 200

    except Exception as e:
        return {"error": str(e)}, 500

@app.route("/shutdown", methods=["POST"])
def shutdown():
    try:
        global angular_process
        if angular_process:
            if os.name == 'posix':  # Linux/macOS
                subprocess.run(f"pkill -f 'ng serve'", shell=True)
            elif os.name == 'nt':  # Windows
                subprocess.run(f"taskkill /F /IM node.exe", shell=True)

            angular_process.wait()
            angular_process = None

        def shutdown_server():
            pid = os.getpid()
            os.kill(pid, signal.SIGTERM)
        Timer(0.5, shutdown_server).start()
        return {"message": "Server shutting down..."}, 200
    except Exception as e:
        return {"message": str(e)}, 500

if __name__ == "__main__":
    run_angular_ui()
    app.run(port=5000, host="127.0.0.1")




