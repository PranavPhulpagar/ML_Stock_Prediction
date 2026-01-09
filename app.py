from flask import Flask, render_template, request
import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
import joblib
import os

app = Flask(__name__)

# ====== Get Current Folder ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ====== Define LSTM Model ======
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ====== Load Model and Scaler ======
model = LSTMModel()
model_path = os.path.join(BASE_DIR, "model.pth")
scaler_path = os.path.join(BASE_DIR, "scaler.save")

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
scaler = joblib.load(scaler_path)

# ====== Flask Routes ======
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        symbol = request.form["symbol"].upper()
        data = yf.download(symbol, period="90d", interval="1d")

        if data.empty:
            prediction = "⚠️ Invalid stock symbol or no data available."
        else:
            close = data['Close'].values.reshape(-1, 1)
            scaled = scaler.transform(close)
            seq = torch.tensor(scaled[-60:].reshape(1, 60, 1), dtype=torch.float32)
            with torch.no_grad():
                pred = model(seq).item()
            prediction = scaler.inverse_transform([[pred]])[0][0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
