# ML_Stock_Prediction
A Flask-based stock price prediction web application using an LSTM neural network. The model is trained on historical stock data using PyTorch and predicts next-day prices in real time using Yahoo Finance data.

📌 Features

Predicts next-day stock closing price using an LSTM neural network

Accepts real-time stock symbols (e.g., AAPL, TSLA, MSFT)

Fetches latest stock data automatically using Yahoo Finance

Scales data using MinMaxScaler for accurate predictions

User-friendly web interface built with HTML and CSS

Displays predicted price dynamically on the web page

Handles invalid or unavailable stock symbols gracefully

🛠️ Technologies Used

Python

Flask (Web Framework)

PyTorch (LSTM Neural Network)

Yahoo Finance API (yfinance)

NumPy

Scikit-learn (MinMaxScaler)

HTML & CSS (Frontend)

Joblib (Model & Scaler persistence)

▶️ How to Run

Clone or download this repository.

Open a terminal/command prompt in the project directory.

(Optional) Train the model:

python train_model.py


Run the Flask application:

python app.py


Open your browser and go to:

http://127.0.0.1:5000/


Enter a stock symbol and click Predict.
