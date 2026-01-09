# ML_Stock_Prediction
A Flask-based stock price prediction web application using an LSTM neural network. The model is trained on historical stock data using PyTorch and predicts next-day prices in real time using Yahoo Finance data.


## 📌 Features

- Predicts next-day stock closing price using an LSTM neural network

- Accepts real-time stock symbols (e.g., AAPL, TSLA, MSFT)

- Fetches latest stock data automatically using Yahoo Finance

- Scales data using MinMaxScaler for accurate predictions

- User-friendly web interface built with HTML and CSS

- Displays predicted price dynamically on the web page

- Handles invalid or unavailable stock symbols gracefully
  

## 🛠️ Technologies Used

- Python

- Flask (Web Framework)

- PyTorch (LSTM Neural Network)

- Yahoo Finance API (yfinance)

- NumPy

- Scikit-learn (MinMaxScaler)

- HTML & CSS (Frontend)

- Joblib (Model & Scaler persistence)

  
## ▶️ How to Run

1. Clone or download this repository.

2. Open a terminal/command prompt in the project directory.

3. Train the model (Optional) :
   '''bash 
   python train_model.py


4. Run the Flask application:
    '''bash
    python app.py


5. Open your browser and go to:
   '''bash
    http://127.0.0.1:5000/


6. Enter a stock symbol and click Predict.
