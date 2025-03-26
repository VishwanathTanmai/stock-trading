import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def predict_next_day(data, ticker):
    """
    Train a machine learning model to predict the next day's stock price
    
    Parameters:
    data (pandas.DataFrame): Historical stock data
    ticker (str): Stock symbol
    
    Returns:
    dict: Prediction results and model metrics
    """
    try:
        # Check if we have enough data
        if len(data) < 30:
            return None
            
        # Create a copy of the data
        df = data.copy()
        
        # Create features for prediction
        # Technical indicators that might help predict price movements
        
        # Moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # Price change
        df['Price_Change'] = df['Close'].pct_change()
        
        # Volatility (standard deviation over a window)
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        
        # Momentum indicators
        df['ROC'] = df['Close'].pct_change(periods=5)  # Rate of Change, 5-day momentum
        
        # Relative price position
        df['Rel_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Target: Next day's closing price
        df['Target'] = df['Close'].shift(-1)
        
        # Drop NaN values
        df = df.dropna()
        
        # Features and target
        features = ['Close', 'Volume', 'MA5', 'MA10', 'MA20', 'Price_Change', 
                    'Volatility', 'Volume_Change', 'Volume_MA5', 'ROC', 'Rel_Position']
        X = df[features]
        y = df['Target']
        
        # Scale the features - convert to numpy array to avoid feature names warning
        X_array = X.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array)
        
        # Split the data: use 80% for training, 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)
        
        # Create and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Predict the next day's price
        # Use the last available row of data
        last_data = X.iloc[-1].values.reshape(1, -1)
        last_data_scaled = scaler.transform(last_data)
        next_day_prediction = model.predict(last_data_scaled)[0]
        
        # Return prediction and metrics
        return {
            'predicted_price': float(next_day_prediction),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        }
        
    except Exception as e:
        print(f"Error in prediction model: {e}")
        return None
