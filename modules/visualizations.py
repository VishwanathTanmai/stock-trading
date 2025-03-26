import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

def create_price_chart(data, ticker, period):
    """
    Create an interactive stock price chart
    
    Parameters:
    data (pandas.DataFrame): Historical stock data
    ticker (str): Stock symbol
    period (str): Time period
    
    Returns:
    plotly.graph_objects.Figure: Interactive stock price chart
    """
    # Create figure
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data['Date'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#2E7D32',  # Green for increasing
            decreasing_line_color='#C62828'   # Red for decreasing
        )
    )
    
    # Add moving averages
    if len(data) >= 20:
        # 20-day moving average
        data['MA20'] = data['Close'].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['MA20'],
                mode='lines',
                line=dict(color='#0D47A1', width=1),
                name='20-day MA'
            )
        )
    
    if len(data) >= 50:
        # 50-day moving average
        data['MA50'] = data['Close'].rolling(window=50).mean()
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['MA50'],
                mode='lines',
                line=dict(color='#FB8C00', width=1),
                name='50-day MA'
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Stock Price',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis_rangeslider_visible=False,  # Hide the range slider
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    # Adjust yaxis to show dollar format
    fig.update_yaxes(tickprefix='$')
    
    return fig

def create_volume_chart(data, ticker):
    """
    Create an interactive volume chart
    
    Parameters:
    data (pandas.DataFrame): Historical stock data
    ticker (str): Stock symbol
    
    Returns:
    plotly.graph_objects.Figure: Interactive volume chart
    """
    # Create figure
    fig = go.Figure()
    
    # Add volume bar chart
    fig.add_trace(
        go.Bar(
            x=data['Date'],
            y=data['Volume'],
            name='Volume',
            marker=dict(
                color='rgba(13, 71, 161, 0.7)',  # Financial blue with opacity
            )
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Trading Volume',
        xaxis_title='Date',
        yaxis_title='Volume',
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        template='plotly_white'
    )
    
    # Format y-axis to show numbers in millions/billions
    fig.update_yaxes(
        tickformat=".2s",
        hoverformat=".2s"
    )
    
    return fig

def create_technical_indicators(data, ticker):
    """
    Create a chart with technical indicators
    
    Parameters:
    data (pandas.DataFrame): Historical stock data
    ticker (str): Stock symbol
    
    Returns:
    plotly.graph_objects.Figure: Chart with technical indicators
    """
    # Create a copy of the data to avoid modifying the original
    df = data.copy()
    
    # Calculate RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD (Moving Average Convergence Divergence)
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # Calculate Bollinger Bands
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA20'] + (df['STD20'] * 2)
    df['Lower_Band'] = df['SMA20'] - (df['STD20'] * 2)
    
    # Create subplots: 3 rows, 1 column
    fig = go.Figure()
    
    # Add main price and Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#212121')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Upper_Band'],
            mode='lines',
            name='Upper Bollinger Band',
            line=dict(color='rgba(46, 125, 50, 0.5)')  # Green with opacity
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['SMA20'],
            mode='lines',
            name='20-day SMA',
            line=dict(color='rgba(13, 71, 161, 0.8)')  # Blue with opacity
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Lower_Band'],
            mode='lines',
            name='Lower Bollinger Band',
            line=dict(color='rgba(198, 40, 40, 0.5)')  # Red with opacity
        )
    )
    
    # Add MACD subplot
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='#0D47A1'),
            yaxis='y2'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Signal_Line'],
            mode='lines',
            name='Signal Line',
            line=dict(color='#FB8C00'),
            yaxis='y2'
        )
    )
    
    # Add MACD Histogram
    colors = ['#2E7D32' if val >= 0 else '#C62828' for val in df['MACD_Histogram']]
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['MACD_Histogram'],
            name='MACD Histogram',
            marker_color=colors,
            yaxis='y2'
        )
    )
    
    # Add RSI subplot
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='#673AB7'),
            yaxis='y3'
        )
    )
    
    # Add overbought/oversold lines for RSI
    fig.add_trace(
        go.Scatter(
            x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
            y=[70, 70],
            mode='lines',
            name='Overbought (70)',
            line=dict(color='rgba(198, 40, 40, 0.5)', dash='dash'),
            yaxis='y3'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
            y=[30, 30],
            mode='lines',
            name='Oversold (30)',
            line=dict(color='rgba(46, 125, 50, 0.5)', dash='dash'),
            yaxis='y3'
        )
    )
    
    # Update layout for subplot arrangement
    fig.update_layout(
        title=f'{ticker} Technical Indicators',
        yaxis=dict(
            title="Price (USD)",
            domain=[0.55, 1],
            tickprefix='$'
        ),
        yaxis2=dict(
            title="MACD",
            domain=[0.3, 0.5],
            anchor="x",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        yaxis3=dict(
            title="RSI",
            domain=[0, 0.25],
            anchor="x",
            overlaying="y",
            side="right",
            range=[0, 100],
            showgrid=False
        ),
        xaxis_title="Date",
        height=800,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    return fig

def create_prediction_chart(hist_data, prediction_data, ticker):
    """
    Create a chart showing historical data and prediction
    
    Parameters:
    hist_data (pandas.DataFrame): Historical stock data
    prediction_data (dict): Prediction results
    ticker (str): Stock symbol
    
    Returns:
    plotly.graph_objects.Figure: Prediction chart
    """
    # Create a copy of the data
    df = hist_data.copy()
    
    # Get the last date from historical data
    last_date = df['Date'].iloc[-1]
    
    # If last_date is already a date object, convert to datetime
    if isinstance(last_date, date):
        last_date = datetime.combine(last_date, datetime.min.time())
    
    # Get the next day for prediction
    if isinstance(last_date, str):
        last_date = datetime.strptime(last_date, '%Y-%m-%d')
        
    next_date = last_date + timedelta(days=1)
    
    # Format date if needed
    if isinstance(next_date, datetime):
        next_date = next_date.date()
    
    # Extract the actual values and predicted value
    actual_dates = df['Date'].tolist()
    actual_prices = df['Close'].tolist()
    
    # Create a single point for the prediction
    predicted_date = [next_date]
    predicted_price = [prediction_data['predicted_price']]
    
    # Create the last actual price point to connect with the prediction
    last_actual_date = [actual_dates[-1]]
    last_actual_price = [actual_prices[-1]]
    
    # Create figure
    fig = go.Figure()
    
    # Add historical price line
    fig.add_trace(
        go.Scatter(
            x=actual_dates,
            y=actual_prices,
            mode='lines',
            name='Historical Close Price',
            line=dict(color='#0D47A1')
        )
    )
    
    # Add the prediction point
    # Determine color based on the prediction direction
    pred_color = '#2E7D32' if predicted_price[0] > last_actual_price[0] else '#C62828'
    
    # Add the prediction line connecting the last actual point to the prediction
    fig.add_trace(
        go.Scatter(
            x=last_actual_date + predicted_date,
            y=last_actual_price + predicted_price,
            mode='lines+markers',
            name='Prediction',
            line=dict(color=pred_color, dash='dash'),
            marker=dict(size=10, color=pred_color)
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    # Adjust yaxis to show dollar format
    fig.update_yaxes(tickprefix='$')
    
    return fig
