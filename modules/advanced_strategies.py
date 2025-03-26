import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class AdvancedStrategies:
    """
    Advanced trading strategies with real-time monitoring and loss protection
    """
    
    @staticmethod
    def calculate_rsi(data, window=14):
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(data, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def momentum_strategy(hist_data, lookback=14, overbought=70, oversold=30):
        """
        Momentum strategy based on RSI
        
        Parameters:
        hist_data (pandas.DataFrame): Historical price data
        lookback (int): Period for RSI calculation
        overbought (int): RSI level considered overbought
        oversold (int): RSI level considered oversold
        
        Returns:
        dict: Strategy signal and metrics
        """
        # Calculate RSI
        df = hist_data.copy()
        df['RSI'] = AdvancedStrategies.calculate_rsi(df['Close'], window=lookback)
        
        # Get latest values
        latest_rsi = df['RSI'].iloc[-1]
        latest_close = df['Close'].iloc[-1]
        
        # Determine signal
        signal = 'HOLD'
        if latest_rsi < oversold:
            signal = 'BUY'
        elif latest_rsi > overbought:
            signal = 'SELL'
        
        # Calculate metrics
        rsi_direction = 'Rising' if df['RSI'].iloc[-1] > df['RSI'].iloc[-2] else 'Falling'
        rsi_strength = 'Strong' if abs(50 - latest_rsi) > 20 else 'Moderate' if abs(50 - latest_rsi) > 10 else 'Weak'
        
        return {
            'strategy': 'RSI Momentum',
            'signal': signal,
            'metrics': {
                'current_rsi': latest_rsi,
                'rsi_direction': rsi_direction,
                'rsi_strength': rsi_strength,
                'oversold_level': oversold,
                'overbought_level': overbought
            },
            'description': f"RSI at {latest_rsi:.2f} is {rsi_direction} and {rsi_strength}"
        }
    
    @staticmethod
    def trend_following_strategy(hist_data, short_window=20, long_window=50):
        """
        Trend following strategy based on moving average crossovers
        
        Parameters:
        hist_data (pandas.DataFrame): Historical price data
        short_window (int): Period for short-term moving average
        long_window (int): Period for long-term moving average
        
        Returns:
        dict: Strategy signal and metrics
        """
        # Calculate moving averages
        df = hist_data.copy()
        df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()
        
        # Get recent values for signal generation
        recent_data = df.dropna().tail(3)
        
        if len(recent_data) < 3:
            return {
                'strategy': 'Moving Average Crossover',
                'signal': 'INSUFFICIENT_DATA',
                'metrics': {},
                'description': f"Insufficient data for strategy calculation"
            }
        
        # Check for crossover
        signal = 'HOLD'
        
        # Current relationship
        current_diff = recent_data['SMA_Short'].iloc[-1] - recent_data['SMA_Long'].iloc[-1]
        
        # Previous day relationship
        prev_diff = recent_data['SMA_Short'].iloc[-2] - recent_data['SMA_Long'].iloc[-2]
        
        # Detect crossover
        if current_diff > 0 and prev_diff < 0:
            # Short MA crossed above Long MA -> Bullish
            signal = 'BUY'
        elif current_diff < 0 and prev_diff > 0:
            # Short MA crossed below Long MA -> Bearish
            signal = 'SELL'
        elif current_diff > 0:
            # Short MA above Long MA -> Bullish trend
            signal = 'BULLISH'
        elif current_diff < 0:
            # Short MA below Long MA -> Bearish trend
            signal = 'BEARISH'
        
        # Calculate trend strength
        trend_strength = abs(current_diff / recent_data['SMA_Long'].iloc[-1]) * 100
        strength_category = 'Strong' if trend_strength > 3 else 'Moderate' if trend_strength > 1 else 'Weak'
        
        return {
            'strategy': 'Moving Average Crossover',
            'signal': signal,
            'metrics': {
                'short_ma': recent_data['SMA_Short'].iloc[-1],
                'long_ma': recent_data['SMA_Long'].iloc[-1],
                'difference': current_diff,
                'trend_strength': trend_strength,
                'strength_category': strength_category
            },
            'description': f"{'Bullish' if current_diff > 0 else 'Bearish'} trend is {strength_category} ({trend_strength:.2f}%)"
        }
    
    @staticmethod
    def mean_reversion_strategy(hist_data, window=20, num_std=2):
        """
        Mean reversion strategy based on Bollinger Bands
        
        Parameters:
        hist_data (pandas.DataFrame): Historical price data
        window (int): Period for moving average calculation
        num_std (float): Number of standard deviations for bands
        
        Returns:
        dict: Strategy signal and metrics
        """
        # Calculate Bollinger Bands
        df = hist_data.copy()
        df['Upper_Band'], df['Middle_Band'], df['Lower_Band'] = AdvancedStrategies.calculate_bollinger_bands(
            df['Close'], window=window, num_std=num_std
        )
        
        # Get latest values
        latest = df.iloc[-1]
        close_price = latest['Close']
        upper_band = latest['Upper_Band']
        middle_band = latest['Middle_Band']
        lower_band = latest['Lower_Band']
        
        # Determine position relative to bands
        band_width = (upper_band - lower_band) / middle_band * 100  # Normalized as percentage
        
        # Determine signal
        signal = 'HOLD'
        
        if close_price > upper_band:
            signal = 'SELL'  # Price above upper band - potential reversal down
        elif close_price < lower_band:
            signal = 'BUY'   # Price below lower band - potential reversal up
        
        # Calculate percentile position within bands (0% = at lower band, 100% = at upper band)
        if upper_band != lower_band:  # Prevent division by zero
            band_position = (close_price - lower_band) / (upper_band - lower_band) * 100
        else:
            band_position = 50
        
        # Determine if bands are expanding or contracting by comparing to previous day
        prev_bandwidth = (df['Upper_Band'].iloc[-2] - df['Lower_Band'].iloc[-2]) / df['Middle_Band'].iloc[-2] * 100
        band_direction = 'Expanding' if band_width > prev_bandwidth else 'Contracting'
        
        # Volatility assessment
        volatility_level = 'High' if band_width > 5 else 'Moderate' if band_width > 3 else 'Low'
        
        return {
            'strategy': 'Bollinger Bands Mean Reversion',
            'signal': signal,
            'metrics': {
                'upper_band': upper_band,
                'middle_band': middle_band,
                'lower_band': lower_band,
                'band_width': band_width,
                'band_position': band_position,
                'band_direction': band_direction,
                'volatility': volatility_level
            },
            'description': f"Price at {band_position:.1f}% position in {volatility_level} volatility bands ({band_direction})"
        }
    
    @staticmethod
    def breakout_strategy(hist_data, window=20):
        """
        Breakout strategy based on price channels
        
        Parameters:
        hist_data (pandas.DataFrame): Historical price data
        window (int): Lookback period for resistance/support
        
        Returns:
        dict: Strategy signal and metrics
        """
        df = hist_data.copy()
        
        # Calculate rolling highs and lows
        df['Resistance'] = df['High'].rolling(window=window).max()
        df['Support'] = df['Low'].rolling(window=window).min()
        
        # Recent data
        recent = df.dropna().tail(2)
        
        if len(recent) < 2:
            return {
                'strategy': 'Breakout Detection',
                'signal': 'INSUFFICIENT_DATA',
                'metrics': {},
                'description': f"Insufficient data for strategy calculation"
            }
        
        # Current and previous values
        current = recent.iloc[-1]
        previous = recent.iloc[-2]
        
        # Detect breakouts
        signal = 'HOLD'
        
        # Breakout above resistance
        if current['Close'] > previous['Resistance']:
            signal = 'BUY'
        
        # Breakdown below support
        elif current['Close'] < previous['Support']:
            signal = 'SELL'
        
        # Calculate distance to resistance and support
        distance_to_resistance = ((current['Resistance'] - current['Close']) / current['Close']) * 100
        distance_to_support = ((current['Close'] - current['Support']) / current['Close']) * 100
        
        return {
            'strategy': 'Breakout Detection',
            'signal': signal,
            'metrics': {
                'resistance': current['Resistance'],
                'support': current['Support'],
                'distance_to_resistance': distance_to_resistance,
                'distance_to_support': distance_to_support
            },
            'description': f"Price is {distance_to_resistance:.2f}% below resistance and {distance_to_support:.2f}% above support"
        }

    @staticmethod
    def analyze_all_strategies(hist_data):
        """
        Run all strategies and provide a consolidated analysis
        
        Parameters:
        hist_data (pandas.DataFrame): Historical price data
        
        Returns:
        list: Results from all strategies
        """
        results = []
        
        # Run all individual strategies
        results.append(AdvancedStrategies.momentum_strategy(hist_data))
        results.append(AdvancedStrategies.trend_following_strategy(hist_data))
        results.append(AdvancedStrategies.mean_reversion_strategy(hist_data))
        results.append(AdvancedStrategies.breakout_strategy(hist_data))
        
        return results
    
    @staticmethod
    def calculate_combined_signal(strategy_results):
        """
        Calculate a combined signal from multiple strategy results
        
        Parameters:
        strategy_results (list): Results from multiple strategies
        
        Returns:
        str: Combined signal (BUY, SELL, or HOLD)
        """
        # Count signals
        buy_count = 0
        sell_count = 0
        
        for result in strategy_results:
            if result['signal'] == 'BUY' or result['signal'] == 'BULLISH':
                buy_count += 1
            elif result['signal'] == 'SELL' or result['signal'] == 'BEARISH':
                sell_count += 1
        
        # Determine combined signal
        if buy_count > sell_count and buy_count >= 2:
            return 'BUY'
        elif sell_count > buy_count and sell_count >= 2:
            return 'SELL'
        else:
            return 'HOLD'