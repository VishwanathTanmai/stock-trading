import datetime
import json
import os
import pandas as pd
import numpy as np
import time
from threading import Thread, Event
from modules.advanced_strategies import AdvancedStrategies

class AutoTrader:
    def __init__(self, portfolio):
        """
        Initialize AutoTrader with a portfolio instance
        
        Parameters:
        portfolio (Portfolio): Portfolio instance for executing trades
        """
        self.portfolio = portfolio
        self.strategies_file = 'data/trading_strategies.json'
        self.trades_log_file = 'data/trade_history.json'
        self.active_strategies = self._load_strategies()
        self.monitoring_active = False
        self.stop_event = Event()
        self.monitoring_thread = None
        self.monitoring_data = {
            'status': 'inactive',
            'last_update': None,
            'monitored_symbols': [],
            'price_history': {},
            'alerts': []
        }
        self._ensure_data_directory()
    
    def _ensure_data_directory(self):
        """Ensure the data directory exists"""
        os.makedirs('data', exist_ok=True)
        
        # Create strategies file if it doesn't exist
        if not os.path.exists(self.strategies_file):
            self._save_strategies()
            
        # Create trade log file if it doesn't exist
        if not os.path.exists(self.trades_log_file):
            with open(self.trades_log_file, 'w') as f:
                json.dump([], f)
                
    def _load_trades_history(self):
        """Load trades history from file or initialize an empty list"""
        try:
            if os.path.exists(self.trades_log_file):
                with open(self.trades_log_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading trades history: {e}")
            return []
            
    def _save_trades_history(self, trade_record):
        """Save a trade to the trade history file"""
        try:
            history = self._load_trades_history()
            history.append(trade_record)
            with open(self.trades_log_file, 'w') as f:
                json.dump(history, f, indent=4)
        except Exception as e:
            print(f"Error saving trade history: {e}")
            
    def start_monitoring(self, symbols, interval_seconds=60, data_fetcher=None):
        """
        Start real-time monitoring of symbols
        
        Parameters:
        symbols (list): List of stock symbols to monitor
        interval_seconds (int): Interval between checks in seconds
        data_fetcher (function): Function that takes a symbol and returns (price, historical_data)
        
        Returns:
        dict: Status of the monitoring process
        """
        if self.monitoring_active:
            return {
                'success': False,
                'message': "Monitoring is already active. Stop it first."
            }
            
        if not data_fetcher:
            return {
                'success': False,
                'message': "No data fetcher provided."
            }
            
        self.stop_event.clear()
        self.monitoring_active = True
        self.monitoring_data = {
            'status': 'active',
            'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'monitored_symbols': symbols,
            'price_history': {symbol: [] for symbol in symbols},
            'alerts': []
        }
        
        # Start monitoring in a separate thread
        self.monitoring_thread = Thread(
            target=self._monitoring_loop,
            args=(symbols, interval_seconds, data_fetcher),
            daemon=True
        )
        self.monitoring_thread.start()
        
        return {
            'success': True,
            'message': f"Started monitoring {len(symbols)} symbols with {interval_seconds}s interval."
        }
        
    def stop_monitoring(self):
        """Stop the real-time monitoring process"""
        if not self.monitoring_active:
            return {
                'success': False,
                'message': "Monitoring is not active."
            }
            
        self.stop_event.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
            
        self.monitoring_active = False
        self.monitoring_data['status'] = 'inactive'
        
        return {
            'success': True,
            'message': "Stopped monitoring process."
        }
        
    def _monitoring_loop(self, symbols, interval_seconds, data_fetcher):
        """Internal monitoring loop that runs in a separate thread"""
        last_predictions = {}
        
        while not self.stop_event.is_set():
            try:
                for symbol in symbols:
                    # Get current price and historical data
                    try:
                        current_price, historical_data = data_fetcher(symbol)
                    except Exception as e:
                        self.monitoring_data['alerts'].append({
                            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'symbol': symbol,
                            'type': 'ERROR',
                            'message': f"Failed to fetch data: {str(e)}"
                        })
                        continue
                    
                    # Add price to history
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.monitoring_data['price_history'][symbol].append({
                        'timestamp': timestamp,
                        'price': current_price
                    })
                    
                    # Limit history size
                    max_history = 1000  # About 16 hours at 1 minute intervals
                    if len(self.monitoring_data['price_history'][symbol]) > max_history:
                        self.monitoring_data['price_history'][symbol] = self.monitoring_data['price_history'][symbol][-max_history:]
                    
                    # Skip prediction for now if we've done it recently
                    if symbol in last_predictions:
                        last_pred_time = datetime.datetime.strptime(
                            last_predictions[symbol]['timestamp'], 
                            '%Y-%m-%d %H:%M:%S'
                        )
                        now = datetime.datetime.now()
                        # Only predict every 15 minutes to avoid excessive CPU usage
                        if (now - last_pred_time).total_seconds() < 900:  # 15 minutes in seconds
                            continue
                    
                    # Make prediction for the symbol
                    try:
                        from modules.predictions import predict_next_day
                        prediction_data = predict_next_day(historical_data, symbol)
                        last_predictions[symbol] = {
                            'timestamp': timestamp,
                            'data': prediction_data
                        }
                    except Exception as e:
                        self.monitoring_data['alerts'].append({
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'type': 'ERROR',
                            'message': f"Failed to make prediction: {str(e)}"
                        })
                        continue
                    
                    # Execute strategies
                    try:
                        results = self.execute_strategies(
                            symbol, 
                            prediction_data, 
                            current_price,
                            historical_data=historical_data
                        )
                        
                        # Log executed trades
                        for result in results:
                            if result['action'] in ['BUY', 'SELL'] and result['success']:
                                trade_record = {
                                    'timestamp': timestamp,
                                    'symbol': symbol,
                                    'action': result['action'],
                                    'price': current_price,
                                    'shares': result.get('shares', 0),
                                    'strategy_id': result.get('strategy_id', 'unknown'),
                                    'strategy_type': result.get('strategy_type', 'unknown'),
                                    'reason': result.get('message', '')
                                }
                                self._save_trades_history(trade_record)
                                
                                # Add alert
                                self.monitoring_data['alerts'].append({
                                    'timestamp': timestamp,
                                    'symbol': symbol,
                                    'type': result['action'],
                                    'message': f"{result['action']} {result.get('shares', 0)} shares at ${current_price:.2f}. {result.get('message', '')}"
                                })
                    except Exception as e:
                        self.monitoring_data['alerts'].append({
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'type': 'ERROR',
                            'message': f"Error executing strategies: {str(e)}"
                        })
                
                # Update last check time
                self.monitoring_data['last_update'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Limit alerts size
                max_alerts = 100
                if len(self.monitoring_data['alerts']) > max_alerts:
                    self.monitoring_data['alerts'] = self.monitoring_data['alerts'][-max_alerts:]
                
            except Exception as e:
                # Log the error but continue monitoring
                error_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.monitoring_data['alerts'].append({
                    'timestamp': error_time,
                    'symbol': 'SYSTEM',
                    'type': 'ERROR',
                    'message': f"Monitoring error: {str(e)}"
                })
            
            # Wait for the next interval or until stopped
            self.stop_event.wait(interval_seconds)
            
    def get_monitoring_status(self):
        """Get the current status of the monitoring process"""
        return {
            'active': self.monitoring_active,
            'last_update': self.monitoring_data.get('last_update'),
            'monitored_symbols': self.monitoring_data.get('monitored_symbols', []),
            'alerts': self.monitoring_data.get('alerts', [])[-10:],  # Last 10 alerts
        }
        
    def get_price_history(self, symbol=None):
        """
        Get the price history collected during monitoring
        
        Parameters:
        symbol (str, optional): Symbol to get history for. If None, returns all symbols.
        
        Returns:
        dict: Price history data
        """
        if not self.monitoring_active and not self.monitoring_data.get('price_history'):
            return {
                'success': False,
                'message': "No price history available. Start monitoring first."
            }
            
        if symbol is not None:
            if symbol in self.monitoring_data.get('price_history', {}):
                return {
                    'success': True,
                    'symbol': symbol,
                    'history': self.monitoring_data['price_history'][symbol]
                }
            else:
                return {
                    'success': False,
                    'message': f"Symbol {symbol} is not being monitored."
                }
        
        return {
            'success': True,
            'symbols': list(self.monitoring_data.get('price_history', {}).keys()),
            'history': self.monitoring_data.get('price_history', {})
        }
    
    def _load_strategies(self):
        """Load trading strategies from file or initialize a new one"""
        try:
            if os.path.exists(self.strategies_file):
                with open(self.strategies_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading strategies: {e}")
            return []
    
    def _save_strategies(self):
        """Save trading strategies to file"""
        try:
            with open(self.strategies_file, 'w') as f:
                json.dump(self.active_strategies, f, indent=4)
        except Exception as e:
            print(f"Error saving strategies: {e}")
    
    def add_strategy(self, symbol, strategy_type, params):
        """
        Add a new trading strategy
        
        Parameters:
        symbol (str): Stock symbol
        strategy_type (str): Type of strategy ('prediction_based', 'threshold')
        params (dict): Strategy parameters
        
        Returns:
        dict: Result of adding the strategy
        """
        # Check if a strategy already exists for this symbol
        for strategy in self.active_strategies:
            if strategy['symbol'] == symbol:
                return {
                    'success': False,
                    'message': f"A strategy for {symbol} already exists. Please remove it first."
                }
        
        # Create new strategy
        strategy = {
            'id': len(self.active_strategies) + 1,
            'symbol': symbol,
            'type': strategy_type,
            'params': params,
            'active': True,
            'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_executed': None,
            'execution_history': []
        }
        
        self.active_strategies.append(strategy)
        self._save_strategies()
        
        return {
            'success': True,
            'message': f"Added new {strategy_type} strategy for {symbol}.",
            'strategy': strategy
        }
    
    def remove_strategy(self, strategy_id):
        """
        Remove a trading strategy
        
        Parameters:
        strategy_id (int): ID of the strategy to remove
        
        Returns:
        dict: Result of removing the strategy
        """
        for i, strategy in enumerate(self.active_strategies):
            if strategy['id'] == strategy_id:
                removed = self.active_strategies.pop(i)
                self._save_strategies()
                return {
                    'success': True,
                    'message': f"Removed {removed['type']} strategy for {removed['symbol']}."
                }
        
        return {
            'success': False,
            'message': f"Strategy with ID {strategy_id} not found."
        }
    
    def toggle_strategy(self, strategy_id, active=None):
        """
        Enable or disable a trading strategy
        
        Parameters:
        strategy_id (int): ID of the strategy to toggle
        active (bool, optional): If provided, set to this value; otherwise toggle
        
        Returns:
        dict: Result of toggling the strategy
        """
        for strategy in self.active_strategies:
            if strategy['id'] == strategy_id:
                if active is None:
                    strategy['active'] = not strategy['active']
                else:
                    strategy['active'] = active
                
                status = "enabled" if strategy['active'] else "disabled"
                self._save_strategies()
                
                return {
                    'success': True,
                    'message': f"{strategy['type'].capitalize()} strategy for {strategy['symbol']} {status}."
                }
        
        return {
            'success': False,
            'message': f"Strategy with ID {strategy_id} not found."
        }
    
    def get_strategies(self):
        """Get all trading strategies as a DataFrame"""
        if not self.active_strategies:
            return pd.DataFrame(columns=['ID', 'Symbol', 'Type', 'Active', 'Created'])
        
        strategies = []
        for strategy in self.active_strategies:
            strategies.append({
                'ID': strategy['id'],
                'Symbol': strategy['symbol'],
                'Type': strategy['type'].replace('_', ' ').title(),
                'Active': "✓" if strategy['active'] else "✗",
                'Created': strategy['created_at'],
                'Last Executed': strategy['last_executed'] or 'Never'
            })
        
        return pd.DataFrame(strategies)
    
    def execute_strategies(self, symbol, prediction_data, current_price, max_investment=10000, historical_data=None):
        """
        Execute all active strategies for a symbol
        
        Parameters:
        symbol (str): Stock symbol
        prediction_data (dict): Prediction results
        current_price (float): Current stock price
        max_investment (float): Maximum amount to invest per trade
        historical_data (pandas.DataFrame, optional): Historical price data for advanced strategies
        
        Returns:
        list: Results of executed strategies
        """
        results = []
        
        # Apply loss protection first if enabled
        if hasattr(self.portfolio, 'check_and_apply_protection'):
            protection_results = self.portfolio.check_and_apply_protection({symbol: current_price})
            for action in protection_results:
                results.append({
                    'strategy_id': 'protection',
                    'symbol': action['symbol'],
                    'action': action['type'],
                    'reason': action['reason'],
                    'success': action['result']['success'],
                    'message': action['result']['message']
                })
                
            # If emergency liquidation occurred, skip other strategies
            if any(r['action'] == 'EMERGENCY_LIQUIDATION' for r in results):
                return results
        
        # Check if we're allowed to execute any more automated trades today
        protection_status = None
        if hasattr(self.portfolio, 'get_protection_status'):
            protection_status = self.portfolio.get_protection_status()
            if protection_status and protection_status.get('auto_trades_remaining', 0) <= 0:
                results.append({
                    'strategy_id': 'protection',
                    'symbol': symbol,
                    'action': 'NONE',
                    'reason': f"Maximum daily automated trades limit reached ({protection_status['max_auto_trades_per_day']})",
                    'success': False,
                    'message': "Trading suspended for today due to maximum trade limit"
                })
                return results
        
        # Process advanced strategies if historical data is available
        if historical_data is not None:
            # Run advanced strategies analysis
            advanced_results = AdvancedStrategies.analyze_all_strategies(historical_data)
            
            # Execute any advanced strategy if configured
            for strategy in self.active_strategies:
                if not strategy['active'] or strategy['symbol'] != symbol:
                    continue
                
                if strategy['type'] == 'advanced':
                    result = self._execute_advanced_strategy(
                        strategy, symbol, advanced_results, current_price, max_investment
                    )
                    results.append(result)
        
        # Execute regular strategies
        for strategy in self.active_strategies:
            if not strategy['active'] or strategy['symbol'] != symbol:
                continue
            
            if strategy['type'] == 'prediction_based':
                result = self._execute_prediction_strategy(
                    strategy, symbol, prediction_data, current_price, max_investment
                )
                results.append(result)
                
            elif strategy['type'] == 'threshold':
                result = self._execute_threshold_strategy(
                    strategy, symbol, current_price, max_investment
                )
                results.append(result)
                
            elif strategy['type'] == 'combined' and historical_data is not None:
                result = self._execute_combined_strategy(
                    strategy, symbol, prediction_data, historical_data, current_price, max_investment
                )
                results.append(result)
        
        return results
    
    def _execute_prediction_strategy(self, strategy, symbol, prediction_data, current_price, max_investment):
        """Execute a prediction-based trading strategy"""
        params = strategy['params']
        predicted_price = prediction_data['predicted_price']
        price_change_pct = ((predicted_price - current_price) / current_price) * 100
        
        buy_threshold = params.get('buy_threshold', 1.0)  # Default 1% predicted increase
        sell_threshold = params.get('sell_threshold', -1.0)  # Default 1% predicted decrease
        
        result = {
            'strategy_id': strategy['id'],
            'symbol': symbol,
            'action': 'NONE',
            'reason': '',
            'success': False,
            'message': ''
        }
        
        # Get current holdings
        portfolio_data = self.portfolio.get_portfolio()
        positions = portfolio_data['positions']
        current_shares = positions.get(symbol, {}).get('shares', 0)
        
        # Determine action based on prediction
        if price_change_pct >= buy_threshold:
            # BUY logic - if prediction is above buy threshold
            if current_shares > 0 and not params.get('allow_averaging', True):
                result['reason'] = f"Already holding {current_shares} shares and averaging is disabled"
            else:
                # Calculate shares to buy based on max investment
                shares_to_buy = max_investment // current_price
                if shares_to_buy > 0:
                    trade_result = self.portfolio.buy(symbol, shares_to_buy, current_price, automated=True)
                    result['action'] = 'BUY'
                    result['success'] = trade_result['success']
                    result['message'] = trade_result['message']
                    result['shares'] = shares_to_buy
                else:
                    result['reason'] = f"Price too high for minimum investment"
        
        elif price_change_pct <= sell_threshold:
            # SELL logic - if prediction is below sell threshold
            if current_shares > 0:
                # Sell all shares
                trade_result = self.portfolio.sell(symbol, current_shares, current_price, automated=True)
                result['action'] = 'SELL'
                result['success'] = trade_result['success']
                result['message'] = trade_result['message']
                result['shares'] = current_shares
            else:
                result['reason'] = f"No shares to sell"
        else:
            result['reason'] = f"Prediction of {price_change_pct:.2f}% not meeting thresholds (buy: {buy_threshold}%, sell: {sell_threshold}%)"
        
        # Record execution in strategy history
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        execution = {
            'timestamp': timestamp,
            'action': result['action'],
            'shares': result.get('shares', 0),
            'price': current_price,
            'prediction': predicted_price,
            'success': result['success'],
            'message': result['message'] or result['reason']
        }
        
        strategy['execution_history'].append(execution)
        strategy['last_executed'] = timestamp
        self._save_strategies()
        
        return result
    
    def _execute_threshold_strategy(self, strategy, symbol, current_price, max_investment):
        """Execute a threshold-based trading strategy"""
        params = strategy['params']
        
        result = {
            'strategy_id': strategy['id'],
            'symbol': symbol,
            'action': 'NONE',
            'reason': '',
            'success': False,
            'message': ''
        }
        
        # Get current holdings
        portfolio_data = self.portfolio.get_portfolio()
        positions = portfolio_data['positions']
        
        # If we have position for this symbol
        if symbol in positions:
            position = positions[symbol]
            current_shares = position['shares']
            avg_price = position['average_price']
            price_change_pct = ((current_price - avg_price) / avg_price) * 100
            
            # Check if we should sell based on profit or loss thresholds
            take_profit = params.get('take_profit', 5.0)  # Default 5% profit
            stop_loss = params.get('stop_loss', -5.0)  # Default 5% loss
            
            if price_change_pct >= take_profit:
                # Sell for profit
                trade_result = self.portfolio.sell(symbol, current_shares, current_price, automated=True)
                result['action'] = 'SELL'
                result['success'] = trade_result['success']
                result['message'] = f"Take profit triggered at {price_change_pct:.2f}%. " + trade_result['message']
                result['shares'] = current_shares
            
            elif price_change_pct <= stop_loss:
                # Sell to limit loss
                trade_result = self.portfolio.sell(symbol, current_shares, current_price, automated=True)
                result['action'] = 'SELL'
                result['success'] = trade_result['success']
                result['message'] = f"Stop loss triggered at {price_change_pct:.2f}%. " + trade_result['message']
                result['shares'] = current_shares
            
            else:
                result['reason'] = f"Current change {price_change_pct:.2f}% not meeting thresholds (profit: {take_profit}%, loss: {stop_loss}%)"
        
        else:
            # We don't have a position - check if we should buy
            buy_price = params.get('buy_price', None)
            
            if buy_price is not None and current_price <= buy_price:
                # Buy at or below target price
                shares_to_buy = max_investment // current_price
                
                if shares_to_buy > 0:
                    trade_result = self.portfolio.buy(symbol, shares_to_buy, current_price, automated=True)
                    result['action'] = 'BUY'
                    result['success'] = trade_result['success']
                    result['message'] = f"Buy price target {buy_price:.2f} reached. " + trade_result['message']
                    result['shares'] = shares_to_buy
                else:
                    result['reason'] = f"Price too high for minimum investment"
            else:
                result['reason'] = f"Current price ${current_price:.2f} not below buy target of ${buy_price:.2f}" if buy_price else "No buy price target set"
        
        # Record execution in strategy history
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        execution = {
            'timestamp': timestamp,
            'action': result['action'],
            'shares': result.get('shares', 0),
            'price': current_price,
            'success': result['success'],
            'message': result['message'] or result['reason']
        }
        
        strategy['execution_history'].append(execution)
        strategy['last_executed'] = timestamp
        self._save_strategies()
        
        return result
        
    def _execute_advanced_strategy(self, strategy, symbol, strategy_results, current_price, max_investment):
        """
        Execute an advanced trading strategy based on technical indicators
        
        Parameters:
        strategy (dict): Strategy configuration
        symbol (str): Stock symbol
        strategy_results (list): Results from advanced strategies analysis
        current_price (float): Current stock price
        max_investment (float): Maximum investment amount
        
        Returns:
        dict: Result of the strategy execution
        """
        params = strategy['params']
        strategy_type = params.get('strategy_type', 'combined')
        
        result = {
            'strategy_id': strategy['id'],
            'symbol': symbol,
            'action': 'NONE',
            'reason': '',
            'success': False,
            'message': ''
        }
        
        # Get current holdings
        portfolio_data = self.portfolio.get_portfolio()
        positions = portfolio_data['positions']
        current_shares = positions.get(symbol, {}).get('shares', 0)
        
        # Find the matching strategy result
        matching_result = None
        for res in strategy_results:
            if res['strategy'] == strategy_type:
                matching_result = res
                break
        
        if not matching_result:
            result['reason'] = f"No results for strategy type '{strategy_type}'"
            return result
        
        # Determine action based on the strategy signal
        signal = matching_result['signal']
        
        if signal in ['BUY', 'BULLISH']:
            # Only buy if confidence is high enough
            confidence_threshold = params.get('confidence_threshold', 0.7)
            
            if 'metrics' in matching_result and 'trend_strength' in matching_result['metrics']:
                trend_strength = matching_result['metrics']['trend_strength'] / 100  # Normalize to 0-1
                if trend_strength < confidence_threshold:
                    result['reason'] = f"Signal is {signal} but confidence ({trend_strength:.2f}) below threshold ({confidence_threshold})"
                    return result
            
            # Check if we already have a position
            if current_shares > 0 and not params.get('allow_adding', False):
                result['reason'] = f"Already holding {current_shares} shares and adding is disabled"
            else:
                # Calculate shares to buy
                shares_to_buy = max_investment // current_price
                if shares_to_buy > 0:
                    trade_result = self.portfolio.buy(symbol, shares_to_buy, current_price, automated=True)
                    result['action'] = 'BUY'
                    result['success'] = trade_result['success']
                    result['message'] = f"{matching_result['description']}. " + trade_result['message']
                    result['shares'] = shares_to_buy
                else:
                    result['reason'] = f"Price too high for minimum investment"
        
        elif signal in ['SELL', 'BEARISH']:
            # Only sell if we have shares
            if current_shares > 0:
                # Sell all or part of position
                sell_percent = params.get('sell_percent', 100)
                shares_to_sell = current_shares if sell_percent >= 100 else int(current_shares * (sell_percent / 100))
                
                if shares_to_sell > 0:
                    trade_result = self.portfolio.sell(symbol, shares_to_sell, current_price, automated=True)
                    result['action'] = 'SELL'
                    result['success'] = trade_result['success']
                    result['message'] = f"{matching_result['description']}. " + trade_result['message']
                    result['shares'] = shares_to_sell
                else:
                    result['reason'] = f"Calculated shares to sell is 0"
            else:
                result['reason'] = f"No shares to sell"
        else:
            result['reason'] = f"Signal '{signal}' does not indicate action. {matching_result['description']}"
        
        # Record execution in strategy history
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        execution = {
            'timestamp': timestamp,
            'action': result['action'],
            'shares': result.get('shares', 0),
            'price': current_price,
            'signal': signal,
            'strategy_type': strategy_type,
            'metrics': matching_result.get('metrics', {}),
            'success': result['success'],
            'message': result['message'] or result['reason']
        }
        
        strategy['execution_history'].append(execution)
        strategy['last_executed'] = timestamp
        self._save_strategies()
        
        return result
    
    def _execute_combined_strategy(self, strategy, symbol, prediction_data, historical_data, current_price, max_investment):
        """
        Execute a combined strategy using both prediction and technical indicators
        
        Parameters:
        strategy (dict): Strategy configuration
        symbol (str): Stock symbol
        prediction_data (dict): Prediction results
        historical_data (pandas.DataFrame): Historical price data
        current_price (float): Current stock price
        max_investment (float): Maximum investment amount
        
        Returns:
        dict: Result of the strategy execution
        """
        params = strategy['params']
        
        result = {
            'strategy_id': strategy['id'],
            'symbol': symbol,
            'action': 'NONE',
            'reason': '',
            'success': False,
            'message': ''
        }
        
        # Get current holdings
        portfolio_data = self.portfolio.get_portfolio()
        positions = portfolio_data['positions']
        current_shares = positions.get(symbol, {}).get('shares', 0)
        
        # Get prediction signal
        predicted_price = prediction_data['predicted_price']
        price_change_pct = ((predicted_price - current_price) / current_price) * 100
        prediction_signal = 'HOLD'
        
        buy_threshold = params.get('prediction_buy_threshold', 1.0) 
        sell_threshold = params.get('prediction_sell_threshold', -1.0)
        
        if price_change_pct >= buy_threshold:
            prediction_signal = 'BUY'
        elif price_change_pct <= sell_threshold:
            prediction_signal = 'SELL'
        
        # Get technical signals
        tech_results = AdvancedStrategies.analyze_all_strategies(historical_data)
        tech_signals = [r['signal'] for r in tech_results]
        
        # Calculate combined signal
        combined_signal = AdvancedStrategies.calculate_combined_signal(tech_results)
        
        # Determine final decision based on strategy mode
        decision_mode = params.get('decision_mode', 'consensus')  # Options: consensus, prediction_priority, technical_priority
        
        final_signal = 'HOLD'
        
        if decision_mode == 'consensus':
            # Need agreement between prediction and technical
            if prediction_signal == 'BUY' and combined_signal == 'BUY':
                final_signal = 'BUY'
            elif prediction_signal == 'SELL' and combined_signal == 'SELL':
                final_signal = 'SELL'
        elif decision_mode == 'prediction_priority':
            # Prediction takes priority, but technical must not strongly contradict
            if prediction_signal == 'BUY' and combined_signal != 'SELL':
                final_signal = 'BUY'
            elif prediction_signal == 'SELL' and combined_signal != 'BUY':
                final_signal = 'SELL'
        elif decision_mode == 'technical_priority':
            # Technical takes priority, but prediction must not strongly contradict
            if combined_signal == 'BUY' and prediction_signal != 'SELL':
                final_signal = 'BUY'
            elif combined_signal == 'SELL' and prediction_signal != 'BUY':
                final_signal = 'SELL'
        
        # Execute based on final signal
        if final_signal == 'BUY':
            if current_shares > 0 and not params.get('allow_adding', False):
                result['reason'] = f"Already holding {current_shares} shares and adding is disabled"
            else:
                # Calculate shares to buy
                shares_to_buy = max_investment // current_price
                if shares_to_buy > 0:
                    trade_result = self.portfolio.buy(symbol, shares_to_buy, current_price, automated=True)
                    result['action'] = 'BUY'
                    result['success'] = trade_result['success']
                    result['message'] = f"Combined strategy: Prediction {price_change_pct:.2f}%, Technical signals: {combined_signal}. " + trade_result['message']
                    result['shares'] = shares_to_buy
                else:
                    result['reason'] = f"Price too high for minimum investment"
        
        elif final_signal == 'SELL':
            if current_shares > 0:
                # Sell all shares
                trade_result = self.portfolio.sell(symbol, current_shares, current_price, automated=True)
                result['action'] = 'SELL'
                result['success'] = trade_result['success']
                result['message'] = f"Combined strategy: Prediction {price_change_pct:.2f}%, Technical signals: {combined_signal}. " + trade_result['message']
                result['shares'] = current_shares
            else:
                result['reason'] = f"No shares to sell"
        else:
            result['reason'] = f"No action taken. Prediction: {prediction_signal} ({price_change_pct:.2f}%), Technical: {combined_signal}"
        
        # Record execution in strategy history
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        execution = {
            'timestamp': timestamp,
            'action': result['action'],
            'shares': result.get('shares', 0),
            'price': current_price,
            'prediction': predicted_price,
            'prediction_signal': prediction_signal,
            'technical_signal': combined_signal,
            'final_signal': final_signal,
            'success': result['success'],
            'message': result['message'] or result['reason']
        }
        
        strategy['execution_history'].append(execution)
        strategy['last_executed'] = timestamp
        self._save_strategies()
        
        return result