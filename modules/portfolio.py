import pandas as pd
import datetime
import json
import os
import uuid
import time

class Portfolio:
    def __init__(self):
        self.portfolio_file = 'data/portfolio.json'
        self.transactions_file = 'data/transactions.json'
        self.protection_file = 'data/protection_settings.json'
        self.demo_balance = 100000.0  # Starting with $100,000 demo money
        self.portfolio = self._load_portfolio()
        self.transactions = self._load_transactions()
        self.protection_settings = self._load_protection_settings()
        self._ensure_data_directory()
        
    def _ensure_data_directory(self):
        """Ensure the data directory exists"""
        os.makedirs('data', exist_ok=True)
        
        # Create portfolio file if it doesn't exist
        if not os.path.exists(self.portfolio_file):
            self._save_portfolio()
        
        # Create transactions file if it doesn't exist
        if not os.path.exists(self.transactions_file):
            self._save_transactions()
            
        # Create protection settings file if it doesn't exist
        if not os.path.exists(self.protection_file):
            self._save_protection_settings()
    
    def _load_portfolio(self):
        """Load portfolio from file or initialize a new one"""
        try:
            if os.path.exists(self.portfolio_file):
                with open(self.portfolio_file, 'r') as f:
                    return json.load(f)
            return {
                'cash': self.demo_balance,
                'positions': {},
                'total_value': self.demo_balance,
                'initial_value': self.demo_balance
            }
        except Exception as e:
            print(f"Error loading portfolio: {e}")
            return {
                'cash': self.demo_balance,
                'positions': {},
                'total_value': self.demo_balance,
                'initial_value': self.demo_balance
            }
    
    def _load_transactions(self):
        """Load transactions from file or initialize a new one"""
        try:
            if os.path.exists(self.transactions_file):
                with open(self.transactions_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading transactions: {e}")
            return []
    
    def _save_portfolio(self):
        """Save portfolio to file"""
        try:
            with open(self.portfolio_file, 'w') as f:
                json.dump(self.portfolio, f, indent=4)
        except Exception as e:
            print(f"Error saving portfolio: {e}")
    
    def _save_transactions(self):
        """Save transactions to file"""
        try:
            with open(self.transactions_file, 'w') as f:
                json.dump(self.transactions, f, indent=4)
        except Exception as e:
            print(f"Error saving transactions: {e}")
    
    def get_portfolio(self):
        """Get the current portfolio"""
        return self.portfolio
    
    def get_positions(self):
        """Get the current positions as a DataFrame"""
        if not self.portfolio['positions']:
            return pd.DataFrame(columns=['Symbol', 'Shares', 'Cost Basis', 'Current Value', 'Profit/Loss', 'P/L %'])
        
        positions = []
        for symbol, data in self.portfolio['positions'].items():
            positions.append({
                'Symbol': symbol,
                'Shares': data['shares'],
                'Cost Basis': data['cost_basis'],
                'Current Value': data['current_value'],
                'Profit/Loss': data['current_value'] - data['cost_basis'],
                'P/L %': ((data['current_value'] - data['cost_basis']) / data['cost_basis']) * 100 if data['cost_basis'] > 0 else 0
            })
        
        return pd.DataFrame(positions)
    
    def get_transactions(self):
        """Get the transaction history as a DataFrame"""
        if not self.transactions:
            return pd.DataFrame(columns=['Date', 'Type', 'Symbol', 'Shares', 'Price', 'Total', 'Status'])
        
        return pd.DataFrame(self.transactions)
    
    def update_position_value(self, symbol, current_price):
        """Update the current value of a position based on the current price"""
        if symbol in self.portfolio['positions']:
            position = self.portfolio['positions'][symbol]
            position['current_value'] = position['shares'] * current_price
            self._update_portfolio_value()
            self._save_portfolio()
    
    def update_all_positions(self, price_data):
        """Update the value of all positions with current prices"""
        if not self.portfolio['positions']:
            return
        
        for symbol in self.portfolio['positions']:
            if symbol in price_data:
                self.update_position_value(symbol, price_data[symbol])
    
    def _update_portfolio_value(self):
        """Update the total value of the portfolio"""
        positions_value = sum(position['current_value'] for position in self.portfolio['positions'].values())
        self.portfolio['total_value'] = self.portfolio['cash'] + positions_value
    
    def buy(self, symbol, shares, price, automated=False):
        """
        Buy shares of a stock
        
        Parameters:
        symbol (str): Stock symbol
        shares (float): Number of shares to buy
        price (float): Current price per share
        automated (bool): Whether this is an automated trade
        
        Returns:
        dict: Result of the transaction (success, message)
        """
        total_cost = shares * price
        
        # Check if we have enough cash
        if total_cost > self.portfolio['cash']:
            return {
                'success': False,
                'message': f"Insufficient funds. Required: ${total_cost:.2f}, Available: ${self.portfolio['cash']:.2f}"
            }
        
        # Add or update position
        if symbol in self.portfolio['positions']:
            # Update existing position
            position = self.portfolio['positions'][symbol]
            total_shares = position['shares'] + shares
            new_cost_basis = position['cost_basis'] + total_cost
            
            position['shares'] = total_shares
            position['average_price'] = new_cost_basis / total_shares
            position['cost_basis'] = new_cost_basis
            position['current_value'] = total_shares * price
        else:
            # Create new position
            self.portfolio['positions'][symbol] = {
                'shares': shares,
                'average_price': price,
                'cost_basis': total_cost,
                'current_value': total_cost
            }
        
        # Deduct cash
        self.portfolio['cash'] -= total_cost
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Save changes
        self._save_portfolio()
        
        # Record transaction
        transaction_id = str(uuid.uuid4())
        transaction = {
            'id': transaction_id,
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'BUY',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'total': total_cost,
            'status': 'COMPLETED',
            'automated': automated
        }
        
        self.transactions.append(transaction)
        self._save_transactions()
        
        return {
            'success': True,
            'message': f"Successfully bought {shares} shares of {symbol} at ${price:.2f} per share.",
            'transaction': transaction
        }
    
    def sell(self, symbol, shares, price, automated=False):
        """
        Sell shares of a stock
        
        Parameters:
        symbol (str): Stock symbol
        shares (float): Number of shares to sell
        price (float): Current price per share
        automated (bool): Whether this is an automated trade
        
        Returns:
        dict: Result of the transaction (success, message)
        """
        # Check if we have the position
        if symbol not in self.portfolio['positions']:
            return {
                'success': False,
                'message': f"You don't own any shares of {symbol}."
            }
        
        position = self.portfolio['positions'][symbol]
        
        # Check if we have enough shares
        if shares > position['shares']:
            return {
                'success': False,
                'message': f"Insufficient shares. Requested: {shares}, Available: {position['shares']}"
            }
        
        # Calculate transaction
        total_value = shares * price
        
        # Update position
        remaining_shares = position['shares'] - shares
        
        if remaining_shares > 0:
            # Update position
            proportion_sold = shares / position['shares']
            cost_basis_sold = position['cost_basis'] * proportion_sold
            
            position['shares'] = remaining_shares
            position['cost_basis'] -= cost_basis_sold
            position['current_value'] = remaining_shares * price
        else:
            # Remove position if all shares are sold
            del self.portfolio['positions'][symbol]
        
        # Add cash
        self.portfolio['cash'] += total_value
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Save changes
        self._save_portfolio()
        
        # Record transaction
        transaction_id = str(uuid.uuid4())
        transaction = {
            'id': transaction_id,
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'SELL',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'total': total_value,
            'status': 'COMPLETED',
            'automated': automated
        }
        
        self.transactions.append(transaction)
        self._save_transactions()
        
        return {
            'success': True,
            'message': f"Successfully sold {shares} shares of {symbol} at ${price:.2f} per share.",
            'transaction': transaction
        }
    
    def reset_portfolio(self):
        """Reset the portfolio to initial state"""
        self.portfolio = {
            'cash': self.demo_balance,
            'positions': {},
            'total_value': self.demo_balance,
            'initial_value': self.demo_balance
        }
        self.transactions = []
        self._save_portfolio()
        self._save_transactions()
        
        return {
            'success': True,
            'message': f"Portfolio reset to initial balance of ${self.demo_balance:.2f}"
        }
    
    def _load_protection_settings(self):
        """Load protection settings from file or initialize with defaults"""
        try:
            if os.path.exists(self.protection_file):
                with open(self.protection_file, 'r') as f:
                    return json.load(f)
            
            # Default protection settings
            return {
                'global': {
                    'enabled': True,
                    'max_loss_percent': 5.0,  # Maximum portfolio loss percentage
                    'max_position_percent': 20.0,  # Maximum percentage of portfolio in one position
                    'trailing_stop_percent': 3.0,  # Default trailing stop loss percentage
                    'volatility_protection': True,  # Enable extra protection for high volatility
                    'max_auto_trades_per_day': 10  # Limit automated trades
                },
                'positions': {}  # Symbol-specific settings
            }
        except Exception as e:
            print(f"Error loading protection settings: {e}")
            return {
                'global': {
                    'enabled': True,
                    'max_loss_percent': 5.0,
                    'max_position_percent': 20.0,
                    'trailing_stop_percent': 3.0,
                    'volatility_protection': True,
                    'max_auto_trades_per_day': 10
                },
                'positions': {}
            }
    
    def _save_protection_settings(self):
        """Save protection settings to file"""
        try:
            with open(self.protection_file, 'w') as f:
                json.dump(self.protection_settings, f, indent=4)
        except Exception as e:
            print(f"Error saving protection settings: {e}")
    
    def get_protection_settings(self):
        """Get the current protection settings"""
        return self.protection_settings
    
    def update_protection_settings(self, settings):
        """
        Update global protection settings
        
        Parameters:
        settings (dict): New protection settings
        
        Returns:
        dict: Result of the update
        """
        # Update global settings
        for key, value in settings.items():
            if key in self.protection_settings['global']:
                self.protection_settings['global'][key] = value
        
        self._save_protection_settings()
        
        return {
            'success': True,
            'message': 'Protection settings updated successfully'
        }
    
    def set_position_protection(self, symbol, settings):
        """
        Set position-specific protection settings
        
        Parameters:
        symbol (str): Stock symbol
        settings (dict): Protection settings for this position
        
        Returns:
        dict: Result of the update
        """
        # Ensure the symbol exists in portfolio
        if symbol not in self.portfolio['positions'] and not settings.get('allow_pre_position', False):
            return {
                'success': False,
                'message': f"No position exists for {symbol}. Enable 'allow_pre_position' to set protection for future positions."
            }
        
        # Update or create position settings
        if symbol not in self.protection_settings['positions']:
            self.protection_settings['positions'][symbol] = {}
        
        # Update settings
        for key, value in settings.items():
            self.protection_settings['positions'][symbol][key] = value
        
        self._save_protection_settings()
        
        return {
            'success': True,
            'message': f"Protection settings for {symbol} updated successfully"
        }
    
    def check_and_apply_protection(self, price_data):
        """
        Check positions against protection settings and execute protective actions
        
        Parameters:
        price_data (dict): Current prices for symbols in portfolio
        
        Returns:
        list: Actions taken for protection
        """
        if not self.protection_settings['global']['enabled']:
            return []
        
        actions = []
        
        # Check overall portfolio loss limit
        current_value = self.portfolio['total_value']
        initial_value = self.portfolio['initial_value']
        total_loss_percent = ((initial_value - current_value) / initial_value) * 100
        
        if total_loss_percent >= self.protection_settings['global']['max_loss_percent']:
            # Portfolio is beyond max loss threshold - liquidate all positions
            for symbol in list(self.portfolio['positions'].keys()):
                if symbol in price_data:
                    position = self.portfolio['positions'][symbol]
                    result = self.sell(symbol, position['shares'], price_data[symbol], automated=True)
                    
                    if result['success']:
                        actions.append({
                            'type': 'EMERGENCY_LIQUIDATION',
                            'symbol': symbol,
                            'reason': f"Portfolio loss of {total_loss_percent:.2f}% exceeded maximum allowed {self.protection_settings['global']['max_loss_percent']}%",
                            'result': result
                        })
            
            return actions
        
        # Check individual positions
        for symbol, position in list(self.portfolio['positions'].items()):
            if symbol not in price_data:
                continue
            
            current_price = price_data[symbol]
            position_settings = self.protection_settings['positions'].get(symbol, {})
            
            # Check position size limit
            position_percent = (position['current_value'] / current_value) * 100
            max_position_percent = position_settings.get('max_position_percent', 
                                                        self.protection_settings['global']['max_position_percent'])
            
            if position_percent > max_position_percent:
                # Position is too large - reduce position
                excess_percent = position_percent - max_position_percent
                excess_value = (excess_percent / 100) * current_value
                shares_to_sell = excess_value / current_price
                
                # Round down to ensure we don't sell too many shares
                shares_to_sell = min(int(shares_to_sell), position['shares'])
                
                if shares_to_sell > 0:
                    result = self.sell(symbol, shares_to_sell, current_price, automated=True)
                    
                    if result['success']:
                        actions.append({
                            'type': 'POSITION_SIZE_REDUCTION',
                            'symbol': symbol,
                            'reason': f"Position size of {position_percent:.2f}% exceeded maximum allowed {max_position_percent}%",
                            'result': result
                        })
            
            # Get average price for calculations
            avg_price = position['average_price']
            
            # Check stop loss
            stop_loss_percent = position_settings.get('stop_loss_percent', None)
            if stop_loss_percent is not None:
                loss_percent = ((avg_price - current_price) / avg_price) * 100
                
                if loss_percent >= stop_loss_percent:
                    # Stop loss triggered - sell position
                    result = self.sell(symbol, position['shares'], current_price, automated=True)
                    
                    if result['success']:
                        actions.append({
                            'type': 'STOP_LOSS',
                            'symbol': symbol,
                            'reason': f"Stop loss triggered at {loss_percent:.2f}% loss (limit: {stop_loss_percent}%)",
                            'result': result
                        })
            
            # Check trailing stop
            trailing_stop_percent = position_settings.get('trailing_stop_percent', 
                                                        self.protection_settings['global']['trailing_stop_percent'])
            highest_price = position_settings.get('highest_price', avg_price)
            
            # Update highest price if current price is higher
            if current_price > highest_price:
                position_settings['highest_price'] = current_price
                self._save_protection_settings()
            
            # Check if price has fallen below trailing stop
            trailing_stop_price = highest_price * (1 - (trailing_stop_percent / 100))
            
            if current_price <= trailing_stop_price:
                # Trailing stop triggered - sell position
                result = self.sell(symbol, position['shares'], current_price, automated=True)
                
                if result['success']:
                    actions.append({
                        'type': 'TRAILING_STOP',
                        'symbol': symbol,
                        'reason': f"Trailing stop triggered at ${current_price:.2f} ({trailing_stop_percent}% below highest price of ${highest_price:.2f})",
                        'result': result
                    })
        
        return actions
    
    def get_protection_status(self):
        """
        Get the current protection status for the portfolio
        
        Returns:
        dict: Protection status information
        """
        # Calculate overall portfolio metrics for protection
        current_value = self.portfolio['total_value']
        initial_value = self.portfolio['initial_value']
        total_loss_percent = ((initial_value - current_value) / initial_value) * 100 if initial_value > current_value else 0
        
        # Count today's automated trades
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        auto_trades_today = sum(1 for t in self.transactions 
                              if t['date'].startswith(today) and t.get('automated', False))
        
        # Get position-specific protection status
        position_status = []
        for symbol, position in self.portfolio['positions'].items():
            position_percent = (position['current_value'] / current_value) * 100 if current_value > 0 else 0
            position_settings = self.protection_settings['positions'].get(symbol, {})
            
            position_status.append({
                'symbol': symbol,
                'value': position['current_value'],
                'percent_of_portfolio': position_percent,
                'has_stop_loss': 'stop_loss_percent' in position_settings,
                'stop_loss_percent': position_settings.get('stop_loss_percent', None),
                'trailing_stop_percent': position_settings.get('trailing_stop_percent', 
                                                            self.protection_settings['global']['trailing_stop_percent']),
                'highest_price': position_settings.get('highest_price', position['average_price'])
            })
        
        return {
            'global_protection_enabled': self.protection_settings['global']['enabled'],
            'portfolio_loss_percent': total_loss_percent,
            'max_loss_percent': self.protection_settings['global']['max_loss_percent'],
            'auto_trades_today': auto_trades_today,
            'max_auto_trades_per_day': self.protection_settings['global']['max_auto_trades_per_day'],
            'auto_trades_remaining': max(0, self.protection_settings['global']['max_auto_trades_per_day'] - auto_trades_today),
            'positions': position_status
        }