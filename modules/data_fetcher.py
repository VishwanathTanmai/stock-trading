import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_historical_data(ticker, period="6mo", interval="1d"):
    """
    Fetch historical stock data from Yahoo Finance
    
    Parameters:
    ticker (str): Stock symbol
    period (str): Time period to fetch (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    interval (str): Data interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
    
    Returns:
    pandas.DataFrame: Historical stock data
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        # Check if data is empty
        if data.empty:
            return None
        
        # Reset index to have Date as a column
        data = data.reset_index()
        
        # Convert datetime to date string if interval is daily or longer
        if interval in ['1d', '5d', '1wk', '1mo', '3mo']:
            data['Date'] = pd.to_datetime(data['Date']).dt.date
        
        return data
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        return None

def get_company_info(ticker):
    """
    Fetch company information for a given ticker
    
    Parameters:
    ticker (str): Stock symbol
    
    Returns:
    dict: Company information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception as e:
        print(f"Error fetching company info for {ticker}: {e}")
        return {}

def get_current_price(ticker):
    """
    Get the current price of a stock
    
    Parameters:
    ticker (str): Stock symbol
    
    Returns:
    float: Current stock price
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        return 0
    except Exception as e:
        print(f"Error fetching current price for {ticker}: {e}")
        return 0

def get_financial_data(ticker):
    """
    Get key financial data for a stock
    
    Parameters:
    ticker (str): Stock symbol
    
    Returns:
    pandas.DataFrame: Financial data
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get financial statements
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        cash_flow = stock.cashflow
        
        # If no data available, return None
        if balance_sheet.empty and income_stmt.empty and cash_flow.empty:
            return None
        
        # Create financial ratios dataframe
        # Extract the most recent data (first column)
        financial_data = {}
        
        # From Income Statement
        if not income_stmt.empty:
            financial_data['Total Revenue'] = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else np.nan
            financial_data['Net Income'] = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else np.nan
            financial_data['Gross Profit'] = income_stmt.loc['Gross Profit'].iloc[0] if 'Gross Profit' in income_stmt.index else np.nan
            financial_data['EBITDA'] = income_stmt.loc['EBITDA'].iloc[0] if 'EBITDA' in income_stmt.index else np.nan
        
        # From Balance Sheet
        if not balance_sheet.empty:
            financial_data['Total Assets'] = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else np.nan
            financial_data['Total Liabilities'] = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else np.nan
            financial_data['Total Equity'] = balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[0] if 'Total Equity Gross Minority Interest' in balance_sheet.index else np.nan
        
        # From Cash Flow
        if not cash_flow.empty:
            financial_data['Operating Cash Flow'] = cash_flow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cash_flow.index else np.nan
            financial_data['Free Cash Flow'] = cash_flow.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cash_flow.index else np.nan
        
        # Calculate ratios
        if 'Total Assets' in financial_data and 'Total Liabilities' in financial_data:
            financial_data['Debt-to-Asset Ratio'] = financial_data['Total Liabilities'] / financial_data['Total Assets']
        
        if 'Net Income' in financial_data and 'Total Equity' in financial_data and financial_data['Total Equity'] != 0:
            financial_data['Return on Equity (ROE)'] = financial_data['Net Income'] / financial_data['Total Equity']
        
        if 'Net Income' in financial_data and 'Total Assets' in financial_data and financial_data['Total Assets'] != 0:
            financial_data['Return on Assets (ROA)'] = financial_data['Net Income'] / financial_data['Total Assets']
        
        # Convert to DataFrame
        df_financial = pd.DataFrame.from_dict(financial_data, orient='index', columns=['Value'])
        
        # Format large numbers to be more readable
        df_financial['Value'] = df_financial['Value'].apply(lambda x: f"${x/1_000_000_000:.2f}B" if isinstance(x, (int, float)) and abs(x) >= 1_000_000_000 
                                                    else f"${x/1_000_000:.2f}M" if isinstance(x, (int, float)) and abs(x) >= 1_000_000 
                                                    else f"${x:.2f}" if isinstance(x, (int, float)) 
                                                    else x)
        
        return df_financial
    except Exception as e:
        print(f"Error fetching financial data for {ticker}: {e}")
        return None
