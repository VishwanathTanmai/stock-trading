import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Import custom modules
import modules.data_fetcher as data_fetcher
import modules.visualizations as visualizations
import modules.predictions as predictions
from modules.portfolio import Portfolio
from modules.auto_trader import AutoTrader

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="Stock Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for portfolio and autotrader
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = Portfolio()
    
if 'auto_trader' not in st.session_state:
    st.session_state.auto_trader = AutoTrader(st.session_state.portfolio)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0D47A1;
    }
    .subheader {
        font-size: 1.5rem;
        color: #424242;
    }
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .profit {
        color: #2E7D32 !important;
    }
    .loss {
        color: #C62828 !important;
    }
    .info-box {
        background-color: #e3f2fd;
        border-radius: 5px;
        padding: 10px;
        border-left: 5px solid #0D47A1;
    }
    .trade-box {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #ddd;
    }
    .portfolio-summary {
        font-size: 1.2rem;
        padding: 10px;
        background-color: #f5f5f5;
        border-radius: 5px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("📈 Stock Analysis & Trading")
st.markdown("Analyze stocks, manage portfolio, and set up automated trading strategies with demo money by vishwanath tanmai")

# Main navigation
tab_market, tab_portfolio, tab_trading = st.tabs(["Market Analysis", "Portfolio", "Auto-Trading"])

# Global variables to store data across tabs
current_stock_data = {}

# -------------- MARKET ANALYSIS TAB --------------
with tab_market:
    # Sidebar for stock selection and timeframe
    with st.sidebar:
        st.header("Market Analysis Settings")
        
        # Stock search box
        stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, GOOGL)", value="AAPL").upper()
        
        # Timeframe selection
        timeframe_options = {
            "1 Day": "1d",
            "5 Days": "5d",
            "1 Month": "1mo",
            "3 Months": "3mo", 
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y",
            "10 Years": "10y",
            "Year to Date": "ytd",
            "Max": "max"
        }
        
        selected_timeframe = st.selectbox(
            "Select Timeframe",
            list(timeframe_options.keys()),
            index=4
        )
        
        timeframe = timeframe_options[selected_timeframe]
        
        # Interval selection based on timeframe
        interval_options = {
            "1 Day": "5m",
            "5 Days": "15m",
            "1 Month": "1h",
            "3 Months": "1d",
            "6 Months": "1d",
            "1 Year": "1d",
            "2 Years": "1d",
            "5 Years": "1wk",
            "10 Years": "1mo",
            "Year to Date": "1d",
            "Max": "1mo"
        }
        
        interval = interval_options[selected_timeframe]
        
        st.caption("Data provided by Yahoo Finance")
        
        # Add quick trade buttons in sidebar
        st.markdown("---")
        st.subheader("Quick Trade")
        
        # Trade form
        with st.form("quick_trade_form"):
            trade_shares = st.number_input("Shares", min_value=1, step=1, value=10)
            
            col1, col2 = st.columns(2)
            with col1:
                buy_button = st.form_submit_button("Buy", use_container_width=True)
            with col2:
                sell_button = st.form_submit_button("Sell", use_container_width=True)

    # Main content
    try:
        # Loading spinner while fetching data
        with st.spinner(f"Fetching data for {stock_symbol}..."):
            # Fetch historical data
            hist_data = data_fetcher.get_historical_data(stock_symbol, timeframe, interval)
            
            # Fetch company info - with fallback for invalid symbols
            if hist_data is None or hist_data.empty:
                st.error(f"No data available for {stock_symbol}. Please check the symbol and try again.")
                st.info("Please try a different stock symbol or timeframe.")
                st.stop()  # Stop execution if no data
                
            company_info = data_fetcher.get_company_info(stock_symbol)
            current_price = data_fetcher.get_current_price(stock_symbol)
            
            # Store in global variable for other tabs
            current_stock_data = {
                'symbol': stock_symbol,
                'price': current_price,
                'company_info': company_info,
                'hist_data': hist_data
            }
            
            # Process quick trade if submitted
            if buy_button:
                result = st.session_state.portfolio.buy(stock_symbol, trade_shares, current_price)
                if result['success']:
                    st.sidebar.success(result['message'])
                else:
                    st.sidebar.error(result['message'])
                    
            if sell_button:
                result = st.session_state.portfolio.sell(stock_symbol, trade_shares, current_price)
                if result['success']:
                    st.sidebar.success(result['message'])
                else:
                    st.sidebar.error(result['message'])
            
            # If we have data, display it
            if hist_data is not None and not hist_data.empty:
                # Display company name and basic info
                st.header(f"{company_info.get('shortName', stock_symbol)}")
                st.subheader(f"{company_info.get('exchange', '')} : {stock_symbol}")
                
                # Create tabs for different analysis sections
                analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs(["Overview", "Financial Data", "Prediction", "About"])
                
                with analysis_tab1:
                    # Top metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Current Price", 
                            f"${current_price:.2f}", 
                            f"{company_info.get('regularMarketChangePercent', 0):.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Previous Close", 
                            f"${company_info.get('regularMarketPreviousClose', 0):.2f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Day Range", 
                            f"${company_info.get('regularMarketDayLow', 0):.2f} - ${company_info.get('regularMarketDayHigh', 0):.2f}"
                        )
                    
                    with col4:
                        st.metric(
                            "52 Week Range", 
                            f"${company_info.get('fiftyTwoWeekLow', 0):.2f} - ${company_info.get('fiftyTwoWeekHigh', 0):.2f}"
                        )
                    
                    # Stock price chart
                    st.subheader(f"{stock_symbol} Stock Price Chart")
                    price_chart = visualizations.create_price_chart(hist_data, stock_symbol, timeframe)
                    st.plotly_chart(price_chart, use_container_width=True)
                    
                    # Volume Chart
                    st.subheader("Trading Volume")
                    volume_chart = visualizations.create_volume_chart(hist_data, stock_symbol)
                    st.plotly_chart(volume_chart, use_container_width=True)
                    
                    # Download data button
                    csv = hist_data.to_csv(index=True)
                    st.download_button(
                        label="Download Data as CSV",
                        data=csv,
                        file_name=f"{stock_symbol}_{timeframe}_data.csv",
                        mime="text/csv",
                    )
                
                with analysis_tab2:
                    # Financial Data Section
                    st.header("Key Financial Metrics")
                    
                    # Key Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Market Cap", f"${company_info.get('marketCap', 0) / 1_000_000_000:.2f}B")
                        st.metric("P/E Ratio", f"{company_info.get('trailingPE', 0):.2f}")
                        st.metric("EPS (TTM)", f"${company_info.get('trailingEps', 0):.2f}")
                    
                    with col2:
                        st.metric("Forward P/E", f"{company_info.get('forwardPE', 0):.2f}")
                        st.metric("Dividend Yield", f"{company_info.get('dividendYield', 0) * 100:.2f}%")
                        st.metric("Beta", f"{company_info.get('beta', 0):.2f}")
                    
                    with col3:
                        st.metric("52W High", f"${company_info.get('fiftyTwoWeekHigh', 0):.2f}")
                        st.metric("52W Low", f"${company_info.get('fiftyTwoWeekLow', 0):.2f}")
                        st.metric("Avg Volume", f"{company_info.get('averageVolume', 0) / 1_000_000:.2f}M")
                    
                    # Financial ratios
                    st.subheader("Financial Ratios")
                    
                    try:
                        # Get financial data
                        financial_data = data_fetcher.get_financial_data(stock_symbol)
                        
                        if financial_data is not None and not financial_data.empty:
                            st.dataframe(financial_data, use_container_width=True)
                        else:
                            st.warning("Financial data not available for this stock")
                    except Exception as e:
                        st.error(f"Error retrieving financial data: {e}")
                    
                    # Technical Indicators
                    st.subheader("Technical Indicators")
                    
                    indicators_chart = visualizations.create_technical_indicators(hist_data, stock_symbol)
                    st.plotly_chart(indicators_chart, use_container_width=True)
                
                with analysis_tab3:
                    st.header("Stock Price Prediction")
                    st.info("This prediction model uses historical stock data to forecast potential future price movements. Please note that these predictions are for educational purposes only and should not be used as financial advice.")
                    
                    # Add auto-trade option based on prediction
                    with st.expander("Set Up Auto-Trading Strategy"):
                        st.subheader("Configure Trading Strategy")
                        
                        with st.form("prediction_strategy_form"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                buy_threshold = st.slider("Buy if prediction is higher by (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
                                allow_averaging = st.checkbox("Allow position averaging", value=True)
                            
                            with col2:
                                sell_threshold = st.slider("Sell if prediction is lower by (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
                                max_investment = st.number_input("Maximum investment per trade ($)", min_value=100, max_value=50000, value=10000, step=100)
                            
                            create_strategy = st.form_submit_button("Create Auto-Trading Strategy")
                        
                        if create_strategy:
                            # Create a prediction-based trading strategy
                            params = {
                                'buy_threshold': buy_threshold,
                                'sell_threshold': -sell_threshold,  # Negative for downward movement
                                'allow_averaging': allow_averaging,
                                'max_investment': max_investment
                            }
                            
                            result = st.session_state.auto_trader.add_strategy(stock_symbol, 'prediction_based', params)
                            
                            if result['success']:
                                st.success(result['message'])
                            else:
                                st.error(result['message'])
                    
                    # Training the prediction model
                    with st.spinner("Training prediction model..."):
                        prediction_data = predictions.predict_next_day(hist_data, stock_symbol)
                    
                    if prediction_data is not None:
                        # Display predictions
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Predicted price for next day
                            predicted_price = prediction_data['predicted_price']
                            current_price = hist_data['Close'].iloc[-1]
                            price_diff = predicted_price - current_price
                            price_change_pct = (price_diff / current_price) * 100
                            
                            # Color coding based on prediction direction
                            if predicted_price > current_price:
                                st.markdown(f"<h3 style='color: #2E7D32'>Predicted Next Day Price: ${predicted_price:.2f}</h3>", unsafe_allow_html=True)
                                st.markdown(f"<h4 style='color: #2E7D32'>Change: +${price_diff:.2f} (+{price_change_pct:.2f}%)</h4>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<h3 style='color: #C62828'>Predicted Next Day Price: ${predicted_price:.2f}</h3>", unsafe_allow_html=True)
                                st.markdown(f"<h4 style='color: #C62828'>Change: ${price_diff:.2f} ({price_change_pct:.2f}%)</h4>", unsafe_allow_html=True)
                            
                            # Model metrics
                            st.subheader("Model Performance Metrics")
                            st.metric("Mean Absolute Error", f"${prediction_data['mae']:.4f}")
                            st.metric("Root Mean Squared Error", f"${prediction_data['rmse']:.4f}")
                            st.metric("R-squared", f"{prediction_data['r2']:.4f}")
                            
                            # Auto-trade section
                            st.subheader("Auto-Trading Based on This Prediction")
                            
                            if st.button("Execute Auto-Trading Strategies Now"):
                                # Execute all active strategies for this symbol
                                trade_results = st.session_state.auto_trader.execute_strategies(
                                    stock_symbol, prediction_data, current_price
                                )
                                
                                if trade_results:
                                    for result in trade_results:
                                        if result['action'] != 'NONE':
                                            if result['success']:
                                                st.success(f"{result['action']}: {result['message']}")
                                            else:
                                                st.error(f"{result['action']} failed: {result['message']}")
                                        else:
                                            st.info(f"No trade executed: {result['reason']}")
                                else:
                                    st.warning("No active trading strategies for this stock. Set up a strategy first.")
                        
                        with col2:
                            # Show prediction chart
                            prediction_chart = visualizations.create_prediction_chart(
                                hist_data, prediction_data, stock_symbol
                            )
                            st.plotly_chart(prediction_chart, use_container_width=True)
                        
                        st.caption("⚠️ Disclaimer: This is a simplified prediction model for educational purposes. Stock market predictions are inherently uncertain.")
                    else:
                        st.error("Unable to generate predictions. Insufficient data or error in the prediction model.")
                
                with analysis_tab4:
                    # About the company section
                    st.header("About the Company")
                    
                    # Company profile
                    if 'longBusinessSummary' in company_info:
                        st.markdown(company_info['longBusinessSummary'])
                    else:
                        st.warning("No company description available")
                    
                    # Key company data
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Company Information")
                        company_data = {
                            "Sector": company_info.get('sector', 'N/A'),
                            "Industry": company_info.get('industry', 'N/A'),
                            "Full Time Employees": company_info.get('fullTimeEmployees', 'N/A'),
                            "Country": company_info.get('country', 'N/A'),
                            "Website": company_info.get('website', 'N/A'),
                        }
                        
                        for key, value in company_data.items():
                            st.markdown(f"**{key}:** {value}")
                    
                    with col2:
                        st.subheader("Key Executives")
                        try:
                            if 'companyOfficers' in company_info and company_info['companyOfficers']:
                                executives_data = []
                                for officer in company_info['companyOfficers'][:5]:  # Limit to top 5
                                    executive = {
                                        "Name": officer.get('name', 'N/A'),
                                        "Title": officer.get('title', 'N/A'),
                                        "Pay": f"${officer.get('totalPay', 0)/1000000:.2f}M" if 'totalPay' in officer else 'N/A'
                                    }
                                    executives_data.append(executive)
                                
                                df_executives = pd.DataFrame(executives_data)
                                st.dataframe(df_executives, use_container_width=True)
                            else:
                                st.info("No executive data available")
                        except Exception as e:
                            st.error(f"Error displaying executive data: {str(e)}")
            else:
                st.error(f"No data available for {stock_symbol}. Please check the symbol and try again.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try a different stock symbol or timeframe.")

# -------------- PORTFOLIO TAB --------------
with tab_portfolio:
    st.header("Investment Portfolio")
    st.subheader("Demo Trading Account")
    
    # Portfolio overview
    portfolio_data = st.session_state.portfolio.get_portfolio()
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Value", f"${portfolio_data['total_value']:.2f}")
    
    with col2:
        cash_pct = (portfolio_data['cash'] / portfolio_data['total_value']) * 100 if portfolio_data['total_value'] > 0 else 0
        st.metric("Cash", f"${portfolio_data['cash']:.2f}", f"{cash_pct:.1f}% of portfolio")
    
    with col3:
        invested = portfolio_data['total_value'] - portfolio_data['cash']
        invested_pct = (invested / portfolio_data['total_value']) * 100 if portfolio_data['total_value'] > 0 else 0
        st.metric("Invested", f"${invested:.2f}", f"{invested_pct:.1f}% of portfolio")
    
    with col4:
        profit_loss = portfolio_data['total_value'] - portfolio_data['initial_value']
        profit_loss_pct = (profit_loss / portfolio_data['initial_value']) * 100
        st.metric("Total P/L", f"${profit_loss:.2f}", f"{profit_loss_pct:.2f}%", delta_color="normal" if profit_loss >= 0 else "inverse")
    
    # Current positions
    st.subheader("Current Positions")
    positions_df = st.session_state.portfolio.get_positions()
    
    if not positions_df.empty:
        # Update positions with current prices if we have data
        if current_stock_data and 'symbol' in current_stock_data:
            st.session_state.portfolio.update_position_value(
                current_stock_data['symbol'], 
                current_stock_data['price']
            )
            # Refresh the dataframe
            positions_df = st.session_state.portfolio.get_positions()
        
        # Style the dataframe
        st.dataframe(
            positions_df.style.format({
                'Cost Basis': '${:.2f}',
                'Current Value': '${:.2f}',
                'Profit/Loss': '${:.2f}',
                'P/L %': '{:.2f}%'
            }).map(
                lambda x: 'color: #2E7D32' if x > 0 else 'color: #C62828',
                subset=['Profit/Loss', 'P/L %']
            ),
            use_container_width=True
        )
    else:
        st.info("You don't have any open positions. Go to the Market Analysis tab to buy stocks.")
    
    # Transaction history
    st.subheader("Transaction History")
    transactions_df = st.session_state.portfolio.get_transactions()
    
    if not transactions_df.empty:
        # Add filter for transaction type
        transaction_types = ['All'] + sorted(transactions_df['type'].unique().tolist())
        selected_type = st.selectbox("Filter by transaction type", transaction_types)
        
        if selected_type != 'All':
            filtered_df = transactions_df[transactions_df['type'] == selected_type]
        else:
            filtered_df = transactions_df
        
        # Display transactions with newest first
        st.dataframe(
            filtered_df.sort_values('date', ascending=False).style.format({
                'price': '${:.2f}',
                'total': '${:.2f}'
            }),
            use_container_width=True
        )
    else:
        st.info("No transactions yet. Start trading to see your transaction history.")
    
    # Reset portfolio button
    if st.button("Reset Portfolio"):
        reset_confirm = st.empty()
        confirm = reset_confirm.checkbox("Are you sure? This will reset your portfolio to the initial state.")
        if confirm:
            result = st.session_state.portfolio.reset_portfolio()
            st.success(result['message'])
            reset_confirm.empty()
            st.rerun()

# -------------- AUTO-TRADING TAB --------------
with tab_trading:
    st.header("Automated Trading Strategies")
    st.markdown("Set up and manage automated trading strategies based on predictions and price thresholds.")
    
    # Get all strategies
    strategies_df = st.session_state.auto_trader.get_strategies()
    
    # Display current strategies
    st.subheader("Active Trading Strategies")
    
    if not strategies_df.empty:
        st.dataframe(strategies_df, use_container_width=True)
        
        # Strategy management
        st.subheader("Manage Strategies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Toggle strategy
            with st.form("toggle_strategy_form"):
                strategy_ids = strategies_df['ID'].tolist()
                strategy_to_toggle = st.selectbox("Select strategy", strategy_ids)
                toggle_action = st.selectbox("Action", ["Enable", "Disable"])
                toggle_submit = st.form_submit_button("Update Strategy")
            
            if toggle_submit:
                active = True if toggle_action == "Enable" else False
                result = st.session_state.auto_trader.toggle_strategy(strategy_to_toggle, active)
                if result['success']:
                    st.success(result['message'])
                    st.rerun()
                else:
                    st.error(result['message'])
        
        with col2:
            # Remove strategy
            with st.form("remove_strategy_form"):
                strategy_to_remove = st.selectbox("Select strategy to remove", strategy_ids, key="remove_select")
                remove_submit = st.form_submit_button("Remove Strategy")
            
            if remove_submit:
                result = st.session_state.auto_trader.remove_strategy(strategy_to_remove)
                if result['success']:
                    st.success(result['message'])
                    st.rerun()
                else:
                    st.error(result['message'])
    
    else:
        st.info("No trading strategies configured. Create a strategy from the Prediction tab or below.")
    
    # Create new strategy
    with st.expander("Create New Trading Strategy"):
        st.subheader("Configure Threshold-Based Strategy")
        
        with st.form("threshold_strategy_form"):
            ticker = st.text_input("Stock Symbol", value="AAPL").upper()
            
            st.markdown("#### Entry Strategy")
            buy_price = st.number_input("Buy when price is below", min_value=0.01, step=0.01, value=150.0)
            max_investment = st.number_input("Maximum investment ($)", min_value=100, max_value=50000, value=10000, step=100)
            
            st.markdown("#### Exit Strategy")
            take_profit = st.slider("Take profit at (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
            stop_loss = st.slider("Stop loss at (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
            
            create_threshold_strategy = st.form_submit_button("Create Strategy")
        
        if create_threshold_strategy:
            # Configure parameters
            params = {
                'buy_price': buy_price,
                'take_profit': take_profit,
                'stop_loss': -stop_loss,  # Negative for downward movement
                'max_investment': max_investment
            }
            
            # Create the strategy
            result = st.session_state.auto_trader.add_strategy(ticker, 'threshold', params)
            
            if result['success']:
                st.success(result['message'])
                st.rerun()
            else:
                st.error(result['message'])
    
    # Strategy execution section
    st.subheader("Manual Strategy Execution")
    
    if not strategies_df.empty:
        # Get current price for selected stock
        with st.form("execute_strategy_form"):
            strategy_id = st.selectbox("Select strategy to execute", strategies_df['ID'].tolist())
            symbol = strategies_df[strategies_df['ID'] == strategy_id]['Symbol'].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"Selected strategy for: {symbol}")
            
            with col2:
                st.write(f"Type: {strategies_df[strategies_df['ID'] == strategy_id]['Type'].iloc[0]}")
            
            execute_submit = st.form_submit_button("Execute Strategy Now")
        
        if execute_submit:
            with st.spinner(f"Fetching current data for {symbol}..."):
                current_price = data_fetcher.get_current_price(symbol)
                historical_data = data_fetcher.get_historical_data(symbol, "1mo", "1d")
                
                if current_price and historical_data is not None:
                    # Get predictions for prediction-based strategies
                    prediction_data = predictions.predict_next_day(historical_data, symbol)
                    
                    # Execute the strategy
                    results = st.session_state.auto_trader.execute_strategies(
                        symbol, prediction_data, current_price
                    )
                    
                    for result in results:
                        if result['action'] != 'NONE':
                            if result['success']:
                                st.success(f"{result['action']}: {result['message']}")
                            else:
                                st.error(f"{result['action']} failed: {result['message']}")
                        else:
                            st.info(f"No trade executed: {result['reason']}")
                else:
                    st.error(f"Could not fetch current data for {symbol}")
    else:
        st.info("Create a trading strategy first to execute it manually.")
