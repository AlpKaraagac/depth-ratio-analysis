import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

def create_trading_decisions(price_data, strategy_type, y_hat_last, start_index):
    decisions = pd.Series(0, index=price_data.index, dtype=int)

    test_end_index = start_index + len(y_hat_last)
    if test_end_index > len(price_data):
        test_end_index = len(price_data)
        y_hat_last = y_hat_last[:len(price_data) - start_index]

    test_indices = price_data.index[start_index:test_end_index]

    if strategy_type == 'momentum':
        # Momentum strategy: buy if predicted price is higher than the previous price, sell if lower.
        prev_prices = price_data['midPrice'].shift(1).loc[test_indices]
        
        # Check if the predicted prices are greater or less than the previous prices
        buy_signals = y_hat_last > prev_prices.values
        sell_signals = y_hat_last < prev_prices.values
        
        # Apply the decisions to the Series
        decisions.loc[test_indices] = np.select(
            [buy_signals, sell_signals],
            [1, -1],
            default=0
        )
    
    elif strategy_type == 'mean_reversion':
        # Mean reversion strategy: buy if predicted price is below the moving average, sell if above.
        window_size = 20
        moving_avg = price_data['midPrice'].rolling(window=window_size).mean()
        
        ma_values = moving_avg.loc[test_indices]
        predicted_prices = y_hat_last
        
        # Use a small threshold to avoid spurious signals
        threshold = 0.001 
        
        # Buy signals are generated when predicted price is significantly below the moving average
        buy_signals = (predicted_prices < ma_values.values) & (ma_values.values - predicted_prices > threshold * ma_values.values)
        
        # Sell signals are generated when predicted price is significantly above the moving average
        sell_signals = (predicted_prices > ma_values.values) & (predicted_prices - ma_values.values > threshold * ma_values.values)
        
        # Apply the decisions to the Series
        decisions.loc[test_indices] = np.select(
            [buy_signals, sell_signals],
            [1, -1],
            default=0
        )
    
    elif strategy_type == 'volatility_breakout':
        # Volatility Breakout strategy: trade when the predicted price moves outside a volatility band.
        volatility_window = 20
        breakout_factor = 2.0
        
        # Calculate a moving average and standard deviation (volatility)
        moving_avg = price_data['midPrice'].rolling(window=volatility_window).mean()
        std_dev = price_data['midPrice'].rolling(window=volatility_window).std()
        
        # Define the upper and lower bands based on volatility
        upper_band = moving_avg + std_dev * breakout_factor
        lower_band = moving_avg - std_dev * breakout_factor
        
        # Get the band values for the test period
        upper_values = upper_band.loc[test_indices]
        lower_values = lower_band.loc[test_indices]
        predicted_prices = y_hat_last
        
        # Buy signals are generated when the predicted price breaks above the upper band
        buy_signals = predicted_prices > upper_values.values
        
        # Sell signals are generated when the predicted price breaks below the lower band
        sell_signals = predicted_prices < lower_values.values
        
        # Apply the decisions to the Series
        decisions.loc[test_indices] = np.select(
            [buy_signals, sell_signals],
            [1, -1],
            default=0
        )
    
    return decisions

def analyze_decisions(decisions, price_data, strategy_type):
    print(f"\n=== Trading Decision Analysis for {strategy_type.upper()} Strategy ===")
    
    buy_count = (decisions == 1).sum()
    sell_count = (decisions == -1).sum()
    hold_count = (decisions == 0).sum()
    total_signals = buy_count + sell_count
    
    print(f"Total Buy Signals: {buy_count}")
    print(f"Total Sell Signals: {sell_count}")
    print(f"Total Hold Periods: {hold_count}")
    print(f"Signal Frequency: {total_signals / len(decisions) * 100:.2f}%")
    
    # Prepare data for plotting
    decisions_df = pd.DataFrame({
        'DateTime': price_data.index,
        'Decisions': decisions,
        'MidPrice': price_data['midPrice']
    })
    
    try:
        plt.figure(figsize=(15, 8))
        
        # Plot price and decisions on separate subplots for clarity
        plt.subplot(2, 1, 1)
        plt.plot(decisions_df['DateTime'], decisions_df['MidPrice'], label='Mid Price', alpha=0.7)
        plt.title(f'Price and Trading Decisions - {strategy_type.upper()} Strategy')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        buy_points = decisions_df[decisions_df['Decisions'] == 1]
        sell_points = decisions_df[decisions_df['Decisions'] == -1]
        
        # Use scatter plots to mark buy and sell points on the price chart
        plt.scatter(buy_points['DateTime'], buy_points['MidPrice'], 
                    color='green', marker='^', s=50, label='Buy Signal', alpha=0.8)
        plt.scatter(sell_points['DateTime'], sell_points['MidPrice'], 
                    color='red', marker='v', s=50, label='Sell Signal', alpha=0.8)
        plt.plot(decisions_df['DateTime'], decisions_df['MidPrice'], alpha=0.3, color='gray')
        plt.ylabel('Price')
        plt.xlabel('DateTime')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'graphs/trading_decisions_{strategy_type}.png')
        plt.show()
    except Exception as e:
        print(f"Warning: Could not create trading decisions plot for {strategy_type}: {e}")
        print("Continuing with analysis...")
    
    return {
        'buy_count': buy_count,
        'sell_count': sell_count,
        'hold_count': hold_count,
        'total_signals': total_signals,
        'signal_frequency': total_signals / len(decisions) * 100
    }

def main():
    print("=== Trading Decision Part ===")
    
    try:
        # Load the results from the previous TensorFlow script
        with open('tensorflow_results.pkl', 'rb') as f:
            tensorflow_results = pickle.load(f)
        print("Successfully loaded TensorFlow results")
    except FileNotFoundError:
        print("Error: 'tensorflow_results.pkl' not found. Please run tensorflow_part.py first.")
        return
    except Exception as e:
        print(f"Error loading TensorFlow results: {e}")
        return
    
    # Extract necessary data
    df = tensorflow_results['df']
    y_hat_last = tensorflow_results['y_hat_last']
    SEQ_LEN = tensorflow_results['SEQ_LEN']
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Predicted prices length: {len(y_hat_last)}")
    print(f"Sequence length: {SEQ_LEN}")
    
    # Define the start index for the test period
    start_index = SEQ_LEN + int(0.8 * (len(df) - SEQ_LEN))
    print(f"Start index for test period: {start_index}")
    
    # Iterate through each trading strategy, including the new one
    strategies = ['momentum', 'mean_reversion', 'volatility_breakout']
    all_results = {}
    
    for strategy_type in strategies:
        print(f"\n--- Creating trading decisions for {strategy_type.upper()} strategy ---")
        
        try:
            decisions = create_trading_decisions(
                df, 
                strategy_type, 
                y_hat_last, 
                start_index
            )
            
            analysis_results = analyze_decisions(decisions, df, strategy_type)
            all_results[strategy_type] = {
                'decisions': decisions,
                'analysis': analysis_results
            }
            
            print(f"Trading decisions created for {strategy_type} strategy")
            
        except Exception as e:
            print(f"Error creating trading decisions for {strategy_type}: {e}")
            print(f"Continuing with other strategies...")
            continue
    
    if not all_results:
        print("No successful trading strategies created. Cannot proceed.")
        return
    
    trading_results = {
        'df': df,
        'strategies': all_results,
        'start_index': start_index,
        'SEQ_LEN': SEQ_LEN
    }
    
    with open('trading_results.pkl', 'wb') as f:
        pickle.dump(trading_results, f)
    
    print("\nTrading decisions saved to 'trading_results.pkl'")
    print("Trading decision part completed successfully!")
    
    print("\n=== Summary ===")
    for strategy, results in all_results.items():
        analysis = results['analysis']
        print(f"{strategy.upper()}: {analysis['buy_count']} buys, {analysis['sell_count']} sells, "
              f"{analysis['signal_frequency']:.2f}% signal frequency")

if __name__ == "__main__":
    main()
