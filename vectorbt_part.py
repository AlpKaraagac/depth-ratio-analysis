import vectorbt as vbt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

vbt.settings.returns['year_freq'] = '252 days'
vbt.settings.array_wrapper['freq'] = '1min'

def run_backtest(df, decisions, strategy_name):
    print(f"\n=== VectorBT Backtest for {strategy_name.upper()} Strategy ===")
    
    try:
        backtest_price = df['midPrice']
        decisions_df = pd.DataFrame(decisions)
        weights = decisions_df.div(decisions_df.abs().sum(axis=1), axis=0).fillna(0)
        
        pf = vbt.Portfolio.from_orders(
            close=backtest_price,
            size=weights,
            size_type='amount',
            freq='1min',
            init_cash=100,
            cash_sharing=True,
            call_seq='auto',
            fees=0.001,
            slippage=0.0005
        )

        full_stats = pf.stats()
        ann_factor = pf.returns().vbt.returns().ann_factor
        
        print(f"Ann Factor:                         {ann_factor}")
        print("\nBacktest Stats:")
        print(f"Total Return [%]:                   {full_stats['Total Return [%]']:.3f}%")
        print(f"Annualized Expected Return [%]:     {(pf.returns().mean() * ann_factor):.3f}%")
        print(f"Annualized Expected Volatility [%]: {pf.returns().std() * (ann_factor ** .5):.3f}%")
        print(f"Sharpe Ratio:                       {full_stats['Sharpe Ratio']:.3f}")
        print(f"Max Drawdown [%]:                   {full_stats['Max Drawdown [%]']:.3f}%")
        print(f"Win Rate:                           {full_stats['Win Rate [%]']:.3f}%")
        print(f"Profit Factor:                      {full_stats['Profit Factor']:.3f}")
        print(f"Calmar Ratio:                       {full_stats['Calmar Ratio']:.3f}")
        
        create_backtest_plots(pf, strategy_name)
        
        return {
            'portfolio': pf,
            'stats': full_stats,
            'ann_factor': ann_factor
        }
        
    except Exception as e:
        print(f"Error in backtest for {strategy_name}: {e}")
        print("Attempting to create basic portfolio...")
        """
        try:
            backtest_price = df['midPrice']
            decisions_df = pd.DataFrame(decisions)
            weights = decisions_df.div(decisions_df.abs().sum(axis=1), axis=0).fillna(0)
            
            pf = vbt.Portfolio.from_orders(
                close=backtest_price,
                size=weights,
                size_type='amount',
                freq='1T',
                init_cash=100,
                cash_sharing=False,
                call_seq='auto',
                fees=0.001,
                slippage=0.0005
            )
            
            full_stats = pf.stats()
            ann_factor = pf.returns().vbt.returns().ann_factor
            
            print(f"Fallback backtest successful!")
            print(f"Total Return [%]: {full_stats['Total Return [%]']:.3f}%")
            
            return {
                'portfolio': pf,
                'stats': full_stats,
                'ann_factor': ann_factor
            }
            
        except Exception as e2:
            print(f"Fallback backtest also failed: {e2}")
            raise e2
        """

def create_backtest_plots(pf, strategy_name):
    os.makedirs('graphs', exist_ok=True)
    
    try:
        plt.figure(figsize=(12, 6))
        pf.value().plot()
        plt.title(f'Portfolio Value - {strategy_name.upper()} Strategy')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'graphs/portfolio_value_{strategy_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(12, 6))
        returns = pf.returns()
        returns.hist(bins=50)
        plt.title(f'Returns Distribution - {strategy_name.upper()} Strategy')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'graphs/returns_distribution_{strategy_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

        """plt.figure(figsize=(12, 6))
        try:
            drawdown_plot = pf.plot_drawdown()
        except Exception as e:
            print(f"Warning: Could not create drawdown plot: {e}")
            plt.plot(pf.drawdown())
        plt.title(f'Drawdown Analysis - {strategy_name.upper()} Strategy')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'graphs/drawdown_{strategy_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(12, 6))
        try:
            underwater_plot = pf.plot_underwater()
        except Exception as e:
            print(f"Warning: Could not create underwater plot: {e}")
            plt.plot(pf.drawdown().cummin())
        plt.title(f'Underwater Plot - {strategy_name.upper()} Strategy')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'graphs/underwater_{strategy_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    """ 
    except Exception as e:
        print(f"Warning: Error creating plots for {strategy_name}: {e}")
        print("Continuing with backtest analysis...")
    
def compare_strategies(backtest_results):
    print("\n=== Strategy Comparison ===")
    
    comparison_data = []
    for strategy, results in backtest_results.items():
        stats = results['stats']
        comparison_data.append({
            'Strategy': strategy.upper(),
            'Total Return [%]': stats['Total Return [%]'],
            'Sharpe Ratio': stats['Sharpe Ratio'],
            'Max Drawdown [%]': stats['Max Drawdown [%]'],
            'Win Rate [%]': stats['Win Rate [%]'],
            'Profit Factor': stats['Profit Factor'],
            'Calmar Ratio': stats['Calmar Ratio']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False, float_format='%.3f'))
    
    create_comparison_plots(backtest_results)
    
    return comparison_df

def create_comparison_plots(backtest_results):
    try:
        plt.figure(figsize=(15, 8))
        for strategy, results in backtest_results.items():
            pf = results['portfolio']
            pf.value().plot(label=f'{strategy.upper()}', linewidth=2)
        
        plt.title('Portfolio Value Comparison - All Strategies')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('graphs/strategy_comparison_portfolio.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(15, 8))
        for strategy, results in backtest_results.items():
            pf = results['portfolio']
            returns = pf.returns()
            returns.plot(label=f'{strategy.upper()}', alpha=0.7)
        
        plt.title('Returns Comparison - All Strategies')
        plt.xlabel('Time')
        plt.ylabel('Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('graphs/strategy_comparison_returns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(10, 8))
        for strategy, results in backtest_results.items():
            stats = results['stats']
            returns = stats['Total Return [%]']
            volatility = results['portfolio'].returns().std() * (results['ann_factor'] ** 0.5) * 100
            
            plt.scatter(volatility, returns, s=100, label=f'{strategy.upper()}', alpha=0.8)
            plt.annotate(f'{strategy.upper()}', (volatility, returns), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Annualized Volatility [%]')
        plt.ylabel('Total Return [%]')
        plt.title('Risk-Return Profile - All Strategies')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('graphs/risk_return_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Warning: Error creating comparison plots: {e}")
        print("Continuing with analysis...")

def save_final_results(backtest_results, comparison_df):
    serializable_results = {}
    for strategy_name, results in backtest_results.items():
        serializable_results[strategy_name] = {
            'stats': results['stats'],
            'ann_factor': results['ann_factor']
        }
    
    final_results = {
        'backtest_results': serializable_results,
        'comparison': comparison_df,
        'timestamp': pd.Timestamp.now()
    }
    
    with open('vectorbt_final_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)
    
    comparison_df.to_csv('strategy_comparison.csv', index=False)
    
    print("\nFinal results saved to 'vectorbt_final_results.pkl'")
    print("Strategy comparison saved to 'strategy_comparison.csv'")
    print("Note: Portfolio objects were excluded from saved results (not serializable)")

def main():
    print("=== VectorBT Backtesting Part ===")
    
    try:
        with open('trading_results.pkl', 'rb') as f:
            trading_results = pickle.load(f)
        print("Successfully loaded trading decision results")
    except FileNotFoundError:
        print("Error: 'trading_results.pkl' not found. Please run trading_decision_part.py first.")
        return
    except Exception as e:
        print(f"Error loading trading decision results: {e}")
        return
    
    # Extract data
    df = trading_results['df']
    strategies = trading_results['strategies']
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Available strategies: {list(strategies.keys())}")
    
    # Run backtests for all strategies
    backtest_results = {}
    
    for strategy_name, strategy_data in strategies.items():
        decisions = strategy_data['decisions']
        
        print(f"\n{'='*50}")
        print(f"Running backtest for {strategy_name.upper()} strategy")
        print(f"{'='*50}")
        
        try:
            results = run_backtest(df, decisions, strategy_name)
            backtest_results[strategy_name] = results
            print(f"Backtest completed successfully for {strategy_name}")
        except Exception as e:
            print(f"Error running backtest for {strategy_name}: {e}")
            print(f"Trying to continue with other strategies...")
            continue
    
    if not backtest_results:
        print("No successful backtests completed. Exiting.")
        return
    
    comparison_df = compare_strategies(backtest_results)
    
    save_final_results(backtest_results, comparison_df)
    
    print("\n" + "="*60)
    print("VECTORBT BACKTESTING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\n=== Final Summary ===")
    best_strategy = comparison_df.loc[comparison_df['Total Return [%]'].idxmax()]
    print(f"Best Performing Strategy: {best_strategy['Strategy']}")
    print(f"Total Return: {best_strategy['Total Return [%]']:.3f}%")
    print(f"Sharpe Ratio: {best_strategy['Sharpe Ratio']:.3f}")
    print(f"Max Drawdown: {best_strategy['Max Drawdown [%]']:.3f}%")

if __name__ == "__main__":
    main()
