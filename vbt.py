import numpy as np
import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt
import seaborn as sns

vbt.settings.returns['year_freq'] = '252 days'
vbt.settings.array_wrapper['freq'] = 'd'

sns.set_style('darkgrid')
dates = pd.read_csv('commodity_prices.csv', index_col='date', parse_dates=['date']).index[:10]
assets = ["A", "B", "C"]
price_df = pd.DataFrame({
    "A":     np.asarray([1, 2, 4,  12, 6,   1,   2, 4, 6, 3]),
    "B": 4 * np.asarray([1, 3, 12, 3,  1.5, 6,   3, 1, 4, 1]),
    "C": 2 * np.asarray([5, 1, 4,  2,  6,   1.5, 3, 9, 7, 14])
}, index=dates)

decisions = np.array([
    [1, 0, 0],  # 1000
    [1, 0, 0],  # 2000
    [1, 0, 0],  # 4000
    [-1, 1, 1], # 12000
    [-1, 1, 1], # 6000 + 2000 + 12000 = 20000
    [0, 1, 0],  # (20000*(2-1/6) + 20000*4 + 20000/4)/3 = 40555.555555555555
    [0, 0, 0],  # 20277.777778
    [0, 0, 0],  # 20277.777778
    [-1, 0, 0], # 20277.777778
    [0, 0, 0]   # 20277.777778*1.5 = 30416.666666999998
])
decisions_df = pd.DataFrame(decisions, index=dates, columns=assets)
weights = decisions_df.div(decisions_df.abs().sum(axis=1), axis=0).fillna(0)

pf = vbt.Portfolio.from_orders(
    close=price_df,
    size=weights,
    size_type='targetpercent',
    init_cash=1000,
    freq="D",
    cash_sharing=True,
    call_seq='auto'
)

print("Prices:")
print(price_df, "\n")

print("Decisions (1 = allocated long, 0 = not allocated):")
print(decisions_df, "\n")

print("Portfolio Stats:")
full_stats = pf.stats()
ret_stats = pf.returns_stats()
ann_factor = pf.returns().vbt.returns().ann_factor
print(f"Ann Factor:                         {ann_factor}")
print(f"Total Return [%]:                   {full_stats['Total Return [%]']:.3f}%")
print(f"Annualized Expected Return [%]:     {(pf.returns().mean() * ann_factor):.3f}%")
print(f"Annualized Expected Volatility [%]: {pf.returns().std() * (ann_factor ** .5):.3f}%")
print(f"Sharpe Ratio:                       {full_stats['Sharpe Ratio']:.3f}")
print(f"Sharpe Ratio:                       {((pf.returns().mean() * ann_factor)/(pf.returns().std() * (ann_factor ** .5))):.3f}")
print(f"Max Drawdown [%]:                   {full_stats['Max Drawdown [%]']:.3f}%")

pf.value().plot()
plt.show()

print('Values', pf.value())
print('Returns', pf.returns())
