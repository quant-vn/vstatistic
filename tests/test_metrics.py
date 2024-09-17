import pandas as pd
import vstatistic as vs

data = pd.read_csv(
    '/Users/thync/Projects/open/vstatistic/tests/returns.csv', index_col='Date', parse_dates=True
)
prices = data.Close
print(prices)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

metrics = vs.metrics.metrics(prices, mode="full", display=False)
print(metrics)
