import pandas as pd
import vstatistic as vs

meta_data = pd.read_csv('.\\tests\\return_meta.csv', index_col='Date', parse_dates=True)
meta_return = meta_data.Return
print("META RETURN: ", meta_return)
spy_data = pd.read_csv('\\tests\\return_spy.csv', index_col='Date', parse_dates=True)
spy_return = spy_data.Return
print("SPY RETURN: ", spy_return)
orderbook = pd.read_csv('./tests/orderbook.csv', parse_dates=True)
print("ORDERBOOK: ", orderbook)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

metrics = vs.metrics.Metrics()
metrics.returns = meta_return
metrics.orderbook = orderbook
metrics.add_benchmark(benchmark=spy_return, name='SPY')
print(metrics.metrics())
