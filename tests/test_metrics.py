import pandas as pd
import vstatistic as vs

meta_data = pd.read_csv('./tests/return_meta.csv', index_col='Date', parse_dates=True)
meta_return = meta_data.Return
print("META RETURN: ", meta_return)
spy_data = pd.read_csv('./tests/return_spy.csv', index_col='Date', parse_dates=True)
spy_return = spy_data.Return
print("SPY RETURN: ", spy_return)
dia_data = pd.read_csv('./tests/return_dia.csv', index_col='Date', parse_dates=True)
dia_return = dia_data.Return
print("DIA RETURN: ", dia_return)

# ORDER
orderbook = pd.read_csv('./tests/orderbook.csv', parse_dates=True)
print("ORDERBOOK: ", orderbook)
# TRANSACTION
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# metrics = vs.metrics.metrics(meta_return)

metrics = vs.metrics.Metrics()
metrics.returns = meta_return
metrics.add_benchmark(benchmark=spy_return, name='SPY')
metrics.add_benchmark(benchmark=dia_return, name='DIA')
metrics.orderbook = orderbook
metrics.transaction = orderbook
metrics.trigger = orderbook
metrics.report()


# print(_m)
# _m.to_csv('./tests/metrics.csv')
