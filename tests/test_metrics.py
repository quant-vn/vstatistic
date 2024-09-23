import pandas as pd
import vstatistic as vs

meta_data = pd.read_csv('.\\tests\\return_meta.csv', index_col='Date', parse_dates=True)
meta_return = meta_data.Return
print("META RETURN: ", meta_return)
spy_data = pd.read_csv('\\tests\\return_spy.csv', index_col='Date', parse_dates=True)
spy_return = spy_data.Return
print("SPY RETURN: ", spy_return)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

metrics = vs.metrics.Metrics()
metrics.returns = meta_return
metrics.add_benchmark(benchmark=spy_return, name='SPY')
print(metrics.metrics())
