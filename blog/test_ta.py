from ta import TABacktester
import matplotlib.pyplot as plt

stock = 'A7RU.SI'

smabt = TABacktester(stock,42,252,14,14,'2010-2-28', '2021-2-28',0)
smabt.optimize()
smabt.plot_all()
fig = smabt.plot_RSI_results()
plt.show(fig)
