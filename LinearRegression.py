import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
'''
	[출처] : https://www.kaggle.com/code/ryanholbrook/linear-regression-with-time-series
'''
'''
	first lesson
'''
from warnings import simplefilter
simplefilter("ignore")  # ignore warnings to clean up output cells
plt.style.use("ggplot")  # ggplot 스타일 사용

plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc("axes",labelweight="bold",labelsize="large",titleweight="bold",titlesize=14,titlepad=10,)
plot_params = dict(color="0.75",style=".-",markeredgecolor="0.25",markerfacecolor="0.25",legend=False,)

# Load Tunnel Traffic dataset
tunnel = pd.read_csv(
    "./data/tunnel.csv", 
    index_col = 'Day',
    parse_dates=['Day'])

tunnel = tunnel.to_period()

df = tunnel.copy()
df['Time'] = np.arange(len(tunnel.index))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Training data
X = df.loc[:, ['Time']]  # features
y = df.loc[:, 'NumVehicles']  # target

# Train the model
model = LinearRegression()
model.fit(X, y)

# Store the fitted values as a time series with the same time index as
# the training data
y_pred = pd.Series(model.predict(X), index=X.index)

mse_time = mean_squared_error(y, y_pred)
print(f'Mean Squared Error (Time Model): {mse_time}')
# plot_params = dict(
#     color="0.75",
#     style=".-",
#     markeredgecolor="0.25",
#     markerfacecolor="0.25",
#     legend=False,
# )

# ax = y.plot(**plot_params)  # 실제 데이터 시각화
# ax = y_pred.plot(ax=ax, linewidth=3, color='orange', label='Predicted')  # 예측 데이터 시각화
# ax.set_title('Time Plot of Tunnel Traffic')
# ax.set_xlabel('Date')
# ax.set_ylabel('Number of Vehicles')
# ax.legend()
# plt.show()

df['Lag_1'] = df['NumVehicles'].shift(1)
X = df.loc[:, ['Lag_1']]
X.dropna(inplace=True)  # drop missing values in the feature set
y = df.loc[:, 'NumVehicles']  # create the target
y, X = y.align(X, join='inner')  # drop corresponding values in target

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)

mse_lag = mean_squared_error(y, y_pred)
print(f'Mean Squared Error (Lag Model): {mse_lag}') 
'''
	이전 속성값을 활용하여 LinearRegression()을 활용한 결과 오차가 더 작아짐
'''

# fig, ax = plt.subplots()
# ax.plot(X['Lag_1'], y, '.', color='0.25')
# ax.plot(X['Lag_1'], y_pred)
# ax.set_aspect('equal')
# ax.set_ylabel('NumVehicles')
# ax.set_xlabel('Lag_1')
# ax.set_title('Lag Plot of Tunnel Traffic')
# ax.legend()
# plt.show()

'''
	second lesson
'''
from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

simplefilter("ignore")  # ignore warnings to clean up output cells

# Set Matplotlib defaults
plt.style.use("ggplot")  # ggplot 스타일 사용
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc("axes",labelweight="bold",labelsize="large",titleweight="bold",titlesize=14,titlepad=10,)
plot_params = dict(color="0.75",style=".-",markeredgecolor="0.25",markerfacecolor="0.25",legend=False,)

# Load Tunnel Traffic dataset
tunnel = pd.read_csv(
    "./data/tunnel.csv", 
    index_col = 'Day',
    parse_dates=['Day']
    )

tunnel = tunnel.to_period()
moving_average = tunnel.rolling(
    window=365,       # 365-day window
    center=True,      # puts the average at the center of the window
    min_periods=183,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)

from statsmodels.tsa.deterministic import DeterministicProcess

dp = DeterministicProcess(
    index=tunnel.index,  # dates from the training data
    constant=True,       # dummy feature for the bias (y_intercept)
    order=1,             # the time dummy (trend)
    drop=True,           # drop terms if necessary to avoid collinearity
)
# `in_sample` creates features for the dates given in the `index` argument
X = dp.in_sample()

from sklearn.linear_model import LinearRegression

y = tunnel["NumVehicles"]  # the target

# The intercept is the same as the `const` feature from
# DeterministicProcess. LinearRegression behaves badly with duplicated
# features, so we need to be sure to exclude it here.
model = LinearRegression(fit_intercept=False)
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)

X = dp.out_of_sample(steps=30)

y_fore = pd.Series(model.predict(X), index=X.index)

