
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
MSE, RMSE, MAD, MAPE
'''

# 그래프를 그리는 함수 정의
def plot_moving_average(df, column_name, window_size):
	# 그래프 그리기
	plt.figure(figsize=(12, 6))
	plt.plot(df[column_name], label='Original Data', color='blue', alpha=0.6)
	plt.plot(df['Moving_Average'], label=f'{window_size}-Day Moving Average', color='orange', linewidth=2)
	plt.title('Time Series and Moving Average')
	plt.xlabel('Date')
	plt.ylabel(column_name)
	plt.legend()
	plt.grid()
	plt.show()

# 그래프를 그리는 함수 정의
def both_plot_moving_average(df, column_name, window_size):
	# 그래프 그리기
	plt.figure(figsize=(12, 6))
	plt.plot(df[column_name], label='Original Data', color='blue', alpha=0.6)
	plt.plot(df['Moving_Average'], label=f'{window_size}-Day Moving Average', color='orange', linewidth=2)
	plt.plot(df['Double_Moving_Average'], label=f'Double {window_size}-Day Moving Average', color='green', linewidth=2)
	plt.title('Time Series and Moving Averages')
	plt.xlabel('Date')
	plt.ylabel(column_name)
	plt.legend()
	plt.grid()
	plt.show()

# 성능 지표 계산 함수
def calculate_metrics(actual, predicted):
	mse = np.mean((actual - predicted) ** 2)
	rmse = np.sqrt(mse)
	mad = np.mean(np.abs(actual - predicted))
	mape = np.mean(np.abs((actual - predicted) / actual)) * 100
	return mse, rmse, mad, mape

# 이중 지수 평활법 적용 함수
def double_exponential_smoothing(series, alpha, beta):
	n = len(series)
	level = np.zeros(n)
	trend = np.zeros(n)
	smoothed = np.zeros(n)
	level[0] = series[0]
	trend[0] = series[1] - series[0]
	for t in range(1, n):
		if t == 1: smoothed[t] = level[0]
		else: smoothed[t] = level[t-1] + trend[t-1]
		level[t] = alpha * series[t] + (1 - alpha) * (level[t-1] + trend[t-1])
		trend[t] = beta * (level[t] - level[t-1]) + (1 - beta) * trend[t-1]

	return smoothed

## 홀트모형
def holt_smoothing(series, alpha, beta):
	level = series[0]
	trend = series[1] - series[0]
	smoothed_values = [level + trend]
	
	for i in range(1, len(series)):
		last_level = level
		level = alpha * series[i] + (1 - alpha) * (level + trend)
		trend = beta * (level - last_level) + (1 - beta) * trend
		smoothed_values.append(level + trend)
	
	return np.array(smoothed_values)

# 윈터스 모형 적용 함수
def winters_smoothing(series, alpha, beta, gamma, seasonal_periods, model_type='additive'):
    n = len(series)
    seasonal = np.zeros(seasonal_periods)
    level = series[0]
    trend = series[1] - series[0]
    
    # 계절성 초기화
    for i in range(seasonal_periods):
        seasonal[i] = series[i] / level if model_type == 'multiplicative' else series[i] - level

    smoothed_values = np.zeros(n)
    
    for t in range(n):
        if t == 0:
            smoothed_values[t] = level
        else:
            if model_type == 'additive':
                smoothed_values[t] = level + trend + seasonal[t % seasonal_periods]
            else:
                smoothed_values[t] = (level + trend) * seasonal[t % seasonal_periods]

        last_level = level
        level = alpha * (series[t] - seasonal[t % seasonal_periods]) + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        seasonal[t % seasonal_periods] = gamma * (series[t] - level) + (1 - gamma) * seasonal[t % seasonal_periods]
    
    return smoothed_values

# 분해법 모형 적용 함수
def seasonal_decompose(series, model_type):
    trend = series.rolling(window=12, center=True).mean()
    if model_type == 'additive':
        seasonal = series - trend
    else:  # multiplicative
        seasonal = series / trend
    seasonal = seasonal.rolling(window=12, center=True).mean()
    return trend, seasonal


def centered_moving_average(df, column_name, N):
    if N < 1: raise ValueError("N must be a positive integer.")
    
    # 이동평균 열 초기화
    moving_averages = pd.Series(index=df.index)
    
    # 홀수일 경우
    if N % 2 == 1:
        half_window = N // 2
        for i in range(half_window, len(df) - half_window):
            avg = df[column_name].iloc[i - half_window:i + half_window + 1].mean()
            moving_averages.iloc[i] = avg
            
    # 짝수일 경우
    else:
        half_window = N // 2
        for i in range(half_window - 1, len(df) - half_window):
            avg = df[column_name].iloc[i - half_window + 1:i + half_window + 1].mean()
            moving_averages.iloc[i] = avg
            
    return moving_averages


book_sales = pd.read_csv(
	"./data/book_sales.csv",
	index_col = 'Date',
	parse_dates=['Date']
	)
df = book_sales.copy()
df['Time'] = np.arange(len(book_sales.index))
column_name = 'Hardcover'


# tunnel = pd.read_csv(
# 	"./data/tunnel.csv", 
# 	index_col = 'Day',
# 	parse_dates=['Day'])
# df = tunnel.copy()
# df['Time'] = np.arange(len(tunnel.index))
# column_name = 'NumVehicles'
# window_size = 30

'''단순 이동평균법'''
window_size = 4
df['Moving_Average'] = df[column_name].rolling(window=window_size).mean() # plot_moving_average(df, column_name, window_size)

'''중심화 이동평균방법'''
df['Centered_Moving_Average'] = centered_moving_average(book_sales, column_name, window_size)
print(df)

'''이중 이동 평균 계산'''
df['Double_Moving_Average'] = df['Moving_Average'].rolling(window=window_size).mean() # plot_moving_average(df, column_name, window_size)

# 단순 지수 평활법 적용
alpha = 0.2  # 평활 계수
df['SES'] = df[column_name].ewm(alpha=alpha, adjust=False).mean() # plot_moving_average(df, column_name, window_size)

# 이중 지수 평활법
beta = 0.2
df['Double_Exponential_Smoothing'] = double_exponential_smoothing(df[column_name].values, alpha, beta)

# 홀트 모형 적용
alpha = 0.8  # 수준 평활 계수
beta = 0.2   # 추세 평활 계수
df['Holt_Smoothing'] = holt_smoothing(df[column_name].values, alpha, beta)


alpha = 0.8  # 수준 평활 계수
beta = 0.2   # 추세 평활 계수
gamma = 0.1  # 계절성 평활 계수
seasonal_periods = 12  # 계절성 주기
n_preds = 12  # 예측할 기간 수

# 윈터스 모형 적용 (가법)
df['Winters_Additive'] = winters_smoothing(df[column_name].values, alpha, beta, gamma, seasonal_periods, model_type='additive')
# 윈터스 모형 적용 (승법)
df['Winters_Multiplicative'] = winters_smoothing(df[column_name].values, alpha, beta, gamma, seasonal_periods, model_type='multiplicative')

# 분해법 모형 적용 (가법)
trend_add, seasonal_add = seasonal_decompose(df[column_name], model_type='additive')
df['Trend_Additive'] = trend_add
df['Seasonal_Additive'] = seasonal_add
df['Decomposed_Additive'] = df['Trend_Additive'] + df['Seasonal_Additive']

# 분해법 모형 적용 (승법)
trend_mul, seasonal_mul = seasonal_decompose(df[column_name], model_type='multiplicative')
df['Trend_Multiplicative'] = trend_mul
df['Seasonal_Multiplicative'] = seasonal_mul
df['Decomposed_Multiplicative'] = df['Trend_Multiplicative'] * df['Seasonal_Multiplicative']

metrics_ma = calculate_metrics(df[column_name].dropna(), df['Moving_Average'].dropna())
metrics_dma = calculate_metrics(df[column_name].dropna(), df['Double_Moving_Average'].dropna())
metrics_ses = calculate_metrics(df[column_name].dropna(), df['SES'].dropna()) ## 평활 계수를 고려
metrics_des = calculate_metrics(df[column_name].dropna(), df['Double_Exponential_Smoothing'].dropna()) ## 평활계수와 시간에 따른 추세를 고려
metrics_holt = calculate_metrics(df[column_name].dropna(), df['Holt_Smoothing'].dropna()) ## 장기예측에 적합
metrics_winter_additive = calculate_metrics(df[column_name].dropna(), df['Winters_Additive'].dropna()) ## 홀트모형에 계절성을 추가반영하여 확장(가법)
metrics_winter_multiplicative = calculate_metrics(df[column_name].dropna(), df['Winters_Multiplicative'].dropna()) ## 홀트모형에 계절성을 추가반영하여 확장(승법)
metrics_decompose_additive = calculate_metrics(df[column_name].dropna(), df['Decomposed_Additive'].dropna())  # 추세와 계절성을 분해한 후 예측시 다시 결합(가법)
metrics_decompose_multiplicative = calculate_metrics(df[column_name].dropna(), df['Decomposed_Multiplicative'].dropna())  # 추세와 계절성을 분해한 후 예측시 다시 결합(승법)


print("단순 이동 평균 성능 지표 (MSE, RMSE, MAD, MAPE):", metrics_ma)
print("이중 이동 평균 성능 지표 (MSE, RMSE, MAD, MAPE):", metrics_dma)
print("단순 지수 평활법 성능 지표 (MSE, RMSE, MAD, MAPE):", metrics_ses)
print("이중 지수 평활법 성능 지표 (MSE, RMSE, MAD, MAPE):", metrics_des)
print("홀트 모형 성능 지표 (MSE, RMSE, MAD, MAPE):", metrics_holt)
print("윈터스 모형-가법 성능 지표 (MSE, RMSE, MAD, MAPE):", metrics_winter_additive)
print("윈터스 모형-승법 성능 지표 (MSE, RMSE, MAD, MAPE):", metrics_winter_multiplicative)
print("분해법 성능 지표-가법 (MSE, RMSE, MAD, MAPE):", metrics_decompose_additive)
print("분해법 성능 지표-승법 (MSE, RMSE, MAD, MAPE):", metrics_decompose_multiplicative)