#필요한 패키지 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 1. 시계열 모델
	# ARIMA (AutoRegressive Integrated Moving Average): 주가 데이터가 계절성을 보이지 않거나 안정적인 경우에 적합합니다.
	# SARIMA (Seasonal ARIMA): 계절성이 있는 주가 데이터에 적합합니다.
	# Exponential Smoothing: 당신이 사용한 것처럼, 간단하게 예측할 수 있는 모델입니다. 계절성을 고려한 Holt-Winters 방법도 가능합니다.
# 2. 머신러닝 모델
	# 랜덤 포레스트 (Random Forest): 주가 데이터를 예측하기 위해 사용할 수 있는 강력한 앙상블 모델입니다. 다양한 특성을 입력으로 사용할 수 있습니다.
	# XGBoost: 부스팅 방법론을 사용하여 성능이 뛰어난 예측 모델을 구축할 수 있습니다.
	# 신경망 (Neural Networks): LSTM(Long Short-Term Memory)과 같은 순환 신경망(RNN)은 시계열 데이터에 적합하며, 긴 시퀀스를 기억하고 처리할 수 있는 장점이 있습니다.
# 3. 추가적인 고려사항
	# 특성 엔지니어링: 주가 데이터는 여러 외부 요인에 영향을 받기 때문에, 이동 평균, 변동성, 거래량 등의 특성을 추가하면 예측 성능을 향상시킬 수 있습니다.
	# 전처리: 데이터의 정상성 확인, 로그 변환, 차분 등을 통해 시계열 데이터를 안정화할 수 있습니다.
	# 교차 검증: 시계열 데이터는 일반적인 교차 검증 방법을 사용할 수 없기 때문에, 시간 기반의 교차 검증 방법을 사용해야 합니다.
# 4. 예측 방법
	# 시뮬레이션: 잔차의 분포를 고려한 시뮬레이션을 통해 불확실성을 평가할 수 있습니다.
	# 배깅/부스팅: 여러 모델을 조합하여 예측의 정확도를 높일 수 있습니다.

## 자동차 판매 데이터 https://www.hyundai.com/worldwide/ko/company/ir/ir-resources/sales-results
origin_data = pd.read_csv('./combined_sorted.csv', encoding='cp949')
origin_data['Month'] = origin_data['Month'].str.replace('-00', '-01') ## 데이터 형식이 00으로 끝나기 때문에 변경
origin_data['Month'] = pd.to_datetime(origin_data['Month'], errors='coerce')

origin_data['판매량'] = pd.to_numeric(origin_data['판매량'], errors='coerce')
origin_data.dropna(inplace=True)  # NaN 제거

# # 모델별로 그룹화
# cleaned_data = []

# for model_name, group in origin_data.groupby('모델'):
#     # 중복된 날짜에서 판매량이 가장 큰 레코드만 선택
#     group = group.loc[group.groupby('Month')['판매량'].idxmax()]
	
#     cleaned_data.append(group)

# # 모든 모델의 데이터를 하나의 DataFrame으로 결합
# cleaned_data = pd.concat(cleaned_data, ignore_index=True)
# cleaned_data.to_csv("cleaning_data.csv", encoding='cp949')

# 모델별로 그룹화
for model_name, group in origin_data.groupby('모델'):
	# 중복된 날짜 찾기
	duplicate_dates = group[group['Month'].duplicated(keep=False)]
	
	if not duplicate_dates.empty:
		print(f"{model_name}에서 중복된 날짜:")
		print(duplicate_dates)

# 차량 종류별 모델 피팅
models = {}
for model_name, group in origin_data.groupby('모델'):
    group = group.groupby('Month').last().reset_index()
    group.set_index('Month', inplace=True)
    group = group.asfreq('M')  # 월별 주기 설정

    if len(group) > 0:
        try:
            model = ExponentialSmoothing(group['판매량'], seasonal=None).fit()
            models[model_name] = model
            # print(f"{model_name} 모델 피팅 성공, 매개변수: {model.params}")
        except Exception as e:
            print(f"{model_name} 모델 피팅 중 에러 발생: {e}")
    else:
        # print(f"{model_name}의 유효한 데이터가 부족합니다: {len(group)}개의 데이터 포인트.")
        pass

# 예측할 경우
for model_name, model in models.items():
    forecast = model.forecast(steps=12)  # 다음 12개월 예측
    if forecast.isna().any():
        # print(f"{model_name}의 예측 결과가 NaN입니다.")
        pass
    else:
        print(f"{model_name}의 예측: {forecast}")
        
        
        
# 모델의 복잡성:

# 단순한 모델 (예: 단순 이동 평균, 지수 평활법 등)에는 몇 개월에서 1년 정도의 데이터로도 충분할 수 있습니다.
# 복잡한 모델 (예: ARIMA, SARIMA, LSTM 등)에는 적어도 3년 이상의 데이터가 필요합니다. 특히 계절성을 고려하는 모델은 계절 주기보다 2배 이상의 데이터가 필요합니다.
# 데이터의 주기성:

# 일간 데이터: 1~3년
# 주간 데이터: 2~5년
# 월간 데이터: 3~10년
# 연간 데이터: 10년 이상