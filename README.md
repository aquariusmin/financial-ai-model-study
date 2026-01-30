Applied Machine Learging & Financial Modeling Deep-Dive

본 저장소는 '금융과 인공지능' 과정을 통해 머신러닝의 핵심 알고리즘을 직접 구현하고, 금융 및 공공 데이터를 활용하여 모델의 성능과 한계를 심도 있게 분석한 프로젝트 모음입니다.

1. 프로젝트 목적 (Objective)
단순한 라이브러리 활용을 넘어, 각 알고리즘의 파라미터 변화가 모델 성능에 미치는 영향을 정량적으로 분석하고, 데이터 특성에 최적화된 모델을 선정하는 데이터 과학적 의사결정 역량을 기르는 것을 목적으로 합니다.

2. 주요 과제별 수행 내용 (Key Tasks)
- Task 1: 다항 회귀(Polynominal Regression) 및 주성분 분석(PCA)
    > 다항 회귀: 급여와 연령 데이터를 활용하여 1차부터 5차까지의 다항 회귀 모형을 구축하고, RMSE를 기준으로 과적합(Overfiting)을 최소화 하는 최적의 차수 선정.
    > PCA: 국가 데이터(Corruption, Peace, Legal 등)의 차원을 축소하여 핵심 동인을 파악하고, Scree Plot을 통해 데이터 분산의 88% 이상을 설명하는 주성분 추출.

- Task 2: 의사결정나무(Decision Tree) 최적화 분석
  > max_depth와 min_samples_split 파라미터 변화에 따른 Train/Test AUC 변화 추이를 시각화.
  > 모델 복잡도와 일반화 성능 사이의 Trade-off를 분석하여 최적의 트리 구조 도출.

- Task 3: SVM & Linear SVR 기반 예측
  > 소프트 마진 SVM의 파라미터 C값 변화에 따른 결정 경계 및 마진 너비 변화 분석.
  > IOWA 주택 가격 데이터를 활용하여 Linear SVR 모델의 하이퍼파라미터(C, epsilon) 튜닝 및 MSE 최소화 모델 구축.

- Task 4: 인공신경망(ANN/DNN) vs 선형 회귀 비교
  > Keras를 활용하여 단일 은닉층(ANN) 및 다중 은닉층(DNN) 모델 구축.
  > 선형 회귀 모델과의 성능 비교를 통해 데이터의 비선형성이 딥러닝 모델의 성능 향상에 기여하는 정도를 정량적으로 평가.
   
3. 핵심 역량 (Core Strenghts)
- 알고리즘 깊이: 모델의 내부 동작 원리를 이해하고 최적의 하이퍼파라미터를 찾아내는 분석력.
- 다양한 도구 활용: Python(scikit-learn, TensorFlow/Keras), R, Excel을 넘나드는 유연한 도구 활용 능력
- 시각화 기반 해석: ROC 커브, Loss 그래프, Scree Plot 등을 통해 분석 결과를 시각적으로 설득하는 역량
