import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from statsmodels.sandbox.regression.penalized import coef_restriction_meandiff
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input

#데이터 불러오기 및 전처리
iowa = pd.read_csv('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\Houseprice_data_scaled.csv')
y = iowa.pop('Sale Price')
x = iowa.copy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=100)

y_train = np.asarray(y_train)
y_valid = np.asarray(y_valid)
y_test = np.asarray(y_test)

#Keras를 사용한 신경망 모델 정의
input_num = x_train.shape[1]
hidden_num = 5
output_num = 1

model = keras.models.Sequential(name='HousePrice')
model.add(Input(shape=(input_num,), name='input_layer'))
model.add(Dense(hidden_num, activation='relu', name='hidden_layer'))
model.add(Dense(output_num, activation='linear', name='output_layer'))

model.summary()

#컴파일 최적화 (Adam)
learning_rate = 0.01
epoch = 500
batch = 100
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

ann_history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=epoch, batch_size=batch, verbose=1)

#학습 결과 출력
loss = model.evaluate(x_train, y_train, verbose=0)
print(f'Mean Squared Error on Training Data: {loss}')

predictions = model.predict(x_valid)

#학습 결과 시각화
plt.plot(np.log10(ann_history.history['loss']), label='Training Loss')
plt.plot(np.log10(ann_history.history['val_loss']), label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (ln)')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\ANN Loss.png')
plt.show()

#최적의 에폭 찾기
val_loss = ann_history.history['val_loss']
best_epoch_val_loss = np.argmin(val_loss) + 1
print(f'Epoch with lowest validation loss: {best_epoch_val_loss}, Validation Loss: {val_loss[best_epoch_val_loss-1]}')

#딥러닝 모델 구축
input_num = x_train.shape[1]
hidden_num = 5
output_num = 1

model = keras.models.Sequential(name='HousePrice')
model.add(Input(shape=(input_num,), name='input_layer'))
model.add(Dense(hidden_num, activation='relu', name='hidden_layer1'))
model.add(Dense(hidden_num, activation='relu', name='hidden_layer2'))
model.add(Dense(hidden_num, activation='relu', name='hidden_layer3'))
model.add(Dense(output_num, activation='linear', name='output_layer'))

model.summary()

#모델 컴파일
learning_rate = 0.01
epoch = 500
batch = 100
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

#모델 학습
dnn_history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=epoch, batch_size=batch, verbose=1)

#학습 결과 출력
loss = model.evaluate(x_train, y_train, verbose=0)
print(f'Mean Squared Error on Training Data: {loss}')

#학습 결과 시각화
plt.plot(np.log10(dnn_history.history['loss']), label='Training Loss')
plt.plot(np.log10(dnn_history.history['val_loss']), label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (ln)')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\DNN Loss.png')
plt.show()

#최적의 에폭 찾기
val_loss = dnn_history.history['val_loss']
best_epoch_val_loss = np.argmin(val_loss) + 1
print(f'Epoch with lowest validation loss: {best_epoch_val_loss}, Validation Loss: {val_loss[best_epoch_val_loss-1]}')

#선형 회귀 모델 구축
lr = LinearRegression()
lr.fit(x_train, y_train)
y_train_pred = lr.predict(x_train)
y_valid_pred = lr.predict(x_valid)
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_valid, y_valid_pred)
print(f'Mean Squared Error on Training Data: {train_mse}')
print(f'Mean Squared Error on Validation Data: {val_mse}')
print(f'Linear Regression Coefficients: {lr.coef_}')
print(f'Linear Regression Intercept: {lr.intercept_}')

#선형 회귀 모델 결과 시각화
plt.scatter(y_valid, y_valid_pred, alpha=0.6, s=20, label='Validation Data')
coef = np.polyfit(y_valid, y_valid_pred, 1)
reg_line = np.poly1d(coef)
plt.plot(y_valid, reg_line(y_valid), color='green', label='Regression Line')
plt.plot([min(y_valid), max(y_valid)], [min(y_valid), max(y_valid)], color='red', label='Perfect Prediction', linestyle='--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Linear Regression Model')
plt.grid(alpha=0.3)
plt.legend(fontsize=8, loc='lower right')
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\Linear Regression Model.png')
plt.show()