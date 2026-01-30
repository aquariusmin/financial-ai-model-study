import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

#지도학습: 서포트 벡터 머신(SVM)
#1. Baby Example
babyexample = {
    'income': [30, 55, 63, 35, 28, 140, 100, 95, 64, 63],
    'credit': [40, 30, 30, 80, 100, 30, 30, 90, 120, 150],
    'loan': [-1, -1, 1, -1, -1, 1, 1, -1, 1, 1]
}
df = pd.DataFrame(data=babyexample)

baby_x = df[['income', 'credit']]
baby_y = df['loan']

C_values = [0.01, 0.001, 0.0005, 0.0003, 0.0002]
results = []

colors = df['loan'].map({-1: 'red', 1: 'blue'}).values

for i, C in enumerate(C_values):
    clf = SVC(kernel='linear', C=C, tol=1e-5)
    clf.fit(baby_x, baby_y)

    baby_w = clf.coef_[0]
    baby_b = -clf.intercept_[0]
    baby_w1, baby_w2 = baby_w[0], baby_w[1]

    baby_x1 = np.linspace(0, 160, 100)
    y0 = (baby_b - baby_w1 * baby_x1) / baby_w2
    y1 = (baby_b + 1 - baby_w1 * baby_x1) / baby_w2
    y2 = (baby_b - 1 - baby_w1 * baby_x1) / baby_w2

    predictions = clf.predict(baby_x)
    misclassified = (predictions != baby_y).sum()

    margin_width = 0 if np.linalg.norm(baby_w) == 0 else 2 / np.linalg.norm(baby_w)

    results.append({
        'C': C,
        'w1': baby_w1,
        'w2': baby_w2,
        'b': baby_b,
        '대출 오분류': misclassified,
        '통로 너비': round(margin_width, 4)
    })

    plt.scatter(df['income'], df['credit'], c=colors)
    plt.plot(baby_x1, y0, '-', color='green')
    plt.plot(baby_x1, y1, '--', color='green')
    plt.plot(baby_x1, y2, '--', color='green')
    plt.title(f"SVM with C={C}")
    plt.xlabel('Income')
    plt.ylabel('Adjusted Credit Score')
    plt.xlim(0, 160)
    plt.ylim(0, 350)
    plt.savefig(f'C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\SVM_C_{C}.png')
    plt.show()

results_df = pd.DataFrame(results)
results_df['통로 너비'] = results_df['통로 너비'].round(4)

#2. IOWA Example
iowa_train = pd.read_excel('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\IOWA_Training_Data.xlsx')
iowa_valid = pd.read_excel('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\IOWA_Validation_Data.xlsx')
iowa_test = pd.read_excel('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\IOWA_Test_Data.xlsx')

X_train = iowa_train[['GrLivArea', 'OverallQual']].values
Y_train = iowa_train['Sale Price'].values.ravel()
X_valid = iowa_valid[['GrLivArea', 'OverallQual']].values
Y_valid = iowa_valid['Sale Price'].values.ravel()
X_test = iowa_test[['GrLivArea', 'OverallQual']].values
Y_test = iowa_test['Sale Price'].values.ravel()

C_values = [0.0001, 0.001, 0.01, 0.1, 1]
epsilon_values = [50, 100, 200]

train_results = []
validation_results = []
test_results = []

for C in C_values:
    for epsilon in epsilon_values:
        model = LinearSVR(C=C, epsilon=epsilon, max_iter=4000000, dual=True)
        model.fit(X_train, Y_train)

        w1, w2 = model.coef_
        b = model.intercept_[0]

        predictions_train = model.predict(X_train)
        mse_train = mean_squared_error(Y_train, predictions_train)

        predictions_valid = model.predict(X_valid)
        mse_valid = mean_squared_error(Y_valid, predictions_valid)

        predictions_test = model.predict(X_test)
        mse_test = mean_squared_error(Y_test, predictions_test)

        #2-1) Train Data에 대한 결과
        train_results.append({
            'C': C,
            'Epsilon': epsilon,
            'w1': round(w1, 4),
            'w2': round(w2, 4),
            'b': round(b, 4),
            'MSE_Train': round(mse_train, 4),
            'Passage Width': round(2 * epsilon, 4)
        })

        #2-2) Validation Data에 대한 결과
        validation_results.append({
            'C': C,
            'Epsilon': epsilon,
            'MSE_Train': round(mse_train, 4),
            'MSE_Valid': round(mse_valid, 4),
            'Passage Width': round(2 * epsilon, 4),
            'MSE Difference': round(abs(mse_train - mse_valid), 4)
        })

        #2-4) Test Data에 대한 결과
        test_results.append({
            'C': C,
            'Epsilon': epsilon,
            'MSE_Test': round(mse_test, 4),
        })

train_results_df = pd.DataFrame(train_results)
validation_results_df = pd.DataFrame(validation_results)
test_results_df = pd.DataFrame(test_results)

with pd.ExcelWriter('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\IOWA_SVR_Results.xlsx') as writer:
    train_results_df.to_excel(writer, sheet_name='Train Results', index=False)
    validation_results_df.to_excel(writer, sheet_name='Validation Results', index=False)
    test_results_df.to_excel(writer, sheet_name='Test Results', index=False)
    results_df.to_excel(writer, sheet_name='Results', index=False) #1번 문제에 대한 결과 함께 저장

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
sc_train = ax.scatter(train_results_df['C'], train_results_df['Epsilon'], train_results_df['MSE_Train'],
                        c=train_results_df['MSE_Train'], cmap='viridis', marker='o')
ax.set_xlabel('C')
ax.set_ylabel('Epsilon')
ax.set_zlabel('MSE', labelpad=10)
ax.set_title('3D Plot Train Results')
cbar = plt.colorbar(sc_train, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label('MSE', labelpad=15)
cbar.ax.tick_params(labelsize=10, pad=10)
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\IOWA_SVR_Train_Results.png')
plt.show()

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
sc1 = ax.scatter(train_results_df['C'], train_results_df['Epsilon'], train_results_df['MSE_Train'],
                 c=train_results_df['MSE_Train'], cmap='viridis', marker='o', label='MSE_Train')
sc2 = ax.scatter(validation_results_df['C'], validation_results_df['Epsilon'], validation_results_df['MSE_Valid'],
                 c=validation_results_df['MSE_Valid'], cmap='plasma', marker='^', label='MSE_Valid')
ax.set_xlabel('C')
ax.set_ylabel('Epsilon')
ax.set_zlabel('MSE', labelpad=10)
ax.set_title('3D Plot Comparison (Train vs Validation)')
cbar = plt.colorbar(sc1, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label('MSE', labelpad=15)
cbar.ax.tick_params(labelsize=10, pad=10)
plt.legend()
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\IOWA_SVR_Results.png')
plt.show()

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
sc_test = ax.scatter(test_results_df['C'], test_results_df['Epsilon'], test_results_df['MSE_Test'],
                    c=test_results_df['MSE_Test'], cmap='viridis', marker='o')
ax.set_xlabel('C')
ax.set_ylabel('Epsilon')
ax.set_zlabel('MSE', labelpad=10)
ax.set_title('3D Plot Test Results')
cbar = plt.colorbar(sc_test, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label('MSE', labelpad=15)
cbar.ax.tick_params(labelsize=10, pad=10)
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\IOWA_SVR_Test_Results.png')
plt.show()

best_valid_model = validation_results_df[validation_results_df['MSE_Valid'] == validation_results_df['MSE_Valid'].min()]
balanced_model = validation_results_df[validation_results_df['MSE Difference'] == validation_results_df['MSE Difference'].min()]
best_test_model = test_results_df[test_results_df['MSE_Test'] == test_results_df['MSE_Test'].min()]

param_grid = {'C': C_values, 'epsilon': epsilon_values}
model = LinearSVR(max_iter=25000000, dual=True)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, Y_train)
best_model = grid_search.best_estimator_

print("\n최적 파라미터")
print(f"Best Parameters: {grid_search.best_params_}")
print("\n최적 모델 정보")
print(f"Best Estimator: {grid_search.best_estimator_}")
print("\n검증 데이터 결과")
print(best_valid_model.to_string(index=False))
print("\nMSE 차이가 가장 작은 모델")
print(balanced_model.to_string(index=False))
print("\n테스트 데이터 결과")
print(best_test_model.to_string(index=False))
print("\n최적 모델 요약")
print(f"Best Model: {best_model}")
print(f"Best Model Score (Validation): {grid_search.best_score_:.4f}")


