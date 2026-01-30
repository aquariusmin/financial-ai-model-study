import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#문제1
df = pd.read_excel('/Users/sangmin/Desktop/KW/금융과인공지능/data_salary_age.xlsx')

train = df[df['Type'] == 'train']
valid = df[df['Type'] == 'valid']
test = df[df['Type'] == 'test']
X_train, y_train = train[['Age']], train['Salary']
X_valid, y_valid = valid[['Age']], valid['Salary']
X_test, y_test = test[['Age']], test['Salary']

degrees = [1, 2, 3, 4, 5]

rmse_train = {}
rmse_valid = {}
rmse_test = {}

x_range_train = np.linspace(X_train['Age'].min(), X_train['Age'].max(), 100).reshape(-1, 1)
x_range_valid = np.linspace(X_valid['Age'].min(), X_valid['Age'].max(), 100).reshape(-1, 1)
x_range_test = np.linspace(X_test['Age'].min(), X_test['Age'].max(), 100).reshape(-1, 1)


def get_ordinal_suffix(degree):
    if degree % 10 == 1 and degree % 100 != 11:
        return 'st'
    elif degree % 10 == 2 and degree % 100 != 12:
        return 'nd'
    elif degree % 10 == 3 and degree % 100 != 13:
        return 'rd'
    else:
        return 'th'


def plot_polynomial_regression(x_range, X, y, set_type, rmse_dict, title):
    plt.figure(figsize=(10, 6))
    for degree in degrees:
        poly = PolynomialFeatures(degree=degree)

        X_poly = poly.fit_transform(X.values)

        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        rmse_dict[degree] = np.sqrt(mean_squared_error(y, y_pred))

        X_range_poly = poly.transform(x_range)
        y_range_pred = model.predict(X_range_poly)

        suffix = get_ordinal_suffix(degree)
        plt.plot(x_range, y_range_pred, label=f'{degree}{suffix} Polynomial Regression ({set_type})', linewidth=2)

    plt.scatter(X, y, label=f'{set_type} Data', color='green', s=20)
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Salary')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_polynomial_regression(x_range_train, X_train, y_train, 'Train', rmse_train,
                           'Polynomial Regression Fit (Training Set)')
plot_polynomial_regression(x_range_valid, X_valid, y_valid, 'Validation', rmse_valid,
                           'Polynomial Regression Fit (Validation Set)')
plot_polynomial_regression(x_range_test, X_test, y_test, 'Test', rmse_test, 'Polynomial Regression Fit (Test Set)')

rmse_table_train = pd.DataFrame({
    'Degree': degrees,
    'Training RMSE': [rmse_train[deg] for deg in degrees]
})

rmse_table_valid = pd.DataFrame({
    'Degree': degrees,
    'Validation RMSE': [rmse_valid[deg] for deg in degrees]
})

rmse_table_test = pd.DataFrame({
    'Degree': degrees,
    'Test RMSE': [rmse_test[deg] for deg in degrees]
})

print('\nTraining Set RMSE Values:')
print(rmse_table_train)
print('\nValidation Set RMSE Values:')
print(rmse_table_valid)
print('\nTest Set RMSE Values:')
print(rmse_table_test)

plt.figure(figsize=(10, 6))
plt.plot(degrees, [rmse_train[deg] for deg in degrees], label='Training RMSE', marker='o', color='blue', linewidth=2)
plt.plot(degrees, [rmse_valid[deg] for deg in degrees], label='Validation RMSE', marker='o', color='orange',
         linewidth=2)
plt.title('Training vs Validation RMSE by Polynomial Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.xticks(degrees)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(degrees, [rmse_train[deg] for deg in degrees], label='Training RMSE', marker='o', color='blue', linewidth=2)
plt.plot(degrees, [rmse_valid[deg] for deg in degrees], label='Validation RMSE', marker='o', color='orange',
         linewidth=2)
plt.plot(degrees, [rmse_test[deg] for deg in degrees], label='Test RMSE', marker='o', color='red', linewidth=2)

plt.title('Training, Validation, and Test RMSE by Polynomial Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.xticks(degrees)
plt.show()

#문제 2
data = pd.read_csv('/Users/sangmin/Desktop/KW/금융과인공지능/countryriskdata.csv')

data_subset = data[['Corruption', 'Peace', 'Legal', 'GDP Growth']]

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_subset)

pca = PCA(n_components=4)
principalComponents = pca.fit_transform(data_scaled)

loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4'], index=data_subset.columns)

explained_variance = pca.explained_variance_ratio_

print("요인 로딩:\n", loadings)
print("\n설명된 분산 비율(EVR):\n", explained_variance)

plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, 5), explained_variance, 'o-', color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.grid(True)
plt.show()
