import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier, plot_tree

#1번
df = pd.read_excel('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\data_salary_age.xlsx')

valid = df[df['Type'] == 'valid'].copy().reset_index(drop=True)
test = df[df['Type'] == 'test'].copy().reset_index(drop=True)
x_valid, y_valid = valid[['Age']].reset_index(drop=True), valid['Salary'].reset_index(drop=True) / 1000
x_test, y_test = test[['Age']].reset_index(drop=True), test['Salary'].reset_index(drop=True) / 1000

#1-1)
poly = PolynomialFeatures(degree=5, include_bias=False)
Z = pd.DataFrame(poly.fit_transform(x_valid), columns=['X1', 'X2', 'X3', 'X4', 'X5'])
scaler = StandardScaler()
Z_standardized = pd.DataFrame(scaler.fit_transform(Z), columns=Z.columns)

Z_copy = Z.copy()
Z_std_copy = Z_standardized.copy()
Z_copy.loc['Mean'] = Z.mean()
Z_copy.loc['Std'] = Z.std()
Z_std_copy.loc['Mean'] = Z_standardized.mean()
Z_std_copy.loc['Std'] = Z_standardized.std()

pd.options.display.float_format = '{:.0f}'.format
print('\nUnstandardized Features:')
print(Z_copy)
pd.reset_option('display.float_format')
print('\nStandardized Features:')
print(Z_std_copy.round(4))

#1-2)
linear5 = LinearRegression(fit_intercept=True)
linear_std5 = LinearRegression(fit_intercept=True)
linear5.fit(Z, y_valid)
linear_std5.fit(Z_standardized, y_valid)

y_pred_valid5 = linear5.predict(Z)
y_pred_std_valid5 = linear_std5.predict(Z_standardized)
rmse_valid = mse(y_valid, y_pred_valid5) ** 0.5
rmse_std_valid = mse(y_valid, y_pred_std_valid5) ** 0.5

x1n = pd.DataFrame(np.linspace(25, 70, 100), columns=['Age'])
x1n_poly = pd.DataFrame(poly.transform(x1n), columns=Z.columns)
ols_pred_valid5 = linear5.predict(x1n_poly)
ols_pred_std_valid5 = linear_std5.predict(pd.DataFrame(scaler.transform(x1n_poly), columns=Z.columns))

plt.scatter(x_valid, y_valid, label='Valid Data', color='blue', s=20)
plt.scatter(x_valid, y_pred_valid5, label='Predicted', color='orange', s=20)
plt.plot(x1n, ols_pred_valid5, label='5th Degree Prediction (Valid)', color='red')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Salary (in thousands)')
plt.title('5th Degree Regression (before Standardized) - Valid Data')
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\5th degree regression before std (valid)')
plt.show()

plt.scatter(x_valid, y_valid, label='Valid Data', color='blue', s=20)
plt.scatter(x_valid, y_pred_std_valid5, label='Predicted', color='orange', s=20)
plt.plot(x1n, ols_pred_std_valid5, label='5th Degree Prediction (Valid)', color='red')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Salary (in thousands)')
plt.title('5th Degree Regression (after Standardized) - Valid Data')
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\5th degree regression after std (valid)')
plt.show()

equation = f'Y = {linear5.intercept_:.4f}'
for i, coef in enumerate(linear5.coef_, 1):
    equation += f' + {coef:.4f}X^{i}'
print('\n5차 다항 회귀 모형식 (정규화 전):', equation)
print('5차 모형 RMSE (Before Std):', rmse_valid.round(4))

equation_std = f'Y = {linear_std5.intercept_:.4f}'
for i, coef in enumerate(linear_std5.coef_, 1):
    equation_std += f' + {coef:.4f}X^{i}'
print('\n5차 다항 회귀 모형식 (정규화 후):', equation_std)
print('5차 모형 RMSE (After Std)', rmse_std_valid.round(4))

#1-3)
alphas = [0.02, 0.05, 0.1]
ridge_results = []
ridge_std_results = []
ridge_mse_list = []
ridge_mse_std_list = []

colors = ['blue', 'green', 'red']
markers = ['o', 's', 'D']


plt.figure(figsize=(10, 6))
for i, alpha in enumerate(alphas):
    ridge = Ridge(alpha=alpha)
    ridge.fit(Z, y_valid)
    ridge_pred = ridge.predict(Z)
    mse_ridge = mse(y_valid, ridge_pred)
    ridge_results.append([alpha, ridge.intercept_, *ridge.coef_, mse_ridge])
    ridge_mse_list.append(mse_ridge)

    plt.scatter(x_valid, ridge_pred, color=colors[i], marker=markers[i], s=20, label=f'λ = {str(alpha)}')

plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Ridge Regression Predictions for Different λ (Before Standardized)')
plt.legend(loc='best', fontsize='small')
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\ridge regression before std valid', bbox_inches='tight')
plt.show()

ridge_df = pd.DataFrame(ridge_results, columns=['λ', 'a', 'b1', 'b2', 'b3', 'b4', 'b5', 'MSE'])
print('\nRidge Regression Result (Before Standardized):')
print(ridge_df.round(4))

plt.figure(figsize=(10, 6))
for i, alpha in enumerate(alphas):
    ridge_std = Ridge(alpha=alpha)
    ridge_std.fit(Z_standardized, y_valid)
    ridge_std_pred = ridge_std.predict(Z_standardized)
    mse_std_ridge = mse(y_valid, ridge_std_pred)
    ridge_std_results.append([alpha, ridge_std.intercept_, *ridge_std.coef_, mse_std_ridge])
    ridge_mse_std_list.append(mse_std_ridge)

    plt.scatter(x_valid, ridge_std_pred, color=colors[i], marker=markers[i], s=20, label=f'λ = {alpha}')

plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Ridge Regression Predictions for Different λ (After Standardized)')
plt.legend(loc='best', fontsize='small')
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\ridge regression after std valid', bbox_inches='tight')
plt.show()

ridge_std_df = pd.DataFrame(ridge_std_results, columns=['λ', 'a', 'b1', 'b2', 'b3', 'b4', 'b5', 'MSE'])
print('\nRidge Regression Result (After Standardized):')
print(ridge_std_df.round(4))

lasso_results = []
lasso_std_results = []
lasso_mse_list = []
lasso_mse_std_list = []

#1-4)
plt.figure(figsize=(10, 6))
for i, alpha in enumerate(alphas):
    lasso = Lasso(alpha=alpha, max_iter=1100000)
    lasso.fit(Z, y_valid)
    lasso_pred = lasso.predict(Z)
    mse_lasso = mse(y_valid, lasso_pred)
    lasso_results.append([alpha, lasso.intercept_, *lasso.coef_, mse_lasso])
    lasso_mse_list.append(mse_lasso)

    plt.scatter(x_valid, lasso_pred, color=colors[i], marker=markers[i], s=20, label=f'λ = {alpha}')

plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Lasso Regression Predictions for Different λ (Before Standardized)')
plt.legend(loc='best', fontsize='small')
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\lasso regression before std valid', bbox_inches='tight')
plt.show()

lasso_df = pd.DataFrame(lasso_results, columns=['λ', 'a', 'b1', 'b2', 'b3', 'b4', 'b5', 'MSE'])
print('\nLasso Regression Result (Before Standardized):')
print(lasso_df.round(4))

plt.figure(figsize=(10, 6))
for i, alpha in enumerate(alphas):
    lasso_std = Lasso(alpha=alpha, max_iter=1100000)
    lasso_std.fit(Z_standardized, y_valid)
    lasso_std_pred = lasso_std.predict(Z_standardized)
    mse_std_lasso = mse(y_valid, lasso_std_pred)
    lasso_std_results.append([alpha, lasso_std.intercept_, *lasso_std.coef_, mse_std_lasso])
    lasso_mse_std_list.append(mse_std_lasso)

    plt.scatter(x_valid, lasso_std_pred, color=colors[i], marker=markers[i], s=20, label=f'λ = {alpha}')

plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Lasso Regression Predictions for Different λ (After Standardized)')
plt.legend(loc='best', fontsize='small')
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\lasso regression after std valid', bbox_inches='tight')
plt.show()

lasso_std_df = pd.DataFrame(lasso_std_results, columns=['λ', 'a', 'b1', 'b2', 'b3', 'b4', 'b5', 'MSE'])
print('\nLasso Regression Result (After Standardized):')
print(lasso_std_df.round(4))

#1-5)
plt.figure(figsize=(10,6))
plt.plot(alphas, lasso_mse_list, color='green', marker='o', markersize=4, label='Lasso mse (Before Standardized)')
plt.plot(alphas, lasso_mse_std_list, color='orange', marker='o', markersize=4, label='Lasso mse (After Standardized)')
plt.plot(alphas, ridge_mse_list, color='blue', marker='o', markersize=4, label='Ridge mse (Before Standardize)')
plt.plot(alphas, ridge_mse_std_list, color='red', marker='o', markersize=4, label='Ridge mse (After Standardized)')
plt.legend(loc='upper left', fontsize='small')
plt.xlabel('lambda (λ)')
plt.ylabel('MSE')
plt.title('MSE vs lambda for Lasso and Ridge Regression (Before and After Standardized)')
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\mse comparison lasso ridge')
plt.show()

table_data = {
    'Model': ['Vanilla', 'Lasso', 'Lasso', 'lasso', 'Ridge', 'Ridge', 'Ridge'],
    'λ': ['-', 0.02, 0.05, 0.1, 0.02, 0.05, 0.1],
    'MSE(valid)': [rmse_valid**2] + lasso_mse_list + ridge_mse_list,
    'MSE(std_valid)': [rmse_std_valid**2] + lasso_mse_std_list + ridge_mse_std_list
}
result_df = pd.DataFrame(table_data)
print('\n 표준화여부에 따른 MSE 비교', '\n', result_df.T.round(4))

Z_test = pd.DataFrame(poly.transform(x_test), columns=Z.columns)
Z_test_standardized = pd.DataFrame(scaler.transform(Z_test), columns=Z.columns)
y_pred_std_test5 = linear_std5.predict(Z_test_standardized)
rmse_std_test = mse(y_test, y_pred_std_test5) ** 0.5
ols_pred_std_test5 = linear_std5.predict(pd.DataFrame(scaler.transform(x1n_poly), columns=Z.columns))

plt.scatter(x_test, y_test, label='Test Data', color='blue', s=20)
plt.scatter(x_test, y_pred_std_test5, label='Predicted', color='orange', s=20)
plt.plot(x1n, ols_pred_std_test5, label='5th Degree Prediction (Test)', color='red')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('5th Degree Regression (Standardized) - Test Data')
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\5th degree regression (test)')
plt.show()

ridge_test_results = []
ridge_test_mse_list = []

plt.figure(figsize=(10,6))
for i, alpha in enumerate(alphas):
    ridge_test = Ridge(alpha=alpha)
    ridge_test.fit(Z_standardized, y_valid)
    ridge_pred_test = ridge_test.predict(Z_test_standardized)
    mse_test_ridge = mse(y_test, ridge_pred_test)
    ridge_test_results.append([alpha, ridge_test.intercept_, *ridge_test.coef_, mse_test_ridge])
    ridge_test_mse_list.append(mse_test_ridge)

    plt.scatter(x_test, ridge_pred_test, color=colors[i], marker=markers[i], s=20, label=f'λ = {alpha}')

plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Ridge Regression Predictions for Different λ (Test Set)')
plt.legend(loc='best')
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\ridge regression (test)', bbox_inches='tight')
plt.show()

lasso_test_results =[]
lasso_test_mse_list = []

plt.figure(figsize=(10,6))
for i, alpha in enumerate(alphas):
    lasso_test = Lasso(alpha=alpha, max_iter=1100000)
    lasso_test.fit(Z_standardized, y_valid)
    lasso_pred_test = lasso_test.predict(Z_test_standardized)
    mse_test_lasso = mse(y_test, lasso_pred_test)
    lasso_test_results.append([alpha, lasso_test.intercept_, *lasso_test.coef_, mse_test_lasso])
    lasso_test_mse_list.append(mse_test_lasso)

    plt.scatter(x_test, lasso_pred_test, color=colors[i], marker=markers[i], s=20, label=f'λ = {alpha}')

plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Lasso Regression Predictions for Different λ (Test)')
plt.legend(loc='best', fontsize='small')
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\lasso regression (test)', bbox_inches='tight')
plt.show()

model_comparison_table = {
    'Model' : ['Vanilla', 'Lasso', 'Lasso', 'Lasso', 'Ridge', 'Ridge', 'Ridge'],
    'λ' : ['-', 0.02, 0.05, 0.1, 0.02, 0.05, 0.1],
    'MSE (Valid)' : [rmse_std_valid ** 2] + lasso_mse_std_list + ridge_mse_std_list,
    'MSE(Test)' : [rmse_std_test ** 2] + lasso_test_mse_list + ridge_test_mse_list
}
model_comparison_df = pd.DataFrame(model_comparison_table)
print('\n모형 성과 비교:')
print(model_comparison_df.round(4).T)

plt.plot(alphas, [rmse_std_valid **2] * len(alphas), color='green', marker='o', markersize=4, label='Vanilla Valid MSE')
plt.plot(alphas, [rmse_std_test **2] * len(alphas), color='orange', marker='x', markersize=4, label='Vanilla Test MSE')
plt.plot(alphas, lasso_mse_std_list, color='blue', marker='o', markersize=4, label='lasso Valid MSE')
plt.plot(alphas, lasso_test_mse_list, color='red', marker='x', markersize=4, label='lasso Test MSE')
plt.plot(alphas, ridge_mse_std_list, color='black', marker='o', markersize=4, label='ridge Valid MSE')
plt.plot(alphas, ridge_test_mse_list, color='grey', marker='x', markersize=4, label='ridge Test MSE')
plt.legend(fontsize='small')
plt.xlabel('lambda (λ)')
plt.ylabel('MSE')
plt.title('Valid vs Test for All Regression')
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\all regression comparison')
plt.show()

with pd.ExcelWriter('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\analysis_results.xlsx') as writer:
    Z_copy.to_excel(writer, sheet_name='Z Table')
    Z_std_copy.to_excel(writer, sheet_name='Z Standardized')
    ridge_df.to_excel(writer, sheet_name='ridge results (valid)')
    ridge_std_df.to_excel(writer, sheet_name='ridge results (std valid)')
    lasso_df.to_excel(writer, sheet_name='lasso results (valid)')
    lasso_std_df.to_excel(writer, sheet_name='lasso results (std valid)')
    result_df.to_excel(writer, sheet_name='standardized mse comparison')
    model_comparison_df.to_excel(writer, sheet_name='model comparison results')

#2번
lending_train_df = pd.read_excel('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\lendingclub_traindata.xlsx')
lending_test_df = pd.read_excel('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\lendingclub_testdata.xlsx')
ld_x_train = lending_train_df.drop('loan_status', axis=1)
ld_x_test = lending_test_df.drop('loan_status', axis=1)
ld_y_train = lending_train_df['loan_status']
ld_y_test = lending_test_df['loan_status']

logi_reg = LogisticRegression(penalty=None, solver='newton-cg')
logi_reg.fit(ld_x_train, ld_y_train)
logi_reg_reverse = LogisticRegression(penalty=None, solver='newton-cg')
ld_y_train_reverse = np.where(ld_y_train == 1, 0, 1)
ld_y_test_reverse = np.where(ld_y_test == 1, 0, 1)

logi_reg_reverse.fit(ld_x_train, ld_y_train_reverse)
print('\nOriginal Definition - Intercept and Coefficients:')
print(logi_reg.intercept_, logi_reg.coef_)
print('Reverse Definition - Intercept and Coefficients:')
print(logi_reg_reverse.intercept_, logi_reg_reverse.coef_)

#2-2)
newx = sm.add_constant(ld_x_train)
model = sm.Logit(ld_y_train, newx)
result = model.fit(method='newton')
print('\n', result.summary())

ld_y_train_pred = logi_reg.predict_proba(ld_x_train)
ld_y_test_pred = logi_reg.predict_proba(ld_x_test)
mle_vector_train = np.log(np.where(ld_y_train == 1, ld_y_train_pred[:,1], ld_y_train_pred[:,0]))
mle_vector_test = np.log(np.where(ld_y_test == 1, ld_y_test_pred[:,1], ld_y_test_pred[:,0]))
cost_function_train = np.negative(np.sum(mle_vector_train)/len(ld_y_train))
cost_function_test = np.negative(np.sum(mle_vector_test)/len(ld_y_test))
print('\ncost function training set =', cost_function_train)
print('cost function test set =', cost_function_test)

ld_pred1 = np.where(logi_reg.predict_proba(ld_x_test)[:,1] > 0.25, 1, 0)
cm1 = (confusion_matrix(ld_y_test, ld_pred1, labels=[0, 1], sample_weight=None) / len(ld_y_test)) * 100
ld_pred2 = np.where(logi_reg.predict_proba(ld_x_test)[:,1] > 0.2, 1, 0)
cm2 = (confusion_matrix(ld_y_test, ld_pred2, labels=[0, 1], sample_weight=None) / len(ld_y_test)) * 100
ld_pred3 = np.where(logi_reg.predict_proba(ld_x_test)[:,1] > 0.15, 1, 0)
cm3 = (confusion_matrix(ld_y_test, ld_pred3, labels=[0, 1], sample_weight=None) / len(ld_y_test)) * 100

cm11 = pd.DataFrame(cm1, columns=['Predict No Default', 'Predict Default'], index=['Outcome No Default', 'Outcome Default'])
cm11.index.name = 'Z= 0.25'
cm21 = pd.DataFrame(cm2, columns=['Predict No Default', 'Predict Default'], index=['Outcome No Default', 'Outcome Default'])
cm21.index.name = 'Z = 0.20'
cm31 = pd.DataFrame(cm3, columns=['Predict No Default', 'Predict Default'], index=['Outcome No Default', 'Outcome Default'])
cm31.index.name = 'Z = 0.15'
print('\n', cm11)
print('\n', cm21)
print('\n', cm31)

THRESHOLD = [0.25, 0.20, 0.15]
ld_results = pd.DataFrame(columns=THRESHOLD, index=['Accuracy', 'True Pos Rate', 'True Neg Rate', 'False Pos Rate', 'Precision', 'F-Score'])

for z in THRESHOLD:
    ld_preds = np.where(logi_reg.predict_proba(ld_x_test)[:,1] > z, 1, 0)

    cm = confusion_matrix(ld_y_test, ld_preds, labels=[1, 0])
    TP = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    TN = cm[1, 1]

    accuracy = accuracy_score(ld_y_test, ld_preds)
    true_positive_rate = TP / (TP + FN) if (TP + FN) > 0 else 0
    true_negative_rate = TN / (FP + TN) if (FP + TN) > 0 else 0
    false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
    precision = precision_score(ld_y_test, ld_preds, pos_label=1, zero_division=0)
    f_score = f1_score(ld_y_test, ld_preds, pos_label=1, zero_division=0)

    ld_results[z] = [accuracy, true_positive_rate, true_negative_rate, false_positive_rate, precision, f_score]

print('\n', ld_results)

#2-3)
lr_prob = logi_reg.predict_proba(ld_x_test)[:,1]
lr_auc = roc_auc_score(ld_y_test, lr_prob)
ns_prob = [0 for _ in range(len(ld_y_test))]
ns_auc = roc_auc_score(ld_y_test, ns_prob)
print('AUC prediction from logistic regression model =', lr_auc)
ns_fpr, ns_tpr, _ = roc_curve(ld_y_test, ns_prob)
lr_fpr, lr_tpr, _ = roc_curve(ld_y_test, lr_prob)

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Prediction')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\roc curve')
plt.show()

print('\n', f'AUC (Logistic Regression) = {lr_auc:.4f}')

with pd.ExcelWriter('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\logistic regression results.xlsx') as writer:
    cm11.to_excel(writer, sheet_name='Z=0.25')
    cm21.to_excel(writer, sheet_name='Z=0.20')
    cm31.to_excel(writer, sheet_name='Z=0.15')
    ld_results.to_excel(writer, sheet_name='Performance Metrics')

#3번
clf = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_split=1000, min_samples_leaf=200, random_state=0)
clf.fit(ld_x_train, ld_y_train)

fig, ax = plt.subplots(figsize=(16, 9))
plot_tree(clf, filled=True, feature_names=ld_x_train.columns, proportion=True, class_names=['Default', 'No Default'])
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\decision tree')
plt.show()

importances = pd.DataFrame({'feature': ld_x_train.columns, 'importance': np.round(clf.feature_importances_, 3)})
importances = importances.sort_values('importance', ascending=False)
print("\nFeature Importances:\n", importances)

z = 0.9
dt_y_test_proba = clf.predict_proba(ld_x_test)[:, 1]
dt_y_pred = np.where(dt_y_test_proba > z, 1, 0)

cm = confusion_matrix(ld_y_test, dt_y_pred, labels=[1, 0])
cm = (cm / len(ld_y_test)) * 100
cm_df = pd.DataFrame(cm, columns=['Predict No Default', 'Predict Default'], index=['Outcome No Default', 'Outcome Default'])
cm_df.index.name = f'Z = {z}'
print("\nConfusion Matrix at Z = 0.9:\n", cm_df.round(4))

dt_THRESHOLD = [0, 0.75, 0.80, 0.85, 0.90, 1]
dt_results = pd.DataFrame(columns=["THRESHOLD", "accuracy", "true pos rate", "true neg rate", "false pos rate", "precision", "f-score"])
dt_results['THRESHOLD'] = dt_THRESHOLD
Q = clf.predict_proba(ld_x_test)[:, 1]

for i in dt_THRESHOLD:
    dt_preds = np.where(Q > i, 1, 0)
    cm = confusion_matrix(ld_y_test, dt_preds, labels=[1, 0])
    TP = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    TN = cm[1, 1]

    accuracy = accuracy_score(ld_y_test, dt_preds)
    true_positive_rate = TP / (TP + FN) if (TP + FN) > 0 else 0
    true_negative_rate = TN / (TN + FP) if (TN + FP) > 0 else 0
    false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
    precision = precision_score(ld_y_test, dt_preds, pos_label=1, zero_division=0)
    f_score = f1_score(ld_y_test, dt_preds, pos_label=1, zero_division=0)

    dt_results.loc[dt_results['THRESHOLD'] == i, 'accuracy'] = accuracy
    dt_results.loc[dt_results['THRESHOLD'] == i, 'true pos rate'] = true_positive_rate
    dt_results.loc[dt_results['THRESHOLD'] == i, 'true neg rate'] = true_negative_rate
    dt_results.loc[dt_results['THRESHOLD'] == i, 'false pos rate'] = false_positive_rate
    dt_results.loc[dt_results['THRESHOLD'] == i, 'precision'] = precision
    dt_results.loc[dt_results['THRESHOLD'] == i, 'f-score'] = f_score

dt_results_transposed = dt_results.set_index("THRESHOLD").T
print('\nPerformance Metrics:\n', dt_results_transposed)

dt_fpr, dt_tpr, _ = roc_curve(ld_y_test, Q)
dt_roc_auc = auc(dt_fpr, dt_tpr)
plt.figure(figsize=(8, 6))
plt.plot(dt_fpr, dt_tpr, color='darkorange', lw=1, label='Decision Tree (AUC = %0.4f)' % dt_roc_auc)
plt.plot([0,1], [0,1], color='navy', lw=1.5, linestyle='--', label='Random Prediction (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\roc curve decision tree')
plt.show()

with pd.ExcelWriter('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\decision tree results.xlsx') as writer:
    importances.to_excel(writer, sheet_name='Feature Importances')
    cm_df.to_excel(writer, sheet_name='Confusion Matrix')
    dt_results.to_excel(writer, sheet_name='Performance Metrics')

#optional problem
max_dept_values = [3, 5, 10, None]
min_samples_split_values = [2, 10, 50, 100]

op_results = []
op_colors = ['blure', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown']
color_index = 0

for max_dept in max_dept_values:
    for min_samples_split in min_samples_split_values:
        op_clf = DecisionTreeClassifier(criterion='entropy', max_depth=max_dept, min_samples_split=min_samples_split, random_state=0)
        op_clf.fit(ld_x_train, ld_y_train)

        op_y_train_proba = op_clf.predict_proba(ld_x_train)[:, 1]
        op_fpr_train, op_tpr_train, _ = roc_curve(ld_y_train, op_y_train_proba)
        op_roc_auc_train = auc(op_fpr_train, op_tpr_train)

        op_y_test_proba = op_clf.predict_proba(ld_x_test)[:, 1]
        op_fpr_test, op_tpr_test, _ = roc_curve(ld_y_test, op_y_test_proba)
        op_roc_auc_test = auc(op_fpr_test, op_tpr_test)

        op_results.append({
            'max_depth': max_dept,
            'min_samples_split': min_samples_split,
            'train_auc': op_roc_auc_train,
            'test_auc': op_roc_auc_test,
            'fpr_train': op_fpr_train,
            'tpr_train': op_tpr_train,
            'fpr_test': op_fpr_test,
            'tpr_test': op_tpr_test
        })

plt.figure(figsize=(12,8))
for op_result in op_results:
    max_dept = op_result['max_depth']
    min_samples_split = op_result['min_samples_split']
    op_fpr_test = op_result['fpr_test']
    op_tpr_test = op_result['tpr_test']
    op_roc_auc_test = op_result['test_auc']

    label = f'Test - depth={max_dept}, split={min_samples_split} (AUC={op_roc_auc_test:.2f})'
    plt.plot(op_fpr_test, op_tpr_test, color=colors[color_index % len(colors)], lw=1.5, label=label)
    color_index += 1

plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC Curves for Different max_depth and min_samples_split Combinations (Test Data)')
plt.legend(loc='lower right', fontsize='small')
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\combined roc curves')
plt.show()

plt.figure(figsize=(12,8))
color_index = 0

for op_result in op_results:
    max_dept = op_result['max_depth']
    min_samples_split = op_result['min_samples_split']
    op_fpr_train = op_result['fpr_train']
    op_tpr_train = op_result['tpr_train']
    op_roc_auc_train = op_result['train_auc']

    label = f'Train - depth={max_dept}, split={min_samples_split} (AUC={op_roc_auc_train:.2f})'
    plt.plot(op_fpr_train, op_tpr_train, color=colors[color_index % len(colors)], lw=1.5, label=label)
    color_index += 1

plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC Curves for Different max_depth and min_samples_split Combinations (Train Data)')
plt.legend(loc='lower right', fontsize='small')
plt.savefig('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\combined roc curves train')
plt.show()

op_results_df = pd.DataFrame(op_results)
op_results_df = op_results_df[['max_depth', 'min_samples_split', 'train_auc', 'test_auc']]
op_results_df = op_results_df.sort_values(by='test_auc', ascending=False).reset_index(drop=True)
print('\nAUC Results:', '\n', op_results_df.to_string(index=False))

with pd.ExcelWriter('C:\\Users\\Sangmin\\Desktop\\광운대학교\\금융과인공지능\\2nd sub results\\optional problem results.xlsx') as writer:
    op_results_df.to_excel(writer, sheet_name='AUC Results')
    for op_result in op_results:
        max_dept = op_result['max_depth']
        min_samples_split = op_result['min_samples_split']
        op_fpr_train = op_result['fpr_train']
        op_tpr_train = op_result['tpr_train']
        op_fpr_test = op_result['fpr_test']
        op_tpr_test = op_result['tpr_test']