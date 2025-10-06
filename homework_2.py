import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


base = ['engine_displacement','horsepower','vehicle_weight','model_year']

def eda(df):
    # EDA
    plt.figure(figsize=(6,4))
    sns.histplot(df.fuel_efficiency_mpg, bins=40, color='b', alpha=0.5)
    plt.ylabel('Frequency')
    plt.xlabel('Fuel efficiency')
    plt.title('Distribution of fuel efficiency')
    plt.savefig("x.jpg")


def prepare_X(df, value_to_fill):
    df = df.copy()
    df_num = df[base]
    df_num = df_num.fillna(value_to_fill)
    X = df_num.values
    return X

def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]

def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]

def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)

filename = '/workspaces/data/car_fuel_efficiency_2.csv'

df = pd.read_csv(filename)

df = df[['engine_displacement','horsepower','vehicle_weight','model_year','fuel_efficiency_mpg']]

# eda(df)

missing_columns = df.isnull().any()
print(missing_columns)

df.horsepower.fillna(0)
print(df.horsepower.median())

np.random.seed(42)
n = len(df)

n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - n_val - n_test

idx = np.arange(n)
np.random.shuffle(idx)

df_shuffled = df.iloc[idx]
df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
df_test = df_shuffled.iloc[n_train+n_val:].copy()

y_train = df_train.fuel_efficiency_mpg.values
y_val = df_val.fuel_efficiency_mpg.values
y_test = df_test.fuel_efficiency_mpg.values

del df_train['fuel_efficiency_mpg']
del df_val['fuel_efficiency_mpg']
del df_test['fuel_efficiency_mpg']

## fill with zeros

X_train = prepare_X(df_train, 0)
w_0, w = train_linear_regression(X_train, y_train)
y_pred = w_0 + X_train.dot(w)

plt.figure(figsize=(6,4))
sns.histplot(df.fuel_efficiency_mpg, bins=40, color='b', alpha=0.5)
sns.histplot(y_pred, bins=40, color='r', alpha=0.5)
plt.ylabel('Frequency')
plt.xlabel('Fuel efficiency')
plt.title('Distribution of fuel efficiency')
score = rmse(y_train, y_pred)
print('RMSE fill with zeros:',round(score,2))

## fill with the mean of horsepower

mean_horsepower = df_train.horsepower.mean()
X_train = prepare_X(df_train, mean_horsepower)
w_0, w = train_linear_regression(X_train, y_train)
y_pred_mean = w_0 + X_train.dot(w)
sns.histplot(y_pred_mean, bins=40, color='g', alpha=0.5)
plt.savefig("z.jpg")
score = rmse(y_train, y_pred_mean)
print('RMSE fill with mean', round(score,2))


X_train = prepare_X(df_train, 0)
for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    w_0, w = train_linear_regression_reg(X_train, y_train, r=r)
    y_pred = w_0 + X_train.dot(w)
    score = rmse(y_train, y_pred)
    print('RMSE r = ', r, round(score,2))


score_results = []
for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:

    df = pd.read_csv(filename)
    df = df[['engine_displacement','horsepower','vehicle_weight','model_year','fuel_efficiency_mpg']]
    np.random.seed(seed)
    
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_shuffled = df.iloc[idx]
    df_train = df_shuffled.iloc[:n_train].copy()
    df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
    df_test = df_shuffled.iloc[n_train+n_val:].copy()

    y_train = df_train.fuel_efficiency_mpg.values
    y_val = df_val.fuel_efficiency_mpg.values
    y_test = df_test.fuel_efficiency_mpg.values

    del df_train['fuel_efficiency_mpg']
    del df_val['fuel_efficiency_mpg']
    del df_test['fuel_efficiency_mpg']

    X_train = prepare_X(df_train, 0)
    X_val = prepare_X(df_val, 0)
    w_0, w = train_linear_regression(X_train, y_train)
    y_pred = w_0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    score_results.append(score)

print("Score std: ", round(np.std(score_results),3))

df = pd.read_csv(filename)
df = df[['engine_displacement','horsepower','vehicle_weight','model_year','fuel_efficiency_mpg']]
np.random.seed(9)
    
idx = np.arange(n)
np.random.shuffle(idx)

df_shuffled = df.iloc[idx]
df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
df_test = df_shuffled.iloc[n_train+n_val:].copy()

y_train = df_train.fuel_efficiency_mpg.values
y_val = df_val.fuel_efficiency_mpg.values
y_test = df_test.fuel_efficiency_mpg.values

del df_train['fuel_efficiency_mpg']
del df_val['fuel_efficiency_mpg']
del df_test['fuel_efficiency_mpg']


df_train_and_val = pd.concat([df_train, df_val])
y_train_and_val = np.concatenate([y_train,y_val], axis = 0)
X_train_and_val = prepare_X(df_train_and_val, 0)
X_test = prepare_X(df_test, 0)
w_0, w = train_linear_regression(X_train_and_val, y_train_and_val)
y_pred = w_0 + X_test.dot(w)
score = rmse(y_test, y_pred)

print("RMSE on test:", round(score,3))