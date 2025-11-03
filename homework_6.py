import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import statistics
import xgboost as xgb

def train_decision_tree(df_train, y_train, max_depth=None):
    train_dicts = df_train.to_dict(orient='records')
    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(train_dicts)
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=1)
    model.fit(X_train, y_train)
    return dv, model

def train_random_forest(df_train, y_train, n_estimators=1, max_depth=None):
    train_dicts = df_train.to_dict(orient='records')
    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(train_dicts)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=1, n_jobs=-1, max_depth=max_depth)
    model.fit(X_train, y_train)
    return dv, model

def predict(df, dv, model):
    dicts = df.to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict(X)
    return y_pred

def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)

# Open the data
filename = '../data/car_fuel_efficiency_6.csv'
df = pd.read_csv(filename)

# Pre-processing
df = df.fillna(0)

df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

numerical_columns = [c for c in df.columns if c not in categorical_columns]

for c in numerical_columns:
    df[c] = df[c].astype(np.int64)

# Train test split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.fuel_efficiency_mpg.values
y_val = df_val.fuel_efficiency_mpg.values
y_test = df_test.fuel_efficiency_mpg.values

del df_train['fuel_efficiency_mpg']
del df_val['fuel_efficiency_mpg']
del df_test['fuel_efficiency_mpg']


dv, model = train_decision_tree(df_train, y_train, max_depth=1)

print(export_text(model, feature_names=list(dv.get_feature_names_out())))

dv, model = train_random_forest(df_train, y_train, n_estimators=10, max_depth=None)

y_pred = predict(df_val, dv, model)
score = rmse(y_val, y_pred)
print(score)


scores=[]
for n in range(10, 201, 10):
    dv, model = train_random_forest(df_train, y_train, n_estimators=n, max_depth=None)
    y_pred = predict(df_val, dv, model)
    score = rmse(y_val, y_pred)
    scores.append((n,round(score,3)))

df_scores = pd.DataFrame(scores, columns=['n_estimators', 'rmse'])
plt.plot(df_scores.n_estimators, df_scores.rmse)
plt.savefig("random_forest_n_estimators.png")
plt.close()

mean_rmse = []


for d in [10, 15, 20, 25]:
    scores = []
    for n in range(10, 201, 10):
        dv, model = train_random_forest(df_train, y_train, n_estimators=n, max_depth=d)
        y_pred = predict(df_val, dv, model)
        score = rmse(y_val, y_pred)
        scores.append(score)
    mean_rmse.append((d, statistics.mean(scores)))
    print(d, statistics.mean(scores))

df_scores = pd.DataFrame(mean_rmse, columns=['max_depth', 'mean_rmse'])
plt.plot(df_scores.max_depth, df_scores.mean_rmse)
plt.savefig("random_forest_max_depth.png")
plt.close()

## Feature importance

dv, model = train_random_forest(df_train, y_train, n_estimators=10, max_depth=20)
features_importance_sorted = sorted(zip(df.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)

print(features_importance_sorted)

# XGBoost
features = list(dv.get_feature_names_out())
train_dicts = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
val_dicts = df_val.to_dict(orient='records')
X_val = dv.fit_transform(val_dicts)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

for eta in [0.3, 0.1]:
    xgb_params = {
    'eta': eta, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
    }
    model = xgb.train(xgb_params, dtrain, num_boost_round=100)
    y_pred = model.predict(dval)

    score = rmse(y_val, y_pred)
    print(eta, score)
