import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

filename = '/workspaces/data/course_lead_scoring_3.csv'

df = pd.read_csv(filename)
missing_columns = df.isnull().any()

df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_').fillna('NA')

numerical_columns = ['number_of_courses_viewed', 'annual_income', 'interaction_count', 'lead_score']
print(df.head().T)
for c in numerical_columns:
    df[c] = df[c].fillna(0)

print(df['industry'].mode())

correlation_matrix = df[numerical_columns].corr()
print(correlation_matrix)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.converted.values
y_val = df_val.converted.values
y_test = df_test.converted.values

del df_train['converted']
del df_val['converted']
del df_test['converted']

def mutual_info_churn_score(series):
    return mutual_info_score(series, y_train)

mi = df_train[categorical_columns].apply(mutual_info_churn_score)
print(round(mi.sort_values(ascending=False),2))

dv = DictVectorizer(sparse=False)

def calculate_accuracy(series, C):
    
    train_dict = df_train[series].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    val_dict = df_val[series].to_dict(orient='records')
    X_val = dv.transform(val_dict)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    converted_decision = (y_pred >= 0.5)
    return (y_val == converted_decision).mean()


original_accuracy = calculate_accuracy(categorical_columns + numerical_columns, 1)

for c in categorical_columns:
    filtered_categories = [item for item in categorical_columns if item != c]
    print(c, abs(calculate_accuracy(filtered_categories + numerical_columns, 1) - original_accuracy))

for c in numerical_columns:
    filtered_categories = [item for item in numerical_columns if item != c]
    print(c, abs(calculate_accuracy(categorical_columns + filtered_categories, 1) - original_accuracy))

C = [0.01, 0.1, 1, 10, 100]

for param in C:
    print(param, round(calculate_accuracy(categorical_columns + numerical_columns, param),3))