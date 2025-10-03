import pandas as pd
import numpy as np

print(pd.__version__)

filename = '/workspaces/data/car_fuel_efficiency_1.csv'

df = pd.read_csv(filename)

num_rows = len(df)
print('Number of rows = ', num_rows)

num_fuel_type = df.fuel_type.nunique()

print('Number of fuel types = ', num_fuel_type)

total_missing = df.isnull().any().sum()

print('Number of collumns with missing values = ', total_missing)

max_car_fuel_efficiency = df.groupby(['origin']).fuel_efficiency_mpg.max()

print('Max car fuel efficiency= ', max_car_fuel_efficiency)


median_value_of_horsepower = df.horsepower.median()

print('Median value of horsepower = ', median_value_of_horsepower)

max_count_values_of_horsepower = df.horsepower.value_counts().index[0]

print('Most frequent value of horsepower = ', max_count_values_of_horsepower)

median_replaced_most_frequent_value = df.horsepower.fillna(max_count_values_of_horsepower).median()

print('Median replacing missing values = ', median_replaced_most_frequent_value)

cars_from_asia = df[df['origin']=='Asia']
X = cars_from_asia[['vehicle_weight', 'model_year']].reset_index().loc[:6].to_numpy()[:,1:]
y_data = [1100, 1300, 800, 900, 1000, 1100, 1200]
y = np.array(y_data)

# Linear regression

XTX = np.matmul(X.T, X) 
X_inv = np.linalg.inv(XTX)
matrix = np.matmul(X_inv, X.T)
w = np.matmul(matrix, y)
print('Sum of weights = ', w.sum())