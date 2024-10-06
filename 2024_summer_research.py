# -*- coding: utf-8 -*-
"""2024 Summer Research
# Central Question

What team factors/statistics are most instrumental in determining March Madness success?
"""

#importing dataset with kaggle
od.download("https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset?select=cbb.csv")
# e481fdb72b7c584a04d2deb7e13f0fc2

#reading and displaying file

dataset = pd.read_csv('/content/college-basketball-dataset/cbb.csv')

dataset.dropna(subset=['SEED'], inplace = True) #deleting all rows with seed as NA
dataset.to_csv('/content/college-basketball-dataset/cbb.csv', index = False)
display(dataset)

"""#Exploring Data

**Summary Statistics**
1. compute *mean* of each numerical column
2. compute *median* of each numerical column
3. compute *standard deviation* of each numerical column


**Visualizations**
1. create *scatter plots* to analyze the relationship between certain features
"""

#importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

!pip install opendatasets
!pip install pandas

import opendatasets as od
import pandas
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

#mean calculations
print("Mean G: ", dataset["G"].mean()) #games played
print("Mean W: ", dataset["W"].mean()) #games won
print("Mean ADJOE: ", dataset["ADJOE"].mean()) #adjusted offensive efficiency (points scored per 100 possessions)
print("Mean ADJDE: ", dataset["ADJDE"].mean()) #adjusted defensive efficiency (points allowed per 100 possessions)
print("Mean BARTHAG: ", dataset["BARTHAG"].mean()) #power rating (chance of beating an average D1 team)
print("Mean EFG_O: ", dataset["EFG_O"].mean()) #effective field goal percentage shot
print("Mean EFG_D:", dataset["EFG_D"].mean()) #effective field goal percentage allowed
print("Mean TOR: ", dataset["TOR"].mean()) #turnover rate
print("Mean TORD: ", dataset["TORD"].mean()) #steal rate
print("Mean ORB: ", dataset["ORB"].mean()) #offensive rebound rate
print("Mean DRB: ", dataset["DRB"].mean()) #offensive rebound rate allowed
print("Mean FTR: ", dataset["FTR"].mean()) #free throw rate
print("Mean FTRD: ", dataset["FTRD"].mean()) #free throw rate allowed
print("Mean 2P_O: ", dataset["2P_O"].mean()) #2 point shooting percentage
print("Mean 2P_D: ", dataset["2P_D"].mean()) #2 point shooting percentage allowed
print("Mean 3P_O: ", dataset["3P_O"].mean()) #3 point shooting percentage
print("Mean 3P_D: ", dataset["3P_D"].mean()) #3 point shooting percentage allowed
print("Mean ADJ_T: ", dataset["ADJ_T"].mean()) #adjusted tempo (possessions a team has per game )
print("Mean WAB: ", dataset["WAB"].mean()) #wins above bubble
print("Mean SEED: ", dataset["SEED"].mean()) #ranked position

#median calculations
print("Median G: ", dataset["G"].median()) #games played
print("Median W: ", dataset["W"].median()) #games won
print("Median ADJOE: ", dataset["ADJOE"].median()) #adjusted offensive efficiency (points scored per 100 possessions)
print("Median ADJDE: ", dataset["ADJDE"].median()) #adjusted defensive efficiency (points allowed per 100 possessions)
print("Median BARTHAG: ", dataset["BARTHAG"].median()) #power rating (chance of beating an average D1 team)
print("Median EFG_O: ", dataset["EFG_O"].median()) #effective field goal percentage shot
print("Median EFG_D:", dataset["EFG_D"].median()) #effective field goal percentage allowed
print("Median TOR: ", dataset["TOR"].median()) #turnover rate
print("Median TORD: ", dataset["TORD"].median()) #steal rate
print("Median ORB: ", dataset["ORB"].median()) #offensive rebound rate
print("Median DRB: ", dataset["DRB"].median()) #offensive rebound rate allowed
print("Median FTR: ", dataset["FTR"].median()) #free throw rate
print("Median FTRD: ", dataset["FTRD"].median()) #free throw rate allowed
print("Median 2P_O: ", dataset["2P_O"].median()) #2 point shooting percentage
print("Median 2P_D: ", dataset["2P_D"].median()) #2 point shooting percentage allowed
print("Median 3P_O: ", dataset["3P_O"].median()) #3 point shooting percentage
print("Median 3P_D: ", dataset["3P_D"].median()) #3 point shooting percentage allowed
print("Median ADJ_T: ", dataset["ADJ_T"].median()) #adjusted tempo (possessions a team has per game )
print("Median WAB: ", dataset["WAB"].median()) #wins above bubble
print("Median: ", dataset["SEED"].median()) #ranked position

#standard deviation calculations
print("Standard Deviation G: ", dataset["G"].std()) #games played
print("Standard Deviation W: ", dataset["W"].std()) #games won
print("Standard Deviation ADJOE: ", dataset["ADJOE"].std()) #adjusted offensive efficiency (points scored per 100 possessions)
print("Standard Deviation ADJDE: ", dataset["ADJDE"].std()) #adjusted defensive efficiency (points allowed per 100 possessions)
print("Standard Deviation BARTHAG: ", dataset["BARTHAG"].std()) #power rating (chance of beating an average D1 team)
print("Standard Deviation EFG_O: ", dataset["EFG_O"].std()) #effective field goal percentage shot
print("Standard Deviation EFG_D:", dataset["EFG_D"].std()) #effective field goal percentage allowed
print("Standard Deviation TOR: ", dataset["TOR"].std()) #turnover rate
print("Standard Deviation TORD: ", dataset["TORD"].std()) #steal rate
print("Standard Deviation ORB: ", dataset["ORB"].std()) #offensive rebound rate
print("Standard Deviation DRB: ", dataset["DRB"].std()) #offensive rebound rate allowed
print("Standard Deviation FTR: ", dataset["FTR"].std()) #free throw rate
print("Standard Deviation FTRD: ", dataset["FTRD"].std()) #free throw rate allowed
print("Standard Deviation 2P_O: ", dataset["2P_O"].std()) #2 point shooting percentage
print("Standard Deviation 2P_D: ", dataset["2P_D"].std()) #2 point shooting percentage allowed
print("Standard Deviation 3P_O: ", dataset["3P_O"].std()) #3 point shooting percentage
print("Standard Deviation 3P_D: ", dataset["3P_D"].std()) #3 point shooting percentage allowed
print("Standard Deviation: ", dataset["ADJ_T"].std()) #adjusted tempo (possessions a team has per game )
print("Standard Deviation: ", dataset["WAB"].std()) #wins above bubble
print("Standard Deviation: ", dataset["SEED"].std()) #ranked position

#scatter plots
sns.scatterplot(data=dataset, x="ADJOE", y="ADJDE", hue="G", size="G")

sns.scatterplot(data=dataset, x="3P_O", y="BARTHAG")

sns.scatterplot(data=dataset, x="ADJOE", y="SEED")

sns.scatterplot(data=dataset, x="2P_O", y="ADJ_T")

sns.scatterplot(data=dataset, x="2P_D", y="ADJ_T")

sns.scatterplot(data=dataset, x="ADJ_T", y="BARTHAG")

sns.scatterplot(data=dataset, x="EFG_O", y="ADJ_T", hue="POSTSEASON", size="POSTSEASON")

sns.barplot(dataset, x="POSTSEASON", y="ADJOE")

sns.barplot(dataset, x="POSTSEASON", y="ORB")

sns.barplot(dataset, x="POSTSEASON", y="DRB")

"""#Data Preprocessing"""

#postseason adjustments

old_value = '2ND'
new_value = int(2)
column_name = 'POSTSEASON'
dataset[column_name] = dataset[column_name].replace(old_value, new_value)

old_value = 'R64'
new_value = int(64)
dataset[column_name] = dataset[column_name].replace(old_value, new_value)

old_value = 'E8'
new_value = int(8)
dataset[column_name] = dataset[column_name].replace(old_value, new_value)

old_value = 'Champions'
new_value = int(1)
dataset[column_name] = dataset[column_name].replace(old_value, new_value)

old_value = 'F4'
new_value = int(4)
dataset[column_name] = dataset[column_name].replace(old_value, new_value)

old_value = 'R32'
new_value = int(32)
dataset[column_name] = dataset[column_name].replace(old_value, new_value)

old_value = 'S16'
new_value = int(16)
dataset[column_name] = dataset[column_name].replace(old_value, new_value)

old_value = 'R68'
new_value = int(68)
dataset[column_name] = dataset[column_name].replace(old_value, new_value)

dataset.to_csv('modified_file.csv', index=False)
display(dataset)

#split data into training and testing sets
#columns_to_convert = ['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O', '3P_D', 'ADJ_T', 'WAB', 'SEED']
dataset['G'] = dataset['G'].astype(float)
dataset['W'] = dataset['W'].astype(float)
dataset['ADJOE'] = dataset['ADJOE'].astype(float)
dataset['ADJDE'] = dataset['ADJDE'].astype(float)
dataset['BARTHAG'] = dataset['BARTHAG'].astype(float)
dataset['EFG_O'] = dataset['EFG_O'].astype(float)
dataset['EFG_D'] = dataset['EFG_D'].astype(float)
dataset['TOR'] = dataset['TOR'].astype(float)
dataset['TORD'] = dataset['TORD'].astype(float)
dataset['ORB'] = dataset['ORB'].astype(float)
dataset['DRB'] = dataset['DRB'].astype(float)
dataset['FTR'] = dataset['FTR'].astype(float)
dataset['FTRD'] = dataset['FTRD'].astype(float)
dataset['2P_O'] = dataset['2P_O'].astype(float)
dataset['2P_D'] = dataset['2P_D'].astype(float)
dataset['3P_O'] = dataset['3P_O'].astype(float)
dataset['3P_D'] = dataset['3P_D'].astype(float)
dataset['ADJ_T'] = dataset['ADJ_T'].astype(float)
dataset['WAB'] = dataset['WAB'].astype(float)
dataset['SEED'] = dataset['SEED'].astype(float)
dataset['POSTSEASON'] = dataset['POSTSEASON'].astype(float)

"""#Modeling

Linear Regression and Random Forest
"""

df = pd.DataFrame(dataset)

X = df[['WAB']]  # Feature must be 2D
y = df['SEED']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R-squared:', r2)

# Make predictions on new data (or the same test data for visualization)
y_pred_full = model.predict(X)

# Plot the results
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred_full, color='red', label='Linear fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Print model coefficients
print('Intercept:', model.intercept_)
print('Slope:', model.coef_[0])

selected_features = ['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D', 'TOR', 'TORD', 'ORB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O', '3P_D',
                     'ADJ_T','WAB', 'SEED']
X = df[selected_features]
y = df['POSTSEASON']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R-squared:', r2)

feature_importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()
