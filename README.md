# week4-data-exploration-ml
week4-data-exploration-ml
 Russell Parkin 
 AAI 201 
 instructor: Becky Deitenbeck
 Submission date 2/10/26

 # Assignment Overview
Practice feature engineering, data exploration and modeling. 

# download dataset and sklearn resources.

!mamba install pandas numpy matplotlib scikit-learn
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_error, r2_score
data = fetch_california_housing()

Explore the dataset:

   # Display the first few rows and summary statistics.
    print("--- First 5 Rows ---")
df.head()
    print("\n--- Summary Statistics ---")
df.describe()
  #  Identify categorical and numerical features.
kb = KBinsDiscretizer(n_bins=10, strategy='uniform', encode='onehot-dense')
rooms_binned = kb.fit_transform(df[['AveRooms']])
    data = fetch_california_housing()

 # Visualize at least two features
plt.scatter(df['AveRooms'], df['MedHouseVal'])
plt.xlabel('Average number of rooms (AveRooms)')
plt.ylabel('House Price (MedHouseVal)')
plt.title('Rooms vs. Price')
plt.show()

plt.figure(figsize=(8, 5))
df['MedHouseVal'].hist(bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Median House Values')
plt.xlabel('Price (in $100,000s)')
plt.ylabel('Frequency (Number of Neighborhoods)')

# Apply at least one encoding technique to categorical variables
kb = KBinsDiscretizer(n_bins=10, strategy='uniform', encode='onehot-dense')

# Create at least one new feature
df_binned = pd.DataFrame(rooms_binned, columns=[f'AveRooms_bin_{i}' 
explanation: I used binning because house age is noisy data this reduces the year to year change.

# Split your data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model 
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate your model using accuracy or another appropriate metric.
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))
An R2 score of 0.58 means that your model can explain approximately 58% of the variance in California house prices

# reflection 
The California Housing dataset predicts median home values based on neighborhood attributes like income and age. Feature engineering involved creating a binary "high room count" comparison and binning the "average rooms" variable into ten categorical columns. What surprised me was that this decreased the accuracy of the model the R2 score dropped from 0.58 to 0.56.My decision to add binned features slightly reduced regression performance, though the classification model performed well. The model used Linear Regression to predict prices and Logistic Regression to classify expensive homes with 83% accuracy. to improve I might try a decision tree instead to deal with so many categories. 


