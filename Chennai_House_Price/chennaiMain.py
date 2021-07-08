import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor


df = pd.read_csv('train.csv')
print(df.shape)

df = df.drop(['PRT_ID', 'DATE_SALE', 'SALE_COND', 'DATE_BUILD', 'BUILDTYPE', 'UTILITY_AVAIL', 'STREET', 'MZZONE', 'QS_ROOMS', 'QS_BATHROOM', 'QS_BEDROOM', 'QS_OVERALL', 'PARK_FACIL'], axis='columns')
print(df.head())

print(df.isnull().sum())

# dropping all na values
df = df.dropna()
print(df.isnull().sum())

# Examine locations which is a categorical variable. We need to apply dimensionality reduction technique here to reduce number of locations
df.AREA = df.AREA.apply(lambda x: x.strip())
location_stats = df['AREA'].value_counts(ascending=False)  # to see location statastics
print(location_stats)

# Any location having less than 10 data points should be tagged as "other" location. This way number of categories can be reduced by huge amount.
# Later on when we do one hot encoding, it will help us with having fewer dummy columns

location_stats_less_than_10 = location_stats[location_stats <= 10]
print(location_stats_less_than_10)

df.AREA = df.AREA.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
print(len(df.AREA.unique()))

# One hot encoding of location
dummies = pd.get_dummies(df.AREA)

# to drop others column in dataframe

df = pd.concat([df, dummies.drop(['other'], axis='columns')], axis='columns')
# to remove location columns
df.drop(['AREA'], axis='columns', inplace=True)
print(df.head())


# Build a Model

X = df.drop(['SALES_PRICE'], axis='columns')
print(X)
y = df.SALES_PRICE
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(len(X_train))

model = LinearRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(score)

# Use K Fold cross validation to measure accuracy of our LinearRegression model

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

crossValScore = cross_val_score(LinearRegression(), X, y, cv=cv)
print(crossValScore)

def predict_price(location, INT_SQFT, DIST_MAINROAD, N_BEDROOM, N_BATHROOM, N_ROOM):
    loc_index = np.where(X.columns == location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = INT_SQFT
    x[1] = DIST_MAINROAD
    x[2] = N_BEDROOM
    x[3] = N_BATHROOM
    x[4] = N_ROOM
    if loc_index >= 0:
        x[loc_index] = 1

    print(model.predict([x])[0])

predict_price('Karapakkam', 3000, 131, 2, 2, 3)

import pickle
with open('Chennai_home_prices_model.pickle','wb') as f:
    pickle.dump(model, f)

# Export location and column information to a file that will be useful later on in our prediction application
import json
columns1 = {
    'data_columns': [col.lower() for col in X.columns]
}
with open("columns1.json", "w") as f:
    f.write(json.dumps(columns1))

