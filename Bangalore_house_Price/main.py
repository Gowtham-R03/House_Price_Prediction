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

df = pd.read_csv('bengaluru_house_prices.csv')

print(df.head())
print(df.shape)

# to measure the area type present in df
print(df.groupby('area_type')['area_type'].agg('count'))

# to drop not useful columns
df = df.drop(['area_type', 'availability', 'society', 'balcony'], axis='columns')
print(df.head())

print(df.isnull().sum())

# dropping all na values
df = df.dropna()
print(df.isnull().sum())

# size column in df has to change some
print(df['size'].unique())  # from unique it shows unique values or features in that columns

df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
print(df.head())

df = df.drop(['size'], axis='columns')
print(df.head())

# explorating total_sqt column
print(df['total_sqft'].unique())  # from its row there many range values we need single value of this

# to convert sqrtft value to float
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

print(df[~df['total_sqft'].apply(is_float)].head()) # prints total_sqrft columns with range values

# convert range to float if the values is in sqrt or perch in words ignore that
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df1 = df.copy()
df1['total_sqft'] = df1['total_sqft'].apply(convert_sqft_to_num)
print(df1.head())

########## Feature Engineering

df2 = df1.copy()
# Add new feature called price per square feet
df2['price_per_sqft'] = df2['price']*100000/df2['total_sqft']
print(df2.head())

# handling location columns
print(len(df2.location.unique()))  # there 1300 to much to one hot encoding
# high dimensional columns will be there we wantt to reduce dimensions

# Examine locations which is a categorical variable. We need to apply dimensionality reduction technique here to reduce number of locations

df2.location = df2.location.apply(lambda x: x.strip())
location_stats = df2['location'].value_counts(ascending=False)  # to see location statastics
print(location_stats)

# Any location having less than 10 data points should be tagged as "other" location. This way number of categories can be reduced by huge amount.
# Later on when we do one hot encoding, it will help us with having fewer dummy columns

location_stats_less_than_10 = location_stats[location_stats <= 10]
print(location_stats_less_than_10)

df2.location = df2.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
print(len(df2.location.unique()))

print(df2.head())

# outlier removal using standard deviation and Mean:
# ouliers are data error sometimes valid but causes extreme variation in data

# minimum sqrft neede for bedroom is 300 if total sqrft divided by bhk is less than 300 that data should be removed because it is unusal

print(df2[df2.total_sqft/df2.bhk < 300])
print(df2.shape)
df3 = df2[~(df2.total_sqft/df2.bhk < 300)]
print(df3.shape)

# price_per_sqft

print(df3.price_per_sqft.describe())  # min pps is 247 and max is 177470 the max is possible for but generic model we can remove extremes
# We should remove outliers per location using mean and one standard deviation

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_out = pd.concat([df_out,reduced_df], ignore_index=True)
    return df_out
df4 = remove_pps_outliers(df3)
print(df4.shape)

# In some cases the 2 BHK value is Greater than 3 BHK values of same location or same sqft
def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    plt.show()


plot_scatter_chart(df4, "Rajaji Nagar")

# Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df5 = remove_bhk_outliers(df4)
# df8 = df7.copy()
print(df5.shape)
# Plot same scatter chart again to visualize price_per_sqft for 2 BHK and 3 BHK properties
plot_scatter_chart(df5, "Rajaji Nagar")
plt.show()

# Outlier Removal Using Bathrooms Feature

print(df5.bath.unique())

# bathrooom greater than bedroom should be removed because in real time it is unusal]

# It is unusual to have 2 more bathrooms than number of bedrooms in a home
print(df5[df5.bath > df5.bhk+2])

df6 = df5[df5.bath < df5.bhk+2]
df7 = df6.drop(['price_per_sqft'], axis='columns')
print(df7.head(3))

# One hot encoding of location
dummies = pd.get_dummies(df7.location)

# to drop others column in dataframe

df8 = pd.concat([df7, dummies.drop(['other'], axis='columns')], axis='columns')
# to remove location columns
df8.drop(['location'], axis='columns', inplace=True)
print(df8.head())

# Build a model

X = df8.drop(['price'], axis='columns')
y = df8.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
print(len(X_train))

model = LinearRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(score)

# Use K Fold cross validation to measure accuracy of our LinearRegression model

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

crossValScore = cross_val_score(LinearRegression(), X, y, cv=cv)
# print(crossValScore)

# Find best model using GridSearchCV

def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    dfc = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
    print(dfc.best_score)

# find_best_model_using_gridsearchcv(X,y)

# LinearRegression is better 70% working good

# Test the method with some parameters


def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    print(model.predict([x])[0])

predict_price('Lingadheeranahalli', 1000, 2, 2)

import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(model, f)

# Export location and column information to a file that will be useful later on in our prediction application
import json
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))
