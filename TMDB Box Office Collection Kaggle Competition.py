!pip install kaggle

get_ipython().system('pip install kaggle')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import json


TEAM_NAME = "The Abnormal Distribution"

KAGGLE_USER_DATA = {"username":"pulkitbhasin","key":"33e92f989803ec683e8d031d2cff7fae"} # looks like this {"username":"ajaynraj","key":"<REDACTED>"}


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
X_train, y_train = train.drop('revenue', axis=1), train['revenue']
X_test = test
df = pd.concat((X_train, X_test), axis=0)

#Apllying log transformation to normalize data
y_train=y_train.apply(np.log)

#Feature Engineering and Data Cleaning

def feature_engineering(df):
    df['belongs_to_collection'] = df['belongs_to_collection'].fillna(0)
    df['belongs_to_collection']=df['belongs_to_collection'].astype(bool).astype(int) 
    df['homepage'] = df['homepage'].fillna(0)
    df['homepage']=df['homepage'].astype(bool).astype(int) 
    df=df.drop(['imdb_id'], axis=1)
    df=df.drop(['poster_path'], axis=1)
    df["genres"]=df["genres"].fillna("[]")
    df["genres"]=df['genres'].apply(eval)
    df["genres"]=df['genres'].apply(len)
    df["production_companies"]=df["production_companies"].fillna("[]")
    df["production_companies"]=df['production_companies'].apply(eval)
    df["production_companies"]=df['production_companies'].apply(len)
    df["spoken_languages"]=df["spoken_languages"].fillna("[]")
    df["spoken_languages"]=df['spoken_languages'].apply(eval)
    df["spoken_languages"]=df['spoken_languages'].apply(len)
    df["production_countries"]=df["production_countries"].fillna("[]")
    df["production_countries"]=df['production_countries'].apply(eval)
    df["production_countries"]=df['production_countries'].apply(len)
    df["Keywords"]=df["Keywords"].fillna("[]")
    df["Keywords"]=df['Keywords'].apply(eval)
    df["Keywords"]=df['Keywords'].apply(len)
    df['status'] = df['status'].fillna(0)
    df['status']=df['status'].astype(bool).astype(int) 
    df['tagline']=df['tagline'].fillna("")
    df["tagline"]=df["tagline"].apply(len)
    df['title']=df['title'].fillna("")
    df['number of words in title']=df['title'].apply(lambda x: 1+x.count(" "))
    df['length of title in characters']=df['title'].apply(len)
    df['overview']=df['overview'].fillna("")
    df['overview']=df['overview'].apply(len)
    df=df.drop(['original_title'], axis=1)
    df=df.drop(['title'], axis=1)
    df["cast"]=df["cast"].fillna("[]")
    df["cast"]=df['cast'].apply(eval)
    df["cast"]=df['cast'].apply(len)
    df["crew"]=df["crew"].fillna("[]")
    df["crew"]=df['crew'].apply(eval)
    df["crew"]=df['crew'].apply(len)
    df=df.drop(['release_date'], axis=1)
    df=pd.get_dummies(df, columns=['original_language'])
    df['runtime'] = df['runtime'].fillna(np.mean(df['runtime']))
    df = df.drop('id', axis=1)
    return df

X = feature_engineering(df)

# Exploratory Data Analysis

sns.distplot(df['belongs_to_collection'], label=y_train)
sns.distplot(df['title'].apply(lambda x: x.count(" ")), label=y_train)
sns.distplot(df['homepage'], label=y_train)
sns.distplot(df['tagline'].apply(len), label=y_train)

# Splitting up our cleaned df back into training and test
X_train = df[:train.shape[0]]
y_train = y_train
X_test = df[train.shape[0]:]
X_train

# Modeling

# Validation and Evaluation


from sklearn.metrics import mean_squared_log_error

def evaluate(y_pred, y_true):
    """Returns the RMSLE(y_pred, y_true)"""
    return (mean_squared_log_error(y_true, y_pred))**0.5

def my_eval(y_pred, y_true):
    return evaluate(np.clip(0, np.e**y_pred, y_pred.max()), y_true)


# Validation

from sklearn.model_selection import train_test_split

train_X, valid_X = train_test_split(X_train, test_size = .2, random_state = 0) 
train_y, valid_y = train_test_split(y_train, test_size = .2, random_state = 0) 


# Linear Regression

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression(fit_intercept=True)
linear_model = linear_model.fit(train_X, train_y)


# Regularized Regression

from sklearn.linear_model import Lasso
linear_model=Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
linear_model = linear_model.fit(train_X, train_y)

#Reverse log transformation and print RMSLE score of model
print(my_eval(linear_model.predict(valid_X), np.e**valid_y))


# Hyperparameter Tuning

from sklearn.model_selection import KFold

kf = KFold(n_splits=5)

alphas = [1e-3, 5e-3, 1e-2, 5e-2, 0.1]

cv_scores = np.zeros(len(alphas))

for alphai, alpha in enumerate(alphas):
    print('Training alpha =', alpha, end='\t')
    scores = np.zeros(5)
    for i, (train_index, test_index) in enumerate(kf.split(train_X)):
        train_x, test_x = train_X.iloc[train_index], train_X.iloc[test_index]
        train_Y, test_Y = train_y.iloc[train_index], train_y.iloc[test_index]
        linear_model = Lasso(alpha=alpha)
        linear_model.fit(train_x, train_Y)
        preds = linear_model.predict(test_x)
        scores[i] = my_eval(preds, np.e**test_Y)
    cv_scores[alphai] = scores.mean()
    print('RMSLE = ', cv_scores[alphai])


best_alpha = alphas[np.argmax(cv_scores)]
model = Lasso(alpha=best_alpha)
model.fit(train_X, train_y)
training_accuracy = my_eval(model.predict(train_X), np.e**train_y)
validation_accuracy = my_eval(model.predict(valid_X), np.e**valid_y)

print('Training accuracy', training_accuracy)
print('Validation accuracy', validation_accuracy)

# Fit a random forest model to the data and report the RMSLE.

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=300, max_depth=20)
model.fit(train_X, train_y)
print(my_eval(model.predict(valid_X), np.e**valid_y))

# Use the model to predict the box office prediction on the test set `X_test` and submit score

preds = np.e**model.predict(X_test)
print(preds)
PATH_TO_SUBMISSION = 'submission.csv'
preds = np.e**model.predict(X_test)
out = pd.DataFrame(data={'id': test['id'], 'revenue': preds}).set_index('id')
assert out.shape[0] == test.shape[0]
out.to_csv(PATH_TO_SUBMISSION)

# Submission

from submit import submit_to_leaderboard, view_submissions
success = submit_to_leaderboard(
    KAGGLE_USER_DATA, 
    TEAM_NAME, 
    path_to_submission=PATH_TO_SUBMISSION, 
    submit_to_kaggle=True
)

#Final Score: 2.22644
#Final rank: #3 out of 46 participants