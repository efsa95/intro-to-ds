---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Assignment 9: Regression and Optimization

+++

## 9.3.1. Linear Regression Basics

```{code-cell} ipython3
# basic libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# models classes
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import cluster
from sklearn import svm
# datasets
from sklearn import datasets, linear_model
# model selection tools
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
```

## Description:

+++

This [Dataset](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who?resource=download) contains statistics from 2000 to 2015 across 193 countries, and contains health and economic data from the WHO and UN. There are 2938 rows with 22 columns with features that include the country and year collected along side metrics that show adult mortality rates, infant deaths, education, various reported disease numbers/metrics, population, and more. In this prediction task Ill be attempting is to determine Life expectancy based off of Adult mortality rate and other features. Ill remove some non contiguous features such as country, year, and status since they are not contiguous. Linear regression is a good choice for targeting Life Expectancy since these features, specifically things like Adult Mortality and Schooling, should intuitively increase or decrease alongside it.

```{code-cell} ipython3
lifeExp_df = pd.read_csv('LifeExpectancyData.csv')
lifeExp_df.head()
```

## Cleaning:

```{code-cell} ipython3
lifeExp_df.columns
```

```{code-cell} ipython3
lifeExp_df = lifeExp_df.rename(columns={'Life expectancy ': 'Life expectancy'})
```

```{code-cell} ipython3
lifeExp_df.shape
```

```{code-cell} ipython3
lifeExp_df['Life expectancy'].describe()
```

```{code-cell} ipython3
lifeExp_clean = lifeExp_df.drop(columns=['Year','Country', 'Status'])
lifeExp_clean.head(1)
```

```{code-cell} ipython3
lifeExp_clean.shape
```

```{code-cell} ipython3
lifeExp_clean = lifeExp_clean.dropna()
lifeExp_clean.shape
```

## Instantiate/fit:

```{code-cell} ipython3
lifeExp_allFeat_X = lifeExp_clean.drop(columns=['Life expectancy'])
```

```{code-cell} ipython3
lifeExp_target_y = lifeExp_clean['Life expectancy'] 
```

```{code-cell} ipython3
X_train, X_test, y_train, y_test = train_test_split(lifeExp_allFeat_X, lifeExp_target_y, test_size=0.25)
```

```{code-cell} ipython3
lifeExp_reg = linear_model.LinearRegression()
lifeExp_reg.fit(X_train, y_train)
```

```{code-cell} ipython3
y_pred = lifeExp_reg.predict(X_test)
```

```{code-cell} ipython3
plt.scatter(y_test, y_pred, color='blue')  #Pred
plt.scatter(y_test, y_test, color='black')  #Actual 
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
```

```{code-cell} ipython3
r2_score(y_test,y_pred)
```

```{code-cell} ipython3
mean_abs_error = np.sqrt(mean_squared_error(y_test,y_pred))
mean_abs_error
```

## Test it on 25% held out test data and measure the fit with two metrics and one plot

+++

The r2_score shows an 82% accuracy score which is close to 1, showing a very good variance and good fit to the model. The mean_abs_error shows the models predictions tend to be 3.7 years off from the true correct values, this again shows the model is fit well.  The line plot above also is a good visual showing the model is finding the general trend of life expectancy correctly.

```{code-cell} ipython3
# draw vertical lines from each data point to its predict value
[plt.plot([yt, yt], [yt, yp], color='red', linewidth=1, markevery=[0], marker ='.')
                 for yp, yt in zip(y_pred,y_test)];

# plot these last so they are visually on top
plt.scatter(y_test, y_pred, color='blue')
plt.scatter(y_test, y_test, color='black')

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
```

```{code-cell} ipython3
lifeExp_reg.coef_
```

## Coefficients and Residuals

+++

The positive coefficients are showing features that increase the life expectancy while the negatives are doing the opposite. We can also see which features are contributing the most to determining the models predictions by how close they are to zero. I f we match up the index's to the cols in the clean_df the first number should be Adult Mortality and the second is infant deaths, since infant deaths is a tangible number and not a rate it makes sense it would have a higher effect on predicting Life expectancy since infants would generally be around the age of 1-3.

+++

## Repeat the split, train, and test steps 5 times

```{code-cell} ipython3
r2_cva = cross_val_score(lifeExp_reg, lifeExp_allFeat_X, lifeExp_target_y)
r2_cva
```

```{code-cell} ipython3
np.mean(r2_cva)
```

Using Kfold method we see that there can be a possibility of a lower score, specifiacally we can see a lower score of 70.7%. We have an average around 78%, this is still a high enough score to make me trust the model.

+++

## Interpret the model and its performance in terms of the application

+++

I think in general this model shows there is a trend fairly well, I personally don't think it would be good enough for real world use due to some of its lower splits being right at 70 and even at its highest being 83%.  Sure we can see a trend of life expectancy going up but the information from the graphs and scores don't really tell us why they go up just only that they can go up.  It would be more useful to try linear regression on individual statistics to find specific trends rather than a general one. We can however learn from these statistics that there is a way to increase mortality rate we just now need to dig deeper with a more complex model, as I do believe machine learning can help us with these types of statistics.

+++

## Just One Feature:

+++

Logically adult mortality seemed linked to life expectancy, where a higher mortality rate would lead to a lower Life expectancy so I wanted to see if the model/data reflected this idea. Looking at lifeExp_reg.coef_ Adult Mortality got a score close to zero around -1 meaning it has lesser effect on the model so I wanted to see why. After Using this one feature It seems like the line like figure shown in the plot below might have to do with its low score prevoiusly. A coefficient of "-0.04908872" shows us as Adult Mortality increases life expectancy decreases. This makes sense, if people are dying sooner than mortality rates should rise. The residuals in the start of the plot show a huge discrepancy, with such a low mortality rate we also have some extremely low life expectancy ages. This could mean some other demographic data was causing a lower age expectancy in specific countries or other historical factors.

```{code-cell} ipython3
lifeExp_X1 = lifeExp_clean['Adult Mortality'].values[:,np.newaxis]
```

```{code-cell} ipython3
lifeExp_X1.shape
```

```{code-cell} ipython3
lifeExp_y1 = lifeExp_clean['Life expectancy'] 
```

```{code-cell} ipython3
lifeExp_y1.shape
```

```{code-cell} ipython3
X1_train, X1_test, y1_train, y1_test = train_test_split(lifeExp_X1, lifeExp_y1, test_size=0.25)
```

```{code-cell} ipython3
lifeExp_reg1 = linear_model.LinearRegression()
lifeExp_reg1.fit(X1_train, y1_train)
```

```{code-cell} ipython3
lifeExp_reg1.coef_
```

```{code-cell} ipython3
y1_pred = lifeExp_reg1.predict(X1_test)
```

```{code-cell} ipython3
plt.scatter(X1_test, y1_test, color = 'black') #Actual Data
plt.scatter(X1_test, y1_pred, color='blue') #Prediction

plt.xlabel("Adult Mortality")
plt.ylabel("Life expectancy")
```

```{code-cell} ipython3
r2_score(y1_test,y1_pred)
```

```{code-cell} ipython3
mean_abs_error1 = np.sqrt(mean_squared_error(y1_test,y1_pred))
mean_abs_error1
```

## Test it on 25% held out test data and measure the fit with two metrics and one plot

+++

The r2 score is low which probably has to do with the large line like figure in the actual data at the start of the plot. The mean_abs_error shows that on average the model’s prediction is about six and a half years off for life expectancy.

```{code-cell} ipython3
# plot line prediction
plt.plot(X1_test, y1_pred, color='blue', linewidth=3);

# draw vertical lines from each data point to its predict value
[plt.plot([x,x],[yp,yt], color='red', linewidth=3, markevery=[0], marker ='^')
                 for x, yp, yt in zip(X1_test, y1_pred,y1_test)];

# plot these last so they are visually on top
plt.scatter(X1_test,y1_test,  color='black');

plt.xlabel("Adult Mortality")
plt.ylabel("Life expectancy")
```

# Optimize a more complex regression model

+++

## Instantiate, Fit, and Score

```{code-cell} ipython3
dt = tree.DecisionTreeRegressor()
dt.fit(X_train, y_train)
```

```{code-cell} ipython3
dt.score(X_train, y_train)
```

```{code-cell} ipython3
params_dt = {'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],'max_depth': [2,5,10,20],'min_samples_split': [2,5,10,20], 'min_samples_leaf': [2,5,10,20]}
```

```{code-cell} ipython3
dt_opt = GridSearchCV(dt,params_dt)
```

```{code-cell} ipython3
dt_opt.fit(X_train, y_train)
```

```{code-cell} ipython3
dt_opt.score(X_train, y_train)
```

```{code-cell} ipython3
dt_opt.best_params_
```

## Examine/score the best fit model, how different is it from the default

+++

The grid search determined poisson max_depth=20, min_samples_leaf=5, min_samples_split=10 while the default would have been squared_error, max_depth=None, min_samples_leaf=1, min_samples_split=2.  Doing this has given a proper score where the default was giving a perfect 1 which points to over fitting. Ive changed max_depth to try higher numbers and seems to always pick the larger number. After 70 the it takes to long to compute and times out, the more depth available the more accurate the score will be. It also picks Poisson over squared_error, looking at the time it takes to fit/score poisson takes much less time. When I did a 10 fold on the data it showed almost the same exact score which means its a stable and not over fitting data.

```{code-cell} ipython3
dt_5cv_df = pd.DataFrame(dt_opt.cv_results_)
dt_5cv_df
```

## Trying more folds:

```{code-cell} ipython3
dt_opt_10F = GridSearchCV(dt,params_dt, cv=10)
dt_opt_10F.fit(X_train, y_train)
```

```{code-cell} ipython3
dt_opt_10F.score(X_train, y_train)
```
