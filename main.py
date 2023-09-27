## EDA on the SimplyRETs sample data
## along with an example of lasso regression process

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from datetime import datetime
plt.style.use('ggplot')

### This is how I would connect to a database with data from
### homeSpotter REPL
### Should replace the "*" with actual columns to make this more efficient

##########################
#engine = create_engine("mysql://root:homeSpotter!@localhost/homespotter")
## create_engine("mysql://[username]:[password]@[host address]/[databaseName]")
#connection = engine.connect()
#statement = "SELECT * FROM properties LEFT OUTER JOIN openhouses ON openhouses.mlsId = properties.mlsId \
#LEFT OUTER JOIN agents ON agents.agentId = properties.agentId \
#LEFT OUTER JOIN photos ON photos.mlsId = properties.mlsId;"

#queryResult = connection.execute(statement)
#colHeader = queryResult.keys()

#df = pd.DataFrame(list(queryResult), columns=colHeader)
##########################

## Instead:
df = pd.read_csv('homedata.csv')

# All agentContact, bathsThreeQuarter, and virtualTourUrl values are null, let's drop those columns
# Also dropping some columns which have junk data from this particular
# dataset (descriptions, etc.)
# ZIP will imply city and state (and is more narrow)
# so don't need to keep redundant information
# Similarly, agentId is redundant with agentFirstName and agentLastName
df.drop(['virtualTourUrl','agentContact',\
         'bathsThreeQuarter','prvRemarks','inputId','openHouseKey',\
         'openHouseId','openHouseDescription','showingInstructions','city','state',\
         'mlsStatus','agentId.1','agentFirstName','agentLastName','listDate',\
         'streetNum','mlsId.1','mlsId.2','mlsId',\
         'unit', 'endTime'], axis=1, inplace=True)
         
# parkingGarageSpaces should be treated as a decimal instead of a string
df.parkingGarageSpaces = df.parkingGarageSpaces.apply(lambda x: float(x))

# ZIP code (keyed as postalCode) should be considered as a string, since the
# scale of the numerical value is not relevant to the valuation of the
# property
df.postalCode = df.postalCode.apply(lambda x: str(x))

# Let's reinterpret the startTime and endTime with just the day of the week
# and consider this a string - could also add hour of the day
df.startTime = df.startTime.fillna("1900-1-1 01:01:01")
# Because this is coming from CSV, need to force datetime formatting
df.startTime = df.startTime.apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
df.startTime = df.startTime.apply(lambda x: str(x.weekday()))

# We also don't need the photo URLs - but the number of photos on the listing could be interesting - URLs are split by commas in the string
# In this dataset, having photos causes an error since they all have
# the same number of photos (the correlation calulation divides by 0)
df.photos = df.photos.apply(lambda x: len(x.split(',')))

# Let's find all of the numerical and Categorical Features
# so we can work with them separately. We will also
# fill any missing numerical data with zero and categorical data with 'Missing'.

# Numerical Features
numerical_feats = df.select_dtypes(include=['int64','float64']).columns
numerical_feats = numerical_feats.drop(['listPrice'])
df[numerical_feats] = df[numerical_feats].fillna(0)

# Categorical Features
categorical_feats = df.select_dtypes(include=['object']).columns
df[categorical_feats] = df[categorical_feats].fillna(value='Missing')

# Now look at the correlation between all numerical features
# If we are going to create a regression it may be important
# to remove attributes which are very correlated with one another
# and this will help us find those
corr = df[['listPrice'] + numerical_feats.tolist()].corr()
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
colorMap = sns.diverging_palette(150, 275, s=80, l=55, n=9, as_cmap=True)
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.index.values, cmap=colorMap)
ax.xaxis.tick_top()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.savefig('correlationNumericalHeat.png')

# Sort features by correlation to saleprice from positive to negative
# Could also look at the absolute value
corr = corr.sort_values('listPrice', ascending=False)
plt.figure(figsize=(20,20))
sns.barplot(corr.listPrice[1:], corr.index[1:], orient='h')
plt.savefig('correlationNumericalBar.png')


# Look for effects of categorical variables on sales price by
# calulating the one-way analysis of variance (ANOVA)
# between groups of listPrice gathered for each unique value of each feature
# streetName returns no correlation as all the values are different
anova = {'feat':[], 'f':[], 'p':[]}
for category in categorical_feats:
    group_prices = []
    for group in df[category].unique():
      group_prices.append(df[df[category] == group]['listPrice'].values)
    f, p = scipy.stats.f_oneway(*group_prices)
    anova['feat'].append(category)
    anova['f'].append(f)
    anova['p'].append(p)
anova = pd.DataFrame(anova)
anova.sort_values('p', inplace=True)


plt.figure(figsize=(20,10))
sns.barplot(anova.feat, np.log(1./anova['p']))
plt.xticks(rotation=60)
plt.savefig('correlationCategorical.png')

## Lasso Regression - this is mostly just to demonstrate the process
## as this set has only 20 datapoints the output is somewhat
## meaningless - note the negative fit "score" printed at the end

# Convert categorical features to binary features
# Regressor can't understand strings - this splits each categorical column
# into "k-1" columns where "k" is the number of unique values
# for the attribute. The "first" value is dropped to avoid a redundant
# feature.
df_origin = pd.get_dummies(df, drop_first = True)

# Split out X (features) and y (target = listPrice) data
y = df_origin.listPrice
X = df_origin.drop('listPrice',axis=1)

# Split into train and test sets
# Regressor is trained and tested with different data to ensure
# fit is not overly dependent on the data used to create the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Setup the hyperparameter grid - this will test a range of
# the alpha variable used by the lasso regressor to find the "best" value
# for this data
c_space = np.arange(0.5,1.0,0.1)
param_grid = {'alpha':c_space}

# Instantiate Lasso regressor
lasso = Lasso(tol=0.0001, normalize=True, max_iter=1e5)

# Instantiate the GridSearchCV object: logreg_cv
# The CV portion will also split out the training set to improve
# ability of the model to be used to with unseen data
lasso_cv = GridSearchCV(lasso, param_grid, cv=2)

# Fit regressor to the data
lasso_cv.fit(X,y)

# Print the tuned parameters and score
print '\nLasso Regressor finished tuning parameters'
print 'Tuned Logistic Regression Parameters: {}'.format(lasso_cv.best_params_) 
print 'Best score is {}'.format(lasso_cv.best_score_)
#print(zip(list(X.columns.values),lasso_cv.best_estimator_.coef_))