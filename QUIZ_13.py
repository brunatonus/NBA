import pandas as pd
import statsmodels.api as sm

#DATA
df = pd.read_csv(r'C:\Users\bruna\Downloads\Faculdade\Python\QUIZ 13\nbaShots99_00.csv')


#REGRESSION ONE
# DROPPED SHOT ZONE RANGE AND SHOT TYPE FOR MULTICOLLINEARITY 
X = pd.get_dummies(df[['SHOT_DISTANCE', 'SHOT_ZONE_AREA', 'ACTION_TYPE','PERIOD','MINUTES_REMAINING','LOC_X','LOC_Y']], drop_first=True).astype(float)
y = df['SHOT_MADE_FLAG'].astype(float)
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
# R SQUARED WAS LOW. I PLAYED WITH DIFFERENT X VARIABLES AND STILL HAD A LOW R SQUARED.


#SUMMARY STATS.
print("Summary Statistics:")
df[['SHOT_DISTANCE', 'SHOT_ZONE_AREA', 'ACTION_TYPE','PERIOD','MINUTES_REMAINING']].describe()