# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:19:27 2020

@author: mdj72
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import datasets, linear_model
from scipy import stats
import statsmodels.stats.weightstats as ws
from statsmodels.stats.power import tt_ind_solve_power
# import scipy.stats as ss

boston=load_boston()
pd.set_option('display.max_columns', 20)

df = pd.DataFrame(np.c_[boston['data'], boston['target']],columns= np.append(boston['feature_names'], ['MEDV']))
print ('Part 1\n------------------------------------')
dfnoxmean=df.NOX.mean(axis=0)
print ("The mean of NOX is:" ,dfnoxmean)
dfnoxstd=df.NOX.std(axis=0)
print ("The standard deviation of NOX is:" ,dfnoxstd)

df.NOX.plot.hist(bins=15)
plt.title('Histogram of NOX')
plt.xlabel('nitric oxides concentration (parts per 10 million)')
plt.show()
# print (df[['NOX']].corrwith(df[['MEDV']]))
dfnox=pd.DataFrame(df.NOX)
dfmedv=df.MEDV
print ("Correlation of NOX to MEDV:",dfnox.corrwith(dfmedv))

print ("Regression 1st way results for NOX on MEDV:")
predictors = sm.add_constant(df.MEDV, prepend = False)
lm_mod = sm.OLS(df.NOX, predictors)
res = lm_mod.fit()
print(res.summary())
dfmedv=pd.DataFrame(df.MEDV)

print ("\nRegression 2nd ways results for NOX on MEDV:")
regr = linear_model.LinearRegression()
regr.fit(dfnox, dfmedv)
plt.scatter(dfnox, dfmedv,  color='black')
plt.plot(dfnox, regr.predict(dfnox), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.xlabel('nitric oxides concentration (parts per 10 million)')
plt.ylabel('Median value of owner-occupied homes in $1000s')
plt.show()

print ('\n\nPart 2\n---------------------------')
print ("The null hypothesis examines the data to find how often we could get the same data randomly")
print ("Normally, we reject the null hypothesis if by random we could only get that result less than 5% of the time.")
dfchas1=df.MEDV[df.CHAS==1]
dfchas0=df.MEDV[df.CHAS==0]
ttest,pval = stats.ttest_ind(dfchas1,dfchas0)
print ('P-val: ',pval, 'ttset value', ttest)

means = ws.CompareMeans(ws.DescrStatsW(dfchas1), ws.DescrStatsW(dfchas0))
confint = means.tconfint_diff(alpha=0.05, alternative='two-sided', usevar='unequal')
print ('Confidence interval:', confint[0], confint[1])
ratio = len(dfchas0)/len(dfchas1)
gsize=tt_ind_solve_power(effect_size=0.6, nobs1 = None, alpha=0.05, power=0.8, ratio=ratio, alternative='two-sided')
print ('Assume an effect size (Cohen’s d) of 0.6. If you want 80% power, what group size is necessary?', gsize)

print ('\n\nPart 3\n---------------------------')
print (' Design an experiment to explore the effects of these features on the media house price in census tracts. You should include an explanation of the experimental design as well as a plan of analysis, which should include a discussion of group size and power.')
print ('My experiment is to see the effect of house prices in a tracts/communities where there are no utility poles.')
print ('A google search on this topic found studies on this issue here: https://www.nar.realtor/sites/default/files/reports/2013/Price-Effects-of-High-Voltage-Lines-March-2013.pdf')
print ('Based on these studies it would seem that you would have to find 2 commmunities that are relatively similiar in most respects.')
print ('I would survey buyers on 5 issues on likert scale how much would these 5 issues effect the price and likelyhood of buying this property ')
print ('other questions could be gas or oil heat? Age of house? What type of AC system?')
print ('this would make it hard for the participants to know exactly what I''m interested in')
print ('from the previous studies it seems the effect of utility poles near houses in Vancouver is about 6.3% with a study of 12k participants.')
print ('that seems like a powerful study for individual houses but what about a whole communities?')
print ('some other questions that would need to be answered how many houses on avg can be serviced by one pole?')
print ('if for argument sake you say 6 that means that 1 of 6 houses in each tract loses 6% value')
print ('That would be a 1% percent recduction of house values for a tract/community with utility poles.')
print ('To see such a small change would need a large study with many participants and power to see the small effect.')

print ('I would need a couple more weeks to feel comfortable using the statistics and python.')


