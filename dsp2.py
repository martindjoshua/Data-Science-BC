# -*- coding: utf-8 -*-
"""
Spyder Editor

Data Querying and Cleaning: 2nd Data Science Project
"""
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


conn = sqlite3.connect('lahmansbaseballdb.sqlite')
df = pd.read_sql('''
Select nameFirst, nameLast, TotalRBI
FROM people join (                             
    Select sum(RBI) as TotalRBI, playerID
    From Batting 
    Where RBI > 0 and yearID >=2015 and G >50 
    Group by playerID
	)	as RBITOTAL on people.playerID = RBITotal.playerID 
where finalGame > '2018-01-01 00:00:00'
Order by TotalRBI desc'''
, conn)
print ("Which player had the most RBI's between 2016 and 2018?")
print (df.head(n=1))

df = pd.read_sql('''
SELECT People.nameFirst || ' ' || People.nameLast as Name,Batting.GIDP
FROM Batting join People on Batting.playerID=People.playerID
WHERE (((Batting.playerID)=(
SELECT People.playerID
FROM People
WHERE (((People.nameLast)="Pujols") AND ((People.nameFirst)="Albert"))
    ) AND ((Batting.yearID)=2016)))'''
,conn)

print  ("How many DP did Albert Pujols ground into in 2016?")
print (df)
    
bat = pd.read_sql(''' SELECT Batting.*
FROM Batting
WHERE (((Batting.yearID)>1960));
''', conn)


# triples histogram
triples=(bat.groupby(by=['yearID'])['3B'].sum())
plt.hist(triples,bins=5)
plt.xlabel('# of Triples')
plt.ylabel('How often?')
plt.title('Histogram of # of Triples')
plt.show()

# scatter plot of 3b and sbs
triples=(bat.groupby(by=['yearID'])[['3B','SB']].sum())
plt.scatter(x=triples['3B'],y=triples['SB'])
plt.xlabel('Triples')
plt.ylabel('Stolen Bases')
plt.title('Scatter plot of Triples to Stolen Bases')
plt.show()

team= pd.read_sql(''' select * from Teams 
                  where yearid>1960''', conn)

# print (team.shape)
teambat=(bat.groupby(by=['yearID','teamID'], as_index=False)[['AB','H','2B','3B','HR','BB','R']].sum())
teambat['obp']=(teambat['H']+teambat['BB'])/teambat['AB']
teambat['slgp']=(teambat['H']+teambat['2B']+teambat['3B']*2+teambat['HR']*3)/teambat['AB']
teambat['ops']=teambat.obp+teambat.slgp
teambat['stdmean']=0.001
stdops=(teambat.groupby(by=['yearID'], as_index=False)[['ops']].std())
batmean=(teambat.groupby(by=['yearID'], as_index=False)[['ops']].mean())
for i, row in teambat.iterrows():
    for x in range(batmean.shape[0]):
        if row.yearID==batmean.iloc[x,0]:
            teambat.iloc[i,12]=(row.ops-batmean.iloc[x,1])/stdops.iloc[x,1]

tophittingteams=teambat.sort_values(by=['stdmean'], ascending=False)
print ("The top five hitting team statistically speaking compared to the rest of MLB that year by OPS are:")
print (tophittingteams.head(5))
print ("AL teams have had the DH since the 70's so they have had an advantage for most of the time. If I had more time I'd use league as a 3rd index.")
print ("How do you use iloc and loc on a dataframe with indexes? I used 'as_index=false' as the solution.")

