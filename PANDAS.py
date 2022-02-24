import pandas as pd
import numpy as np

#s = pd.Series(data, index=index)

s1 = pd.Series(np.arange(0,5), index = ['I', 'II', 'III', 'IV', 'V'])
#print(s1)

s2 = pd.Series(data = (0.1, 12, 'Bristol', 1000), index = ('a', 'b', 'c', 'd'))
#location of index
#print(s2.loc['a'])
#print(s2.loc[['d', 'c', 'a']])

#location of integer index: integer location index
#print(s2.iloc[[2, 3, 0]])


s3 = pd.Series(data = {'q': 8, 'r': 16, 's': 24})
#print(s3)
#no need for index

d = {'X' : pd.Series(np.arange(0,5), index = ['cheese', 'wine', 'bread','olives', 'gin']),
     'Y' : pd.Series(data = ['Glasgow', 'London', 'Bristol'], index = ['wine','cheese', 'cider'])}
dF = pd.DataFrame(d)
#print(dF)
#the item contains two dictionaries which are pandas series

#print (dF.columns)
#print (dF.shape)
#print(dF.describe())
#print (dF.head())


#assess a column
#print(dF['X'])

#would print the indices of both items
#print(dF[0:3])

#create new column
list = ['a','b','c','','','']
dF['New Column']= list
#print(dF)

list = pd.Series(data= ['a','b','c','','p'], index= ['cheese', 'wine', 'beans','olives', 'gino'])
dF['New Column']= list
#print(dF)
#it omitted the indices that were already absent


 # Extract the rows of dF where the values in the column X are greater than 2.
dF_new = dF[dF['X'] > 2]
#print (dF_new)

#descriptive statistics

NBA = pd.read_csv("C:/Users/sj21399/Downloads/NBA_Stats (1).csv")
#print (NBA.head(5))
print (NBA.columns)

#Find the minimum, maximum, mean, and
# standard deviation of player_height of the players listed in the database

calc = NBA['player_height']
print(calc)

print(calc.min())
print(calc.max())
print(calc.mean())
print(calc.std())

#Using the functions .idxmax() and idxmin() who is (are) the tallest/shortest player(s) in the
#database? What is the difference in their heights?

id1 = calc.idxmax()
id2 = calc.idxmin()
print(f'The difference in their heights are: {id1 - id2}')

#Create a new DataFrame containing the columns player_height,
# offensive rebound percentage (oreb_pct), and defensive rebound percentage (dreb_pct)

new_DataFramel =  NBA[['player_height', 'oreb_pct', 'dreb_pct']]

#Using the function .corr(), analyze whether there exists a correlation between each of these three
#quantites for the players listed in the DataFrame

delo = pd.DataFrame(new_DataFramel)
print(delo.corr())

#Write a function that takes as input a year (draft_year) and season (season) and returns
#the player who was drafted in the year 2017 and scored the most points per game (pts) in
#the season 2018-2019?

def function(year, season):

