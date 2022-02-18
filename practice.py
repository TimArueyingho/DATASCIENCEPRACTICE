#import libraries
import  numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import seaborn as sea

#load the data
dataset = pd.read_csv("C:/Users/sj21399/Downloads/Life_Expectancy_Data (1).csv")
print(dataset)
print (dataset.columns)


#now we have seen all the columns, let us show Hepatitis B, HIV/AIDS, Measles and Polio for
#The countries Nigeria, Ghana, Togo, Ghana, Egypt, Zimbabwe, and Lebanon

#lineplot

#this is our x
Afghanistan = dataset[dataset.Country == 'Afghanistan'].set_index('Year').sort_index()
Belgium = dataset[dataset.Country == 'Belgium'].set_index('Year').sort_index()
Australia = dataset[dataset.Country == 'Australia'].set_index('Year').sort_index()
Colombia = dataset[dataset.Country == 'Colombia'].set_index('Year').sort_index()

fig, ax = plt.subplots(figsize = (15,5))
Afghanistan.plot(kind="line", y = ['Adult Mortality'], label = ["Afghanistan"],  ax = ax )
Belgium.plot(kind="line", y = ['Adult Mortality'], label = ["Belgium"],  ax = ax )
Australia.plot(kind="line", y = ['Adult Mortality'], label = ["Australia"],  ax = ax )
Colombia.plot(kind="line", y = ['Adult Mortality'], label = ["Colombia"],  ax = ax )
ax.set(title = 'Comparing Polio for different countries', xlabel="Year", ylabel="Life expectancy") #Change axis labels and title
plt.show()





#before we start anything...we would look at the data independently
#we import the data .................1
dataset = pd.read_csv("C:/Users/sj21399/Downloads/survey_results_public.csv")
dataset_schema = ("C:/Users/sj21399/Downloads/survey_results_schema.csv")
print (dataset)
#pd.read_csv to see columns of schema
#print(dataset_schema)

#It appears that shortcodes for questions have been used as column names.

#we arrange the data into the columns we want to use..............2
#we want the schema to contain the answers and questions
#we would open dataset_schema here, everything under the second column (QuestionText) would enter under Column
schema_raw = pd.read_csv(dataset_schema , index_col='Column').QuestionText
print(schema_raw)

#we can now look for individual questions
#method for relating the parts of two different columns
#use one column which is indexed, to bring out what is in the other column
print (schema_raw['Age'])
print (schema_raw['Age1stCode'])
print (schema_raw['WelcomeChange'])

#we are looking at the main data, we have chosen the coluns we want to work with..........3
#Because we want to analyze:
#Demographics of the survey respondents and the global programming community
#Distribution of programming skills, experience, and preferences
#Employment-related information, preferences, and opinions

#refer to dataset.columns

selected_info = ['Country','Age','Gender','EdLevel','UndergradMajor','Hobbyist','Age1stCode','YearsCode','YearsCodePro','LanguageWorkedWith',
                 'LanguageDesireNextYear','NEWLearn','NEWStuck','Employment','DevType','WorkWeekHrs', 'JobSat','JobFactors', 'NEWOvertime','NEWEdImpt']

#These are the columns we want to work with...........4
#let us save all of these data in a different dataset
#we are selecting this info from that data set and storing it elsewhere

new_dataset = dataset[selected_info].copy()
print(new_dataset)

#let us assess what this new dataset is like........5
#?can we sample our data set here and clean it the way we like?

print (new_dataset.info())
#Total number of rows = 64460, but the values of the entries for each column vary..hence, some are empty
#there are 20 columns, with two float64...(Only two of the columns were detected as numeric columns (Age and WorkWeekHrs))
# the rest are objects, although some have integers
#we will convert everything (integers and floats) to numeric values to create a balance.............6
#we will leave the strings as strings

new_dataset['Age1stCode'] = pd.to_numeric(new_dataset.Age1stCode, errors='coerce')
new_dataset['YearsCode'] = pd.to_numeric(new_dataset.YearsCode, errors='coerce')
new_dataset['YearsCodePro'] = pd.to_numeric(new_dataset.YearsCodePro, errors='coerce')
new_dataset['Age'] = pd.to_numeric(new_dataset.Age, errors='coerce')
new_dataset['WorkWeekHrs'] = pd.to_numeric(new_dataset.WorkWeekHrs, errors='coerce')

#let us see what they all look like now, the basic stats are seen with describe...........7
print (new_dataset.describe())

#pay attention to the stats and ranges etc..........................8
#The minimum range and maximum range of Age look off
#we are going to sort out ages between 18 and 65

new_dataset.drop(new_dataset[new_dataset.Age < 18].index, inplace=True)
new_dataset.drop(new_dataset[new_dataset.Age > 65].index, inplace=True)

#let us also restrict work hours to 140
new_dataset.drop(new_dataset[new_dataset.WorkWeekHrs > 140].index, inplace=True)

#see the changes made
print (new_dataset.describe())

#let us work with sample 10 rows of this dataset...............9
work = new_dataset.sample(10)
print (work)

#Time to visualize what we have
#REMEMBER
#Because we want to analyze:
#Demographics of the survey respondents and the global programming community
#Distribution of programming skills, experience, and preferences
#Employment-related information, preferences, and opinions

#import important libraries...go to the top........................10
#we have our sample data with 20 columns

#let us set basic parameters for our libraries..............11
sea.set_style('darkgrid')
mat.rcParams['font.size'] = 14
mat.rcParams['figure.figsize'] = (9, 5)
mat.rcParams['figure.facecolor'] = '#00000000'


#let us plot 10 countries with the highest number of responses
# this would be from schema since it had responses...................12

print(schema_raw.Country)
#let us get the number of unique responses from our new dataset
print (new_dataset.Country.nunique())
#There are 183 unique responses

top_countries = new_dataset.Country.value_counts().head(10)
print (top_countries)
#Top 10 countries show
#Time to visualize....................13

plt.figure(figsize=(12,6))
plt.xticks(rotation=75)
plt.title(schema_raw.Country)
#show the countries and their number distribution based on the question from schema
sea.barplot(x=top_countries.index, y=top_countries);
#plt.show()

#look at the age of all respondents.............14

plt.figure(figsize=(12, 6))
plt.title(schema_raw.Age)
plt.xlabel('Age')
plt.ylabel('Number of respondents')
plt.show()

#let us look at employment and education separately