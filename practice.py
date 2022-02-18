#import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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