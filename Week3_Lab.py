# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 14:33:29 2023

@author: adfw980
"""

import statsmodels.api as sm

import pandas as pd

import csv

import matplotlib.pyplot as plt

f = open('C:/Users/adfw980/Downloads/accord_sedan (1).csv')

df1 = pd.DataFrame(csv.reader(f))

df1.columns = ['Price', 'Mileage', 'Year', 'Trim', 'Engine', 'Transmission']

plt.hist(df1['Price'])
plt.hist(df1['Mileage'])
plt.hist(df1['Year'])
plt.hist(df1['Trim'])
plt.hist(df1['Engine'])
plt.hist(df1['Transmission'])

df1 = df1.iloc[1:]
df1['Price'] = df1['Price'].astype(float)

plt.boxplot(df1['Price'])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Car Price Boxplot')
plt.show()

df1['Mileage'] = df1['Mileage'].astype(float)
plt.boxplot(df1['Mileage'])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Mileage')
plt.show()

plt.scatter(df1['Price'], df1['Mileage'])

df2 = pd.DataFrame({'isOutlierPrice': [0] * 417, 'isOutlierMileage' : [0]*417})

df = pd.concat([df1, df2], axis=1)

df.describe()
df['Price'].describe()
df['Mileage'].describe()

#We will make both 'isOutlierPrice' and 'isOutlierMileage' as columns withg the stds of Price and Mileage columns, 
#Then make a loop for each and turn stds> 2 to one and the others 0

import math

for row in df['Price']:
 if (row - 12084)/2061 > 2:   
   df.loc[ row,'isOutlierPrice'] = 1
 else:
    df.loc[row, 'isOutlierPrice'] = 0
    
    
#for row in df['Milegae']

for row in df['Mileage']:
 if (row - 89725)/25957 > 2:   
   df.loc[ row,'isOutlierMileage'] = 1
 else:
    df.loc[row, 'isOutlierMileage'] = 0

#Histograms of Outliers

plt.hist(df['isOutlierMileage'])

plt.hist(df['isOutlierPrice'])

#Differently COloured Outliers

import seaborn as sns

boxplot = sns.boxplot (data=df, x='Price')

Q1 = df['Price'].quantile(0.25)

Q3 = df['Price'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5*IQR

upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Price'] < lower_bound) | (df['Price'] > upper_bound)]

sns.swarmplot(data=outliers, x='Price', color='red', ax=boxplot)



#Q2:

#P1    
    f1 = open('C:/Users/adfw980/Downloads/TB_burden_countries_2014-09-29 (5).csv')

    df3 = pd.DataFrame(csv.reader(f1))
    
    plt.hist(df3[10])
    
    import numpy as np
    
    plt.hist(np.random.normal(0, 1, 4904))
    
#P2

 import scipy.stats as stats
 
 sm.qqplot(df3[10], stats.distributions.norm)
 
 #Q3
 
 #P1
 
 s1 = np.random.normal(0, 1, 5)

 s2 = np.random.normal(0, 1, 10)

 s3 = np.random.normal(0, 1, 100)

 s4 = np.random.normal(0, 2, 5)

 s5 = np.random.normal(0, 2, 10)

 s6 = np.random.normal(0, 2, 100)

 s6 = np.random.normal(0, 4, 5)

 s6 = np.random.normal(0, 2, 100)

 s7 = np.random.normal(0, 4, 5)

 s8 = np.random.normal(0, 4, 10)

 s9 = np.random.normal(0, 4, 100)

 s10 = np.random.normal(0, 6, 5)

 s11 = np.random.normal(0, 6, 10)

 s12 = np.random.normal(0, 6, 100)
 
 #For the first sample
 
 plt.hist(s1)
 
 mean_s1 = np.mean(s1)

 std_s1 = np.std(s1)
 
 from scipy.stats import skew, kurtosis

 skewness_s1 = skew(s1)
 
 kurtosis_s1 = kurtosis(s1)
 
#For the 2nd sample

plt.hist(s2)

mean_s2 = np.mean(s2)

std_s2 = np.std(s2)


skewness_s2 = skew(s2)

kurtosis_s2 = kurtosis(s2)

#For the 3rd sample

plt.hist(s3)

mean_s3 = np.mean(s3)

std_s3 = np.std(s3)


skewness_s3 = skew(s3)

kurtosis_s3 = kurtosis(s3)

#For 4th sample

plt.hist(s4)

mean_s4 = np.mean(s4)

std_s4 = np.std(s4)

skewness_s4 = skew(s4)

kurtosis_s4 = kurtosis(s4)

#For 5th sample

plt.hist(s5)

mean_s5 = np.mean(s5)

std_s5 = np.std(s5)

skewness_s5 = skew(s5)

kurtosis_s5 = kurtosis(s5)

#for 6th example

plt.hist(s6)

mean_s6 = np.mean(s6)

std_s6 = np.std(s6)

skewness_s6 = skew(s6)

kurtosis_s6 = kurtosis(s6)

#For 7th example:
    
  plt.hist(s7)

    mean_s7 = np.mean(s7)

    std_s7 = np.std(s7)

    skewness_s7 = skew(s7)

    kurtosis_s7 = kurtosis(s7)
    
    #For 8th example:
        
  plt.hist(s8)

      mean_s8 = np.mean(s8)

      std_s8 = np.std(s8)

      skewness_s8 = skew(s8)

      kurtosis_s8 = kurtosis(s8)   
    
    #For 9th example
    
  plt.hist(s9)

    mean_s9 = np.mean(s9)

    std_s9 = np.std(s9)

    skewness_s9 = skew(s9)

    kurtosis_s9 = kurtosis(s9)
    
    #For 10th example
    
  plt.hist(s10)

    mean_s10 = np.mean(s10)

    std_s10 = np.std(s10)

    skewness_s10 = skew(s10)

    kurtosis_s10 = kurtosis(s10)
    
    #For 11th example
    
  plt.hist(s11)

    mean_s11 = np.mean(s11)

    std_s11 = np.std(s11)

    skewness_s11 = skew(s11)

    kurtosis_s11 = kurtosis(s11)

     #For 12th example
     
   plt.hist(s12)

     mean_s11 = np.mean(s12)

     std_s11 = np.std(s12)

     skewness_s11 = skew(s12)

     kurtosis_s11 = kurtosis(s12)
     
     #Optional
     
     p1 = np.random.poisson(5, 10)

     p2 = np.random.poisson(5, 20)

     p3 = np.random.poisson(5, 50)
     
     p4 = np.random.poisson(5, 5000)
     
     plt.hist(p1)
     plt.hist(p2)
     plt.hist(p3)
     plt.hist(p4)

#Last Part

df3.fillna(0)

plt.hist(df3[9])

df3 = df3.iloc[1:]

df3[9].astype(float)

mean_r1 = np.mean(df3[9])

median_r1 = df3[9].median()

median_r1 = np.median(df3[9])

plt.hist(df3[8])

plt.hist(df3[7])

plt.hist(df3[10])

plt.hist(df3[11])

plt.hist(df3[12])

plt.hist(df3[13])

plt.hist(df3[14])

plt.hist(df3[15])

plt.hist(df3[16])






