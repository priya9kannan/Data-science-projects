#!/usr/bin/env python
# coding: utf-8

# https://drive.google.com/file/d/1mG9Wl87Le0EH3cvmEwkGb1HYMsm44QgI/view?usp=sharing

# In[1]:


get_ipython().system('gdown 1mG9Wl87Le0EH3cvmEwkGb1HYMsm44QgI')


# In[10]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


yu=pd.read_csv("yulu.csv")
yu


# In[5]:


yu.shape


# In[6]:


yu.info()


# In[29]:


yu.describe()


# In[6]:


yu.describe(include=["object"])


# #    			UNIQUE COLUMN VALUES

# In[14]:


yu.season.unique()


# In[15]:


yu.holiday.unique()


# In[16]:


yu.workingday.unique()


# In[17]:


yu.weather.unique()


# In[19]:


yu.temp.unique()


# In[20]:


yu.atemp.unique()


# In[21]:


yu.humidity.unique()


# In[22]:


yu.windspeed.unique()


# In[23]:


yu.casual.unique()


# In[24]:


yu.registered.unique()


# #    				VALUE_COUNTS 

# In[30]:


yu["season"].value_counts()


# In[31]:


yu["holiday"].value_counts()


# In[32]:


yu["workingday"].value_counts()


# In[33]:


yu["weather"].value_counts()


# In[36]:


yu["temp"].value_counts().head(10)


# In[37]:


yu["atemp"].value_counts().head(10)


# In[38]:


yu["humidity"].value_counts().head(10)


# In[40]:


yu["windspeed"].value_counts().head(10)


# In[41]:


yu["casual"].value_counts().head(10)


# In[16]:


yu["registered"].value_counts().head(10)


# In[21]:


yu["count"].value_counts().head(10)


# In[50]:


yu["date"].value_counts().head(10)


# In[49]:


yu["time"].value_counts().head(10)


# In[9]:


yu.isna().sum()


# In[4]:


#converting object to datetime 
yu["datetime"]=yu["datetime"].astype("datetime64[ns]")


# In[15]:


yu.info()


# In[11]:


yu[["date","time"]]=yu["datetime"].str.split(" ",expand=True)


# In[13]:


yu["date"]=yu["date"].astype("datetime64[ns]")


# In[14]:


yu["time"]=yu["time"].astype("datetime64[ns]")


# In[17]:


yu.drop(columns=["datetime"],inplace=True)


# In[48]:


yu[["workingday","weather","season","count","temp","atemp"]].corr()


# In[49]:


yu.corr()


# In[33]:


#convert numerical to categorical
cat_cols=["weather","season"]

for i in cat_cols:
    yu[i]=yu[i].astype("object")


# In[34]:


yu.dtypes


# In[14]:


#a relation between the dependent and independent variable (Dependent “Count” & Independent: Workingday, Weather, Season etc)

from scipy.stats import ttest_ind

ttest_ind(yu["workingday"],yu["count"])


# In[40]:


ttest_ind(yu["weather"],yu["count"])


# In[43]:


ttest_ind(yu["season"],yu["count"])


# # Univariate Analysis

# In[20]:


hist=["count","temp","atemp","humidity","windspeed","casual","registered","holiday"]

fig,axs=plt.subplots(nrows=4,ncols=2,figsize=(30,20))
c=0

for i in range(4):
    for j in range(2):
        sns.histplot(data=yu,x=hist[c],kde=True,ax=axs[i,j])
        c+=1
plt.show()


# # Bivariate Analysis:
# (Relationships between important variables such as workday and count, season and count, weather and count.

# In[19]:


hist=["workingday","holiday","weather","season"]

fig,axs=plt.subplots(nrows=2,ncols=2,figsize=(20,10))
c=0

for i in range(2):
    for j in range(2):
        sns.barplot(data=yu,x=hist[c],y=yu["count"],ax=axs[i,j])
        plt.legend(loc="upper right")
        c+=1
plt.show()


#sns.countplot(data=yu,x="workingday")


# In[28]:


hist=["workingday","weather","season"]

fig,axs=plt.subplots(nrows=1,ncols=3,figsize=(20,10))
c=0

for i in range(3):
        sns.countplot(data=yu,x=hist[c],ax=axs[i])
        plt.legend(loc="upper right")
        c+=1
plt.show()


# In[ ]:





# # Bivariate analysis

# In[51]:


sns.barplot(x=yu["workingday"],y=yu["count"],hue=yu["season"])


# In[52]:


sns.barplot(x=yu["weather"],y=yu["count"],hue=yu["season"])


# In[51]:


hist=["count","temp","atemp","humidity","windspeed","casual","registered"]

fig,axs=plt.subplots(nrows=4,ncols=2,figsize=(30,20))
c=0

for i in range(4):
    for j in range(2):
        sns.boxplot(data=yu,x=hist[c],ax=axs[i,j])
        c+=1
plt.show()


# In[53]:


sns.boxplot(yu["season"])


# In[ ]:





# ## Hypothesis Testing

# In[20]:


#Test for gaussian 
from statsmodels.graphics.gofplots import qqplot
qqplot(yu["workingday"],line="s")


# In[37]:


#checking variances of 2 groups 

levene(yu["workingday"],yu["count"])


# levene test shows that variances are not equal between both groups and there is significant difference.

# #     2- Sample T-Test

# In[56]:


# with assumption of 95% CI ,we consider significance level to be 5%.
#H0:Working Day has no effect on the number of electric cycles rented 
#Ha:Working Day has an effect on the number of electric cycles rented

from scipy.stats import ttest_ind,levene,shapiro,f_oneway,chisquare,chi2,chi2_contingency

t_stat,p_value= ttest_ind(list(yu["workingday"]),list(yu["count"]))


if p_value < 0.05:
    print("t_stat",t_stat)
    print("p_value",p_value)
    
    print("Working Day has an effect on the number of electric cycles rented. \n")
    
else:
    print("Working Day has no effect on the number of electric cycles rented.")


# # ANNOVA 

# In[40]:


#Anova

#1. test gaussian distribution

qqplot(yu["count"],line="s")


# In[41]:


qqplot(yu["weather"],line="s")


# In[42]:


qqplot(yu["season"],line="s")


# In[22]:


#2.levene test

sns.histplot(data=yu,x="count",hue="weather",color="g",kde=True)


# Distribution is right-skewed and is not uniformly distributed.people prefering yulu decreases or is seemingly distributed only until 600 total bikes.

# In[52]:


#H0:weather & count means are equal
#Ha: weather & count means are not equal

ttest_ind(yu["weather"],yu["count"])

# p_value is less than 5% alpha value .Thus reject H0, both groups has different means and are drwan from different population.
# In[53]:


#H0:weather & count variances are equal
#Ha: weather & count variances are not equal
levene(yu["weather"],yu["count"])


# levene test also confirms that variances are not equal.

# In[156]:


yu["randomgp"]=np.random.choice(["g1","g2","g3","g4"],size=len(yu))

g1=yu[yu["randomgp"]=="g1"]["weather"]
g2=yu[yu["randomgp"]=="g2"]["weather"]
g3=yu[yu["randomgp"]=="g3"]["weather"]
g4=yu[yu["randomgp"]=="g4"]["weather"]


f_stat,p_value=f_oneway(g1,g2,g3,g4)

if p_value < 0.05:
    print("f_stat",f_stat)
    print("p_value",p_value)
    
    print("weather has an effect on the number of electric cycles rented. \n")
    
else:
    print("f_stat",f_stat)
    print("p_value",p_value)
    print("weather has no effect on the number of electric cycles rented.")


# In[157]:


sns.boxplot(x="randomgp",y="count",data=yu)


# High p_value greater than 5% signioficance level which tells us that means of all the groups are near to each other.Thus the difference is by chance and not significantly different.

# In[160]:


import warnings
warnings.filterwarnings("ignore") 


g1=yu[yu["weather"]==1]["count"]
g2=yu[yu["weather"]==2]["count"]
g3=yu[yu["weather"]==3]["count"]
g4=yu[yu["weather"]==4]["count"]


f_stat,p_value=f_oneway(g1,g2,g3,g4)

if p_value < 0.05:
    print("f_stat",f_stat)
    print("p_value",p_value)
    
    print("weather has an effect on the number of electric cycles rented.\n")
    
else:
    print("f_stat",f_stat)
    print("p_value",p_value)
    print("weather has no effect on the number of electric cycles rented.")


# Low p_value greater than 5% significance level which tells us that means of all the groups are not near to each other.Thus they are significantly different.

# In[161]:


# season is not gaussian shown in above qqplot

#2.levene test

sns.histplot(data=yu,x="count",hue="season",color="g",kde=True)

Distribution is right-skewed and is not uniformly distributed.people prefering yulu decreases or is seemingly distributed only until range of 600 total bikes.
# In[163]:


#H0:weather & count means are equal
#Ha: weather & count means are not equal

ttest_ind(yu["season"],yu["count"])


# In[ ]:


# p_value is less than 5% alpha value .Thus reject H0, both groups has different means and are drwan from different population.


# In[23]:


#H0:season & count variances are equal
#Ha: season & count variances are not equal
levene(yu["season"],yu["count"])

if p_value < 0.05:
    print("Reject H0, season & count variances are not equal")
else:
    print("season & count variances are equal")


# levene test also confirms that variances are not equal.

# In[29]:


#Creating random groups g1,g2,g3,g4 to test if season has an effect on the number of electric cycles rented

yu["randomgp"]=np.random.choice(["g1","g2","g3","g4"],size=len(yu))

g1=yu[yu["randomgp"]=="g1"]["count"]
g2=yu[yu["randomgp"]=="g2"]["count"]
g3=yu[yu["randomgp"]=="g3"]["count"]
g4=yu[yu["randomgp"]=="g4"]["count"]


f_stat,p_value=f_oneway(g1,g2,g3,g4)

if p_value < 0.05:
    print("f_stat",f_stat)
    print("p_value",p_value)
    
    print("season has an effect on the number of electric cycles rented. \n")
    
else:
    print("f_stat",f_stat)
    print("p_value",p_value)
    print("season has no effect on the number of electric cycles rented.")


# As shown above High p_value greater than 5% significance level which tells us that means of all the groups are near to each other.Thus the difference is by chance and not significantly different.

# In[328]:


#Creating groups based on actual values of season 1,2,3,4.
import warnings
warnings.filterwarnings("ignore") 


g1=yu[yu["season"]==1]["count"]
g2=yu[yu["season"]==2]["count"]
g3=yu[yu["season"]==3]["count"]
g4=yu[yu["season"]==4]["count"]


f_stat,p_value=f_oneway(g1,g2,g3,g4)

if p_value < 0.05:
    print("f_stat",f_stat)
    print("p_value",p_value)
    
    print("season has an effect on the number of electric cycles rented.\n")
    
else:
    print("f_stat",f_stat)
    print("p_value",p_value)
    print("season has no effect on the number of electric cycles rented.")


# Low p_value greater than 5% significance level which tells us that means of all the groups are not near to each other.Thus they are significantly different.

# # Chi-square test 

# In[31]:


val=pd.crosstab(yu["weather"],yu["season"])
val


# In[37]:


#H0:Weather & season are both independent
#Ha:Weather & season are both dependent

chi2_contingency(val)


# # Since pvalue is very low nearly 0 then we can reject H0 i.e.,Weather & season are both independent.

# In[47]:


sns.barplot(x=yu["weather"],y=yu["count"],hue=yu["season"])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




