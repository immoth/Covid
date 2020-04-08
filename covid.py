# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:29:35 2020

@author: jsten
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker

############################### Downloading Data #########################################
df = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv', parse_dates=['Date'])
countries = ['Italy', 'Germany', 'United Kingdom', 'US', 'France', 'China']
df = df[df['Country'].isin(countries)]
df['Cases'] = df[['Confirmed', 'Recovered', 'Deaths']].sum(axis=1)
df = df.pivot(index='Date', columns='Country', values='Cases')
##########################################################################################


Pc = df['France'] #total number of cases number of cases

dPc = [(Pc[i+1]-Pc[i]) for i in range(0,len(Pc)-1)] #number of new cases in a day (i.e. derivative of Pc)

dPcByPc = [(dPc[i]+0.1)/(Pc[i+1]+0.1) for i in range(0,len(dPc))] #ratio of total cases to new cases



######################## Linear Regression ################
x = [Pc[i] for i in range(0,len(dPcByPc))]
yl = dPcByPc
c_reg = 0
c_div = 0
x_avg=np.average(x)
y_avg=np.average(yl)
for i in range(0,len(yl)-1):
    c_reg+=(x[i]-x_avg)*(yl[i]-y_avg)
    c_div+=(x[i]-x_avg)**2
c_reg=c_reg/c_div
y0_reg=y_avg-c_reg*x_avg

k = y0_reg
L= - y0_reg/c_reg
x0_lst = np.array([i+(1/k)*np.log(L/Pc[i]-1) for i in range(40,len(Pc))])
x0_lst = x0_lst[np.isfinite(x0_lst)]
x0 = np.average(x0_lst)
############################################################


dPcByPc_reg=[y0_reg+c_reg*x[i] for i in range(0,len(x))] # Regression model for the ratio of total cases to new cases

dPc_reg=[y0_reg*x[i]+c_reg*x[i]**2 for i in range(0,len(x))] #Regression model for the number of new cases in a day

Pc_reg = [L/(1+np.exp(-k*(i-x0))) for i in range(0,len(Pc))] # regression model for total cases

############# Plotting #################
idx = [i for i in range(0,len(dPcByPc))]
plt.figure('Ratio of new cases to total cases')
plt.scatter(idx,dPcByPc)
plt.scatter(idx,dPcByPc_reg)
plt.show()
########################################

############# Plotting #################
idx = [i for i in range(0,len(dPc))]
plt.figure('New cases')
plt.scatter(idx,dPc)
plt.scatter(idx,dPc_reg)
plt.show()
########################################

############# Plotting #################
idx = [i for i in range(0,len(Pc))]
plt.figure('Total cases')
plt.scatter(idx,Pc)
plt.scatter(idx,Pc_reg)
########################################


################################################## Extending the Models #####################################################


Pc_reg_ex = [L/(1+np.exp(-k*(i-x0))) for i in range(0,len(Pc)+100)] # Regression model for total cases

dPc_reg_ex=[y0_reg*Pc_reg_ex[i]+c_reg*Pc_reg_ex[i]**2 for i in range(0,len(Pc_reg_ex))] #Regression model for the number of new cases in a day

dPcByPc_reg_ex=[y0_reg+c_reg*Pc_reg_ex[i] for i in range(0,len(Pc_reg_ex))] # Regression model for the ratio of total cases to new cases

############# Plotting #################
idx = [i for i in range(0,len(dPcByPc))]
idxx = [i for i in range(0,len(dPcByPc_reg_ex))]
plt.figure('Ratio of new cases to total cases predictions')
plt.scatter(idx,dPcByPc)
plt.scatter(idxx,dPcByPc_reg_ex)
########################################

############# Plotting #################
idx = [i for i in range(0,len(dPc))]
idxx = [i for i in range(0,len(dPc_reg_ex))]
plt.figure('New cases predictions')
plt.scatter(idx,dPc)
plt.scatter(idxx,dPc_reg_ex)
########################################

############# Plotting #################
idx = [i for i in range(0,len(Pc))]
idxx = [i for i in range(0,len(Pc_reg_ex))]
plt.figure('Total cases predictions')
plt.scatter(idx,Pc)
plt.scatter(idxx,Pc_reg_ex)
########################################

