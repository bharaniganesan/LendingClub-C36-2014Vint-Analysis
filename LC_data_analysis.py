#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 22:49:40 2019

@author: Bharani Ganesan
"""

from process_LC_data import load_as_dataframe
from process_LC_data import filter_lc_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set up the files
input_file_name = 'PMTHIST_ALL_201907.csv'
output_file_name = 'PMTHIST_FILTERED2.csv'

# Filter out data using criteria
# performance of grade-C, 3-year loans 
#print('Filtering out the data')
#filter_lc_data(input_file_name, output_file_name)

# Load the filtered data as dataframe  
print('Loading as a data frame')
lc_dataframe = load_as_dataframe(output_file_name)

#Check the column types after importing
print(lc_dataframe.dtypes)

#Convert dates read as objects to date format
#lc_dataframe['MONTH'] = lc_dataframe['MONTH'].apply(pd.to_datetime)

#Add the mid-point application FICO
lc_dataframe['Mid_App_FICO'] = (lc_dataframe['APPL_FICO_BAND'].str.split('-')).apply(lambda x: (float(x[0]) + float(x[1]))/2)

#The data towards the end becomes very thin due to most loans paying off.
#So keep performance data only till end of 2017.

lc_dataframe_short = lc_dataframe.loc[lc_dataframe.MONTH <= '2017-12-31']

#Sort the dataframe by loan ID by date
lc_dataframe_short = lc_dataframe_short.sort_values(by=['LOAN_ID', 'MONTH'])

#Get the first occurance of each loan to plot portfolio stats
first_loan_row = lc_dataframe_short.groupby('LOAN_ID').first().reset_index()

#Export the dataset to csv
first_loan_row.to_csv('FIRST_LOAN_OCCURANCE.CSV')

'''
#Get the first occurance of all "Default and "Fully Paid" loans to plot comparisons
lc_def_paidoff_first = (lc_dataframe.loc[(lc_dataframe['PERIOD_END_LSTAT']=='Fully Paid') |\
                                  (lc_dataframe['PERIOD_END_LSTAT']=='Default')])\
                                    .groupby('LOAN_ID').first().reset_index()

#Export the dataset to csv
lc_def_paidoff_first.to_csv('FIRST_DEFAULT_PAYOFF.CSV')
'''

'''
########### Calculate the Net Monthly and Annualized Returns for each month ###########
'''
data_grpby_month = lc_dataframe_short.groupby('MONTH').sum().reset_index()
data_grpby_month['Net Monthly Return'] = (data_grpby_month['INT_PAID'] + data_grpby_month['FEE_PAID']\
                            - (data_grpby_month['INT_PAID'] + data_grpby_month['FEE_PAID']) * (0.01)\
                            - data_grpby_month['COAMT'] - data_grpby_month['PCO_COLLECTION_FEE']\
                            + data_grpby_month['PCO_RECOVERY'])/data_grpby_month['PBAL_BEG_PERIOD']

data_grpby_month['NAR'] = (1 + data_grpby_month['Net Monthly Return']) ** 12 - 1

#Some descriptive stats and charts about the returns
data_grpby_month.NAR.describe()

plt.plot(data_grpby_month.MONTH, data_grpby_month['NAR'])
plt.title("Monthly Annualized NAR During Performance Window")

sns.distplot(data_grpby_month['NAR'])
plt.title("Distribution of Monthly Annualized NAR During Performance Window")

#Pivot the table to get monthly NAR
monthly_nar_pivot = pd.pivot_table(data_grpby_month, values = 'NAR', index = 'MONTH',\
                                      fill_value = 0)

#Export the data to csv for plotting in Tableau
monthly_nar_pivot.to_csv("Monthly_NAR.csv")

'''
########### Group the NAR data by Month and State ###########
'''
data_grpby_month_state = lc_dataframe_short.groupby(['MONTH','State'] ).sum().reset_index()
data_grpby_month_state['Net Monthly Return'] = (data_grpby_month_state['INT_PAID'] + data_grpby_month_state['FEE_PAID']\
                            - (data_grpby_month_state['INT_PAID'] + data_grpby_month_state['FEE_PAID']) * (0.01)\
                            - data_grpby_month_state['COAMT'] - - data_grpby_month_state['PCO_COLLECTION_FEE']\
                            + data_grpby_month_state['PCO_RECOVERY'])/data_grpby_month_state['PBAL_BEG_PERIOD']

data_grpby_month_state['NAR'] = (1 + data_grpby_month_state['Net Monthly Return']) ** 12 - 1

nar_month_state_pivot = pd.pivot_table(data_grpby_month_state, values = 'NAR', index = 'MONTH',\
                                      columns = 'State', fill_value = 0)

'''
########### Group the NAR data by Month and Loan Status ###########
'''
data_grpby_month_lstat = lc_dataframe_short.groupby(['MONTH','PERIOD_END_LSTAT'] ).sum().reset_index()

monthly_beg_pbal = data_grpby_month_lstat.groupby('MONTH').agg({'PBAL_BEG_PERIOD':'sum'})
monthly_beg_pbal = monthly_beg_pbal.rename(columns = {'PBAL_BEG_PERIOD':'Total_PBAL_Beg'})

#Merge the monthly balance to get the different rates
data_grpby_month_lstat = pd.merge(data_grpby_month_lstat, monthly_beg_pbal, on = 'MONTH')
data_grpby_month_lstat['LSTAT_RATE'] = data_grpby_month_lstat['PBAL_BEG_PERIOD']\
                                            .divide(data_grpby_month_lstat['Total_PBAL_Beg'])

#Export to csv for analysis in Tableau
data_grpby_month_lstat.to_csv('Bal by Month and Loan State.csv')

'''
########### Risk Contribution ###########
'''
#Add the total NAR by month to the NAR by state and month
#Try and calculate risk contribution
nar_state_total_month = pd.merge(nar_month_state_pivot, monthly_nar_pivot, on = 'MONTH')
state_pf_corr = nar_state_total_month.corr()['NAR'][:-1] #Retained Risk
state_std = nar_state_total_month.std(axis = 0)[:-1] #Remove the last row for full portfolio
state_loan_amt = first_loan_row.groupby('State').agg({'PBAL_BEG_PERIOD':'sum'})
state_loan_weight = np.true_divide(state_loan_amt.PBAL_BEG_PERIOD,state_loan_amt['PBAL_BEG_PERIOD'].sum())

riskCont_state = np.multiply(state_pf_corr, state_std)
pfVol = riskCont_state.dot(state_loan_weight)

monthly_nar_pivot.describe()

