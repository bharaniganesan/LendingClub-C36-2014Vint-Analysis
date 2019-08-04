#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 18:07:07 2019

@author: Bharani Ganesan

Script that reads LC "All Payments" data and picks data that matches certain criteria
"""
import csv
import pandas as pd


#Read the PMTHIST_ALL_201907.csv file using generators to handle the large file size
def get_lc_data(csv_fname):
    with open(csv_fname, "r") as all_payment_recs:
        for pmt_rows in csv.reader(all_payment_recs):
            yield pmt_rows
            
def filter_lc_data(input_file_name, output_file_name):
    print('Filtering out data')
    
    year_list = ['2014','2015']

    with open(output_file_name,'w') as file:
        wr = csv.writer(file)
        row_count = 0
        match_count = 0
    
        for row in get_lc_data(input_file_name):
            if row_count == 0:
                wr.writerow(row)
                
            elif row[33] == 'C' and row[34] == '36' and '2014' in row[15]:
                wr.writerow(row)
                match_count += 1
            
            row_count += 1
            #print(f'match count = {match_count}')
            #if match_count == 50:
                #break
            

def load_as_dataframe(file_name):
    return pd.read_csv(file_name, parse_dates = ['MONTH','IssuedDate'],\
                                                       infer_datetime_format = True)

# Call the function when the script is executed
if __name__ == "__main__":
    print("In process_LC_data.py")
    input_file_name = 'PMTHIST_ALL_201907.csv'
    output_file_name = 'PMTHIST_FILTERED2.csv'
    filter_lc_data(input_file_name, output_file_name)
    lc_dataframe = load_as_dataframe(output_file_name)
    
    #The step below gave an error as some entries are coded "MISSING" 
    #lc_dataframe['Mid_Last_FICO'] = (lc_dataframe['Last_FICO_BAND'].str.split('-')).apply(lambda x: (float(x[0]) + float(x[1]))/2)
    not36 = lc_dataframe.loc[(lc_dataframe['term'] != 36)]