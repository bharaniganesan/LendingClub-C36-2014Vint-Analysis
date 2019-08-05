# Risk and Return Analysis of Lending Club's Loans
## Population: 'C' Grade, 36 Month Term Loans Issued in 2014

### Executive Summary
This analysis examines the performance of Lending Club's 'C' grade loans with a term of 36 months, which were issued in 2014. Their performance was tracked till the end of December 2017.

## Design Choices
### Python
1. Generators - Memory efficient way to read the large input data
2. Pandas DataFrames - Efficient way to slice and dice the large data set
3. scipy.optimize - To determine a more optimal portfolio mix

### Tableau
1. Worksheets
2. Dashboards
3. Story to present results

## Code Structure
### Parsing the file from Lending Club (process_LC_data.py)
The script in 'process_LC_data.py' parses the data from the .csv file found here: https://lendingclub.com/company/additional-statistics > All payments (includes payments made to investors and to LendingClub).

In order to handle the large volume of data in the above file, this script implements a Python generator that yields one record at a time. The record is then examined to determine if the loan is a 'C' grade, 36 month loan issued in 2014. This subset of data is stored in an intermediate .csv file "PMTHIST_FILTERED.csv".


### Data Analysis Using Python (LC_data_analysis.py)
This script loads the data from "PMTHIST_FILTERED.csv" to a Pandas DataFrame. This script restricts the performance window till December 2017, and also keeps loans which have a status of "Current" in their first occurance in the file.

This script executes the following analyses:
1. Calculate the Net Monthly and Annualized Returns for each month
2. Calculate the Net Cumulative Annualized Returns for each month
3. Group the NAR data by Month and State
4. Group the NAR data by Month and Loan Status
5. Calculate Risk Contribution and Retained Risk by State
6. Optimize the weights of different States to get better return/risk via scipy.optimize.minimize package
7. Flags defaulted loans to calculate cumulative default rates by different cuts

The script also creates .csv files corresponding to the above analyses, which are used to visualize the results in Tableau.

## Running the Code Locally
To run the code locally, download the Python scripts mentioned above. Execute the following command in the command prompt where the scripts are stored. Ensure that the Lending Club .csv file is also in the same directory.

```
python LC_data_analysis.py
```

## Data Visualization Via Tableau
The results of the analyses performed from the Python scripts above are exported to .csv files and have been converted to visualizations in the Tableau file located here: https://public.tableau.com/profile/bharani.ganesan#!/vizhome/LCDataAnalysis/Story1

## Further Code Optimization

Given more time here are some of the code cleanups and optimizations that can be implemented.
1. Refactor repeating code into functions (for example NAR calculations) to avoid code duplication
2. Move each analysis into a function in a separate file to make the code more readable
