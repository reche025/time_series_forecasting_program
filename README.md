# Seasonal Time Series Forecasting Program

## Program Description:
 - This program will assist users in forecasting time series data by simply inputting historical time series data
 - 
 - This program is intended to run from an executable file, however if you have an environment set up with all the dependencies installed then you can run the program by typing ```streamlit run main.py```

## Program Requirements:

 - This program takes CSV files only
 - The CSV should only have 2 columns, date & metric to be forecasted (i.e. sales, website sessions, etc.)
 - Please be sure that the number of periods in the dataset is consistent with forecasting model used (weekly periods vs. monthly periods)
 - If the dataset is in monthly periods simply set the day to 01, and if in weekly periods assure that the date column is a date type
 - The periodical steps you desire to forecast will be the difference between the start and end date inputed (i.e. Monthly: 2019-12-01 - 2020-01-1 will be 1 step, Weekly (week starting on Monday): 2019-12-02 - 2019-12-09 will be 1 step)
 - The file that runs this program must be placed in the same location as the CSV files to be used in order to feed the forecasting model
 - File name input is not case sensitive but please be sure that all characters are correct, the program will throw an error otherwise
 - The alpha input is to determine the model's level of confidence in it's projections, input must be an integer ranging from 0 to 1
