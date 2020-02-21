import warnings
import itertools
import matplotlib
import seaborn as sns
import re
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np

from pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from datetime import datetime

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'G'


def read_file(file_path_name : str) -> pd.DataFrame:
    '''
    This function takes a CSV with 2 columns and transforms it to a dataframe

    It then cleans the dataframe and returns the cleaned dataframe
    '''

    print('Reading CSV...')
    df = pd.read_csv(file_path_name).dropna()

    if len(df.columns) > 2:
        st.write('File has more than 2 columns, should be:[Date, Metric to be forecasted]')
        return 'File has more than 2 columns, should be:[Date, Metric to be forecasted]'

    print('\nCleaning data...')
    y_total = df[[df.columns[0], df.columns[1]]]
    y_total[df.columns[1]] = y_total[df.columns[1]].astype(str).apply(lambda x: re.sub('\D','',x)).astype(float)
    y_total[df.columns[0]] = pd.to_datetime(y_total[df.columns[0]])

    print('\nDone cleaning data...')

    st.write("**Raw File Table**")
    st.write(y_total.set_index(df.columns[0]))
    return y_total.set_index(df.columns[0])
    

def check_stationarity(cleaned_dataframe : pd.DataFrame) -> int:
    '''
    This function takes a dataframe and checks to see if the data is stationary

    We expect the data to have seasonality (given the model used to forecast)
    '''   

    pvalue = adfuller(cleaned_dataframe[cleaned_dataframe.columns])

    if pvalue[1] > 0.05:
        st.markdown(f"<b style = 'color: red'>Warning: P-value = {round(pvalue[1],2)} indicates that the data may be stationary, you may need to use the stationary model instead of the seasonal model</b>", unsafe_allow_html=True)

    print(f'\nP-value = {round(pvalue[1], 2)}')


def grid_search_hyperparameters(cleaned_dataframe : pd.DataFrame) -> list:
    '''
    This function takes in a cleaned dataframe and will find the optimal hyperparameters to pass through to the model

    It then returns a dictionary containing 
    '''

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    aic = []
    combinations = {}

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(cleaned_dataframe,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
                results = mod.fit()
                combinations[results.aic] = [param, param_seasonal]
                aic.append(results.aic)
            except: 
                continue
                
    min_aic_combinations = combinations[min(aic)]
    
    print(f'\nMinimum AIC Combination:\n\nTrend Elements: {min_aic_combinations[0]}\nSeasonal Trend Elements: {min_aic_combinations[1]}')

    return min_aic_combinations


def fit_model(cleaned_dataframe : pd.DataFrame, hyperparameter_combo : list):
    '''
    This model will take in the hyperparameter combinations and the dataset to fit the model

    It then returns the trained model
    '''

    mod = sm.tsa.statespace.SARIMAX(cleaned_dataframe,
                                order= hyperparameter_combo[0],
                                seasonal_order= hyperparameter_combo[1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results = mod.fit()

    return results


def plot_model_figures(model_results, cleaned_dataframe : pd.DataFrame, alpha_level : int, hyperparameter_combo : list):
    '''
    This function takes in the fitted model and cleaned dataframe to plot forecasted figures and model diagnosis

    This function doesn't return anything
    '''
    st.markdown("<h3 style='text-align: center'>Data's Decomposition</h2>", unsafe_allow_html=True)
    
    rcParams['figure.figsize'] = 18, 8
    decomposition = sm.tsa.seasonal_decompose(cleaned_dataframe, model='additive')
    decomposition_fig = decomposition.plot()
    plt.show()

    st.write(decomposition_fig)


    print('\n\nModel Diagnostics Figure saved as PDF')
    model_results.plot_diagnostics(figsize=(18, 8))

    st.markdown("<h3 style='text-align: center'>Model Diagnostics</h2>", unsafe_allow_html=True)
    st.write(model_results.plot_diagnostics(figsize=(18, 8)))
    
    if st.button('Download Image') == True:
        plt.savefig(f'{cleaned_dataframe.columns[0].lower()}_forecast_model_diagnostics.pdf')
        st.write(f'Saved: {cleaned_dataframe.columns[0].lower()}_forecast_model_diagnostics.pdf')
    
    print('\nForecast Model Overlay saved as PDF')
    
    st.markdown("<h3 style='text-align: center'>Forecast Overlay Evaluation</h2>", unsafe_allow_html=True)

    mod_eval = sm.tsa.statespace.SARIMAX(cleaned_dataframe[:-13],
                                order= hyperparameter_combo[0],
                                seasonal_order= hyperparameter_combo[1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results_eval = mod_eval.fit()

    pred = results_eval.get_prediction(start = cleaned_dataframe.index[-12], end = cleaned_dataframe.index[-1], dynamic=False)
    pred_ci = pred.conf_int(alpha = alpha_level)

    cleaned_dataframe = cleaned_dataframe.reset_index()
    
    prediction_line = pd.DataFrame(
        data = pred.predicted_mean,
        columns = [cleaned_dataframe.columns[1]],
        index = pred.predicted_mean.index
    ).reset_index().rename(columns = {'index' : cleaned_dataframe.columns[0]})

    forecast_overlay_fig, ax = plt.subplots( 1, 1, figsize=(14,7))

    sns.lineplot(x = cleaned_dataframe.columns[0], y = cleaned_dataframe.columns[1], data = cleaned_dataframe.set_index(cleaned_dataframe.columns[0])['2015':].reset_index())
    sns.lineplot(x = cleaned_dataframe.columns[0], y = cleaned_dataframe.columns[1], data = prediction_line)
    ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color = 'k', alpha = alpha_level)
    # ax.set_title('Forecast Overlay', color = '#000000')

    plt.legend(['Current Data', 'Forecast'], loc = 'upper right')
    plt.show()

    st.write(forecast_overlay_fig)

    return decomposition_fig, forecast_overlay_fig

    

def model_evaluation(cleaned_dataframe : pd.DataFrame, hyperparameter_combo : list) -> int:
    '''
    This function takes in a cleaned dataframe and the hyperparameter combinations to evaluate the model's fit

    This function doesn't return anything
    '''
   
    mod_eval = sm.tsa.statespace.SARIMAX(cleaned_dataframe[:-13],
                                order= hyperparameter_combo[0],
                                seasonal_order= hyperparameter_combo[1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results_eval = mod_eval.fit()


    y_pred = pd.DataFrame(
        data = results_eval.get_prediction(start = cleaned_dataframe.index[-12], end = cleaned_dataframe.index[-1] , dynamic=False).predicted_mean,
        columns = [cleaned_dataframe.columns[0] + '_Predictions']
    )
    y_pred[cleaned_dataframe.columns[0] + '_Predictions'] = y_pred[cleaned_dataframe.columns[0] + '_Predictions'].astype(int)

    y_true = cleaned_dataframe[-12:]
    y_true = y_true.reset_index()
    y_true[y_true.columns[0]] = pd.to_datetime(y_true[y_true.columns[0]])
    y_true[y_true.columns[1]] = y_true[y_true.columns[1]].astype(str).apply(lambda x: re.sub('\D','',x))
    y_true[y_true.columns[1]] = y_true[y_true.columns[1]].astype(str).apply(lambda x: re.sub('0$','',x)).astype(float)
    y_true = y_true.set_index(y_true.columns[0])
    
    concat = pd.concat([y_pred, y_true], axis = 1)
    concat['SQRD_ERROR'] =  (concat[concat.columns[0]] - concat[concat.columns[1]]) ** 2
    
    MSE = (concat['SQRD_ERROR'].sum()) / concat.shape[0]
    RMSE = np.sqrt(MSE)
    Weighted_RMSE_Impact = RMSE / np.mean(concat[cleaned_dataframe.columns[0]]) * 100
    
    MAPE_ = round(np.mean(np.abs((concat[cleaned_dataframe.columns[0]].sum() - concat[cleaned_dataframe.columns[0] + '_Predictions'].sum()) / concat[cleaned_dataframe.columns[0]].sum())) * 100,2)
    MSE_ = int(round(MSE))
    RMSE_ = int(round(RMSE))
    WRMSE_Impact = round(Weighted_RMSE_Impact, 2)

    st.markdown("<h3 style='text-align: center'>Model Evluation</h2>", unsafe_allow_html=True)
    st.markdown("<div></div>", unsafe_allow_html=True)
    st.write("Mean Absolute Percentage Error (MAPE) = " + str(MAPE_) + "%")
    st.write("Mean Squared Error (MSE) = " + "{:,}".format(MSE_))
    st.write("Root Mean Squared Error (RMSE) = " + "{:,}".format(RMSE_))
    st.write("Weighted RMSE Impact = " + str(WRMSE_Impact) + "%")


def forecast_vizuals(cleaned_dataframe : pd.DataFrame, alpha_level : int, hyperparameter_combo : list, prediction_start : str, prediction_end : str):
    '''
    This function takes in various parameters to to visualize the forecast

    This function doesn't return anything
    '''

    model = sm.tsa.statespace.SARIMAX(cleaned_dataframe,
                                order= hyperparameter_combo[0],
                                seasonal_order= hyperparameter_combo[1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    model_results = model.fit()

    pred = model_results.get_prediction(start = prediction_start, end = prediction_end, dynamic=False)
    pred_ci = pred.conf_int(alpha = alpha_level)

    prediction_line = pd.DataFrame(
        data = pred.predicted_mean,
        columns = [cleaned_dataframe.columns[0]],
        index = pred.predicted_mean.index
    ).reset_index().rename(columns = {'index' : cleaned_dataframe.reset_index().columns[0]})

    forecast_fig, ax = plt.subplots( 1, 1, figsize=(14,7))

    sns.lineplot(x = cleaned_dataframe.reset_index().columns[0], y = cleaned_dataframe.columns[0], data = cleaned_dataframe['2015':].reset_index())
    sns.lineplot(x = cleaned_dataframe.reset_index().columns[0], y = cleaned_dataframe.columns[0], data = prediction_line)
    ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color = 'k', alpha = alpha_level)

    plt.legend()
    plt.show()

    custom_title = re.sub('_', ' ', cleaned_dataframe.columns[0]).title()
    st.markdown(f"<h3 style='text-align: center'>{custom_title} Forecast</h2>", unsafe_allow_html=True)
    st.write(forecast_fig)


def forecast_results(cleaned_dataframe : pd.DataFrame, alpha_level : int, hyperparameter_combo : list, prediction_start : str, prediction_end : str):
    
    model = sm.tsa.statespace.SARIMAX(cleaned_dataframe,
                                order= hyperparameter_combo[0],
                                seasonal_order= hyperparameter_combo[1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    model_results = model.fit()

    pred = model_results.get_prediction(start = prediction_start, end = prediction_end, dynamic=False)
    pred_ci = pred.conf_int(alpha = alpha_level)

    custom_title = re.sub('_', ' ', cleaned_dataframe.columns[0]).title()

    predictions_mean = pd.DataFrame(pred.predicted_mean)
    predictions_table = (predictions_mean
        .rename(columns = {predictions_mean.columns[0] : "Predicted " + cleaned_dataframe.columns[0]})
    )

    merged_forecast_figs = (predictions_table
        .merge(pred_ci, on = predictions_table.index, how = 'left')
        .rename(columns = {"key_0" : "Date"})
        )

    merged_forecast_figs[merged_forecast_figs.columns[1]] = merged_forecast_figs[merged_forecast_figs.columns[1]].apply(lambda x: "{:,}".format(round(x)))
    merged_forecast_figs[merged_forecast_figs.columns[2]] = merged_forecast_figs[merged_forecast_figs.columns[2]].apply(lambda x: "{:,}".format(round(x)))
    merged_forecast_figs[merged_forecast_figs.columns[3]] = merged_forecast_figs[merged_forecast_figs.columns[3]].apply(lambda x: "{:,}".format(round(x)))

    merged_forecast_figs = merged_forecast_figs[[merged_forecast_figs.columns[0], merged_forecast_figs.columns[2], merged_forecast_figs.columns[1], merged_forecast_figs.columns[3]]]

    st.markdown(f"<h3 style='text-align: center'>merged_forecast_figs</h2>", unsafe_allow_html=True)
    st.write(merged_forecast_figs)


    return merged_forecast_figs


def forecast_program(file_path_name : str, alpha_level : int, prediction_start : str, prediction_end : str):
    
    cleaned_dataframe = read_file(file_path_name)

    title_name = re.sub('_',' ',cleaned_dataframe.columns[0].lower()).title()

    st.markdown(f"<h1 style='text-align: center;'>{title_name} Monthly Forecast</h1>", unsafe_allow_html=True)

    check_stationarity(cleaned_dataframe)

    hyperparameter_combo = grid_search_hyperparameters(cleaned_dataframe)

    model_results = fit_model(cleaned_dataframe, hyperparameter_combo)

    plot_model_figures(model_results, cleaned_dataframe, alpha_level, hyperparameter_combo)

    model_evaluation(cleaned_dataframe, hyperparameter_combo)

    forecast_vizuals(cleaned_dataframe, alpha_level, hyperparameter_combo, prediction_start, prediction_end)

    forecast_results(cleaned_dataframe, alpha_level, hyperparameter_combo, prediction_start, prediction_end)




st.markdown("<h1 style='text-align: center;'>Forecasting Program</h1>", unsafe_allow_html=True)

st.write('''
    **Please read these requirements before utilizing the program:**
    \n- This program takes CSV files only
    \n- The CSV should only have 2 columns, date & metric to be forecasted (i.e. sales, website sessions, etc.)
    \n- Please be sure that the number of periods in the dataset is consistent with forecasting model used (weekly periods vs. monthly periods)
    \n- If the dataset is in monthly periods simply set the day to 01, and if in weekly periods assure that the date column is a date type
    \n- The periodical steps you desire to forecast will be the difference between the start and end date inputed (i.e. Monthly: 2019-12-01 - 2020-01-1 will be 1 step, Weekly (week starting on Monday): 2019-12-02 - 2019-12-09 will be 1 step)
    \n- The file that runs this program must be placed in the same location as the CSV files to be used in order to feed the forecasting model
    \n- File name input is not case sensitive but please be sure that all characters are correct, the program will throw an error otherwise
    \n- The alpha input is to determine the model's level of confidence in it's projections, input must be an integer ranging from 0 to 1 
''')

file_source_input = st.text_input("Please enter file name:")
alpha_level = st.number_input("Please enter alpha level for forecast:")
forecast_start_date = st.text_input("Please enter forecast start date (ex. 2019-12-01 YYYY-MM-DD):")
forecast_end_date = st.text_input("Please enter forecast end date (ex. 2019-12-01 YYYY-MM-DD):")


if '.csv' in file_source_input:
    file_source_clean = file_source_input.lower().strip()
else:
    file_source_clean = file_source_input.lower().strip() + '.csv'


value = st.sidebar.selectbox("Which forecasting:", ["Seasonal Monthly Forecast Program", "Seasonal Weekly Forecast Program"])
st.write(value)

if file_source_clean == '.csv':
    st.write("Waiting for required inputs above ↑")
else:
    forecast_program(file_source_clean, alpha_level, forecast_start_date, forecast_end_date)