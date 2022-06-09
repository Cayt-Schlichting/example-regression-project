import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from itertools import combinations, product, permutations

###### GRAPHING #######
#IMPROVEMENTS TO MAKE: 
# 1) determine type of plot (facetgrid or axes subplot), then set appropriately
# 2) consider making an x vs y parameters
# 3) consider making millions/thousands parameter
# 4) Duplicate plot_variable_pairs and make function that only plots target vs other features
def yticks_mm(ax):
    '''
    Formats the y axis ticks to millions for axes subplots.
    Returns: None
    Inputs: 
        (R) ax: AxesSubplot
    '''
    #Get yticks and format them
    ylabels = ['{:.1f}'.format(y) + 'MM' for y in ax.get_yticks()/1000_000];
    #Force yticks (handles user interaction)
    ax.set_yticks(ax.get_yticks());
    ax.set_yticklabels(ylabels);
    return None

def yticks_k(ax):
    '''
    Formats the y axis ticks to thousands for axes subplots.
    Returns: None
    Inputs: 
        (R) ax: AxesSubplot
    '''
    #Get yticks and format them
    ylabels = ['{:,.1f}'.format(y) + 'K' for y in ax.get_yticks()/1000];
    #Force yticks (handles user interaction)
    ax.set_yticks(ax.get_yticks());
    ax.set_yticklabels(ylabels);
    return None

def xticks_k(ax):
    '''
    Formats the x axis ticks to thousands for axes subplots.
    Returns: None
    Inputs: 
        (R) ax: AxesSubplot
    '''
    #Get yticks and format them
    xlabels = ['{:,.1f}'.format(x) + 'K' for x in ax.get_xticks()/1000];
    #Force yticks (handles user interaction)
    ax.set_xticks(ax.get_xticks());
    ax.set_xticklabels(xlabels);
    return None

def xticks_mm(ax):
    '''
    Formats the x axis ticks to thousands for axes subplots.
    Returns: None
    Inputs: 
        (R) ax: AxesSubplot
    '''
    #Get yticks and format them
    xlabels = ['{:,.1f}'.format(x) + 'MM' for x in ax.get_xticks()/1_000_000];
    #Force yticks (handles user interaction)
    ax.set_xticks(ax.get_xticks());
    ax.set_xticklabels(xlabels);
    return None

def plot_variable_pairs(df,**kwargs):
    '''
    Creates combinations of numeric columns, then creates a scatterplot and regression line.\
    Do not include encoded columns when calling function.
    
    Outputs: Scatterplot with regression line
    Returns: None
    Inputs: 
     (R)          df: Dataframe containing multiple numeric columns.
     (O) sample_size: number of rows to use when plotting.  Default 50_000
    '''
    #only include numeric datatypes 
    #doesn't currently handle datetimes - would want it to plot that on X
    df = df.select_dtypes(include='number')

    #SCATTERPLOTS
    #pull out sample size
    ss = kwargs.get('sample_size',50_000) # Default 50k
    #If sample size is smaller than df rows, pull out a sample of the data
    if ss < df.shape[0]: df = df.sample(n=ss,random_state=88)

    #get combinations
    combos = combinations(df.columns,2)

    #Loop over combinations and plot
    for pair in combos:
        #Add a chart - lmplot generates facetgrid
        sns.lmplot(data=df,x=pair[0],y=pair[1],line_kws={'color':'red'})
        plt.title(f'{pair[1]} vs {pair[0]}')
        plt.show()
    
    return None

def plot_cat_and_continuous(df,**kwargs):
    '''
    Takes dataframe and plots all categorical variables vs all continuous variables. \
    Subset of categorical columns and continuous columns can be passed.  If not specified,\
    Assumes all objects and boolean columns to be categories and all numeric columns to be continuous
    **DOES NOT HANDLE DATETIMES**
    
    OUTPUTS: Charts
    RETURNS: None
    INPUTS:
      (R)            df: Pandas Dataframe containing categorical and continous columns
      (O)   sample_size: number of rows to use when plotting.  Default 50_000
      (O)      cat_cols: List of categorical columns to be plotted. Default: object and boolean dtypes
      (O)     cont_cols: List of continuous columns to be plotted. Default: numeric dtypes
    '''
    #pull out sample size
    ss = kwargs.get('sample_size',50_000) # Default 50k
    #If sample size is smaller than df rows, pull out a sample of the data
    if ss < df.shape[0]: df = df.sample(n=ss,random_state=88)

    #Get categorical and continuous features
    cats = kwargs.get('cat_cols',df.select_dtypes(include=['bool','object']))
    conts = kwargs.get('cont_cols',df.select_dtypes(include='number'))
    
    #create pairs
    pairs = product(cats,conts)
    
    #Loop over pairs to plot
    for pair in pairs:
        #Cats will be first in the pair
        cat= pair[0]
        cont= pair[1]
        #Plot 3 charts (1x3)
        fig, ax = plt.subplots(1,3,figsize=(12,4),sharey=True)
        fig.suptitle(f'{cont} vs. {cat}')
        
        #First Chart
        plt.subplot(1,3,1)
        sns.boxplot(data=df,x=cat,y=cont)
        #Format y axis
        if df[cont].max() > 1_000_000: yticks_mm(ax[0])
        elif df[cont].max() > 2500: yticks_k(ax[0])
        #Other Charts - shared y axis
        plt.subplot(1,3,2)
        sns.violinplot(data=df,x=cat,y=cont)
        plt.subplot(1,3,3)
        sns.stripplot(data=df,x=cat,y=cont)
        plt.tight_layout()
    
    return None
##########################
##########################

###### Model Support  ######
def select_kbest(X,y,k):
    '''
    Uses sklearn.feature_selection.SelectKBest to select top k features.
    
    Returns: List corresp
    Inputs: 
      (R) X: Pandas Dataframe of features and values
      (R) y: target variable
      (R) k: number of features to select
    '''
    #Create feature selector & fit
    f_selector = SelectKBest(f_regression,k=k).fit(X,y)
    # Boolean mask of which columns are selected
    f_mask = f_selector.get_support()
    #get list of top features
    k_features = X.columns[f_mask].tolist()
    #return features as list
    return k_features
##########################
##########################
   
###### Pretty Print Stats  ######
def stats_result(p,null_h,**kwargs):
    """
    Compares p value to alpha and outputs whether or not the null hypothesis
    is rejected or if it failed to be rejected.
    DOES NOT HANDLE 1-TAILED T TESTS
    
    Required inputs:  p, null_h (str)
    Optional inputs: alpha (default = .05), chi2, r, t
    
    """
    #Get alpha value - Default to .05 if not provided
    alpha=kwargs.get('alpha',.05)
    #get any additional statistical values passed (for printing)
    t=kwargs.get('t',None)
    r=kwargs.get('r',None)
    chi2=kwargs.get('chi2',None)
    corr=kwargs.get('corr',None)
    
    #Print null hypothesis
    print(f'\n\033[1mH\u2080:\033[0m {null_h}')
    #Test p value and print result
    if p < alpha: print(f"\033[1mWe reject the null hypothesis\033[0m, p = {p} | α = {alpha}")
    else: print(f"We failed to reject the null hypothesis, p = {p} | α = {alpha}")
    #Print any additional values for reference
    if 't' in kwargs: print(f'  t: {t}')
    if 'r' in kwargs: print(f'  r: {r}')
    if 'chi2' in kwargs: print(f'  chi2: {chi2}')
    if 'corr' in kwargs: print(f'  corr: {corr}')

    return None

#Individual calculations
def residuals(y,yhat):
    return (y - yhat)

def sse(y,yhat):
    return sum(residuals(y,yhat)**2)

def rmse(y,yhat):
    return sqrt(mean_squared_error(y,yhat))

def ess(y,yhat):
    return sum((yhat-y.mean())**2)

def tss(y,yhat):
    return ess(y,yhat) + sse(y,yhat)


#need more than what they provided
def plot_residuals(x,y,yhat,title='Residual'):
    '''
    Creates a scatterplot showing residual vs independent variable
    Outputs: AxesSubplot (scatterplot)
    Returns: None
    Input:
      (R)     x: independent variable (pd.Series or np.array)
      (R)     y: actual values (pd.Series or np.array)
      (R)  yhat: predicted values (pd.Series or np.array)
      (O) title: title of chart (string).  Default: 'Residual'
    '''
    #get residual
    y=residuals(y,yhat)
    #plot
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    ax = sns.scatterplot(x=x, y=y,alpha=.05)
    #Format y axis
    if y.max() > 1_000_000: yticks_mm(ax)
    elif y.max() > 2500: yticks_k(ax)
    #Add actual line (y=0)
    plt.axhline(y=0,c='r')
    #Add text
    plt.xlabel('x')
    plt.ylabel('Residual')
    plt.title(title)
    return None

def regression_errors(y,yhat):
    '''
    Takes in actual and predicted values. Returns dataframe of regression performance statistics.
    
    Returns: Pandas DataFrame
    Input:
      (R)    y: actual values (pd.Series or np.array)
      (R) yhat: predicted values (pd.Series or np.array)
    
    '''
    #set index name for dataframe
    if isinstance(yhat,pd.Series): ind=yhat.name
    else: ind='yhat'
    #Create DataFrame with performance stats as columns
    df = pd.DataFrame({
        'sse': [sse(y,yhat)],
        'ess': [ess(y,yhat)],
        'tss': [tss(y,yhat)],
        'mse': [mean_squared_error(y,yhat)],
        'rmse': [rmse(y,yhat)],
        },index=[ind])
    return df

def get_reg_model_stats(y,yhat):
    '''
    Takes in actual and predicted values. Returns dataframe of regression model performance statistics.
    
    Returns: Pandas DataFrame
    Input:
      (R)    y: actual values (pd.Series)
      (R) yhat: predicted values (pd.Series)
    '''
    #Create DataFrame with performance stats as columns, model name as index
    df = pd.DataFrame({
        'sse': [sse(y,yhat)],
        'mse': [mean_squared_error(y,yhat)],
        'rmse': [rmse(y,yhat)],
        },index=[yhat.name])
    return df

def baseline_mean_errors(y):
    '''
    Takes in actual values. Returns dataframe of regression performance statistics.
    
    Returns: Pandas DataFrame
    Input:
      (R)    y: actual values (pd.Series or np.array)
    '''
    #Create series of yhat_baseline
    if isinstance(y,pd.Series): ind = y.index
    else: ind = range(len(y))
    yhat_b = pd.Series(y.mean(),index=ind)
    #Create DataFrame with performance stats as columns
    df = pd.DataFrame({
        'sse': [sse(y,yhat_b)],
        'mse': [mean_squared_error(y,yhat_b)],
        'rmse': [rmse(y,yhat_b)],
        },index=['yhat_baseline'])
    return df

def better_than_base(y,yhat):
    '''
    Takes in actual and predicted values. Returns True/False on if \
    the model performed better than the dataframe based on rmse.
    
    Returns: Boolean
    Input:
      (R)    y: actual values (pd.Series or np.array)
      (R) yhat: predicted values (pd.Series or np.array)
    
    '''
    #Determine if series or array - use info to create baseline series
    if isinstance(y,pd.Series): ind = y.index
    else: ind = range(len(y))
    yhat_b = pd.Series(y.mean(),index=ind)
    #Get RMSE for model and baseline
    rmse_base = rmse(y, yhat_b)
    rmse_mod = rmse(y,yhat)
    return rmse_mod < rmse_base


