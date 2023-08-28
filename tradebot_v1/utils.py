## Import Libraries
import pandas as pd
import numpy as np
from vmdpy import VMD
import statsmodels.api as sm
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans , AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lagmat import lagmat
import warnings
warnings.filterwarnings('ignore')




def standardization(Data):
    """
    Standardize data to have zero mean and unit variance.
    """

    transformer = StandardScaler()
    standardized_data = transformer.fit_transform(Data)
    return standardized_data, transformer
    
    
    

def VMDdecomposition(Target, Numimf, plotResults):
    """
    Decompose data via Variational Mode Decomposition (VMD).

    """
    
    if not (len(Target) % 2 == 0) : Target=Target[1:]
    
    # Call the VMD function to decompose the data
    imfs, _, _ = VMD(Target, 5000, 0, Numimf, 0, 1, 5e-6)

    # Convert the IMFs to a Pandas DataFrame for easier manipulation
    imfs = pd.DataFrame(np.transpose(imfs))

    return imfs

## Fourier Transform
def fourier(data, forecast_horizen):
    
    """
    Perform long-term forecasting using Fourier transform.
    
    Parameters:
        data: A 1D numpy array containing the input data.
        forecast_horizon: An integer representing the number of time steps to forecast.
    
    Returns:
        A 1D numpy array containing the forecasted values.
    
    The function first estimates a linear trend in the input data using ordinary least squares (OLS). The trend is then subtracted from the data to obtain a detrended series. The function then performs Fourier transform on the detrended series and extracts the `n_harm` most significant harmonics. The harmonics are then used to perform Fourier extrapolation to generate the forecasted values.
    
    The forecasted values are then added back to the linear trend to obtain the final forecast.
    
    Examples:
        # Perform long-term forecasting using Fourier transform
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        forecast_horizon = 5
        forecast = Forecast(data, forecast_horizon)
    """
    
    # Linear Trend Estimation Using OLS
    n = data.size
    t = np.arange(0, n + forecast_horizen)

    p= np.polyfit(t[:n], data, deg=1)
    trend=np.polyval(p,t)
    detrend=data-trend[:n]
    
    # Fourier Transform
    n_harm=10
    x_freqdom = np.fft.fft(detrend)  
    f = np.fft.fftfreq(n)              
    indexes = np.array(range(0,n))
    sorted(indexes,key = lambda i: np.absolute(f[i]))
    
    # Fourier Extrapolation
    prediction = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        prediction += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    
    return prediction+trend

## The Hodrick-Prescott (HP) filter

def hp_filter(signal, lamb=80):
    # Convert signal to a numpy array
    signal = np.asarray(signal)
    
    # Apply the hpfilter
    cycle, trend = sm.tsa.filters.hpfilter(signal, lamb)
    
    # Return the filtered components as a tuple
    return trend




def label_returns(df):
    """
    Given a DataFrame `df`, this function labels the returns as 'sell', 'neutral', 'buy' based on their percentile rank
    within the distribution of returns.

    the function assigns labels as follows:
    - Sell or 0: returns between the 0th and 35th percentile
    - Neutral or 1: returns between the 35th and 65th percentile
    - Buy or 2: returns between the 65th and 100th percentile
    
    The function returns a new DataFrame with columns for the labels and the returns.
    """

    percentiles = df['diff'].quantile([0.35, 0.65])

    def label_return(ret):
        if ret <= percentiles[0.35]:
            return 0
        elif ret <= percentiles[0.65]:
            return 1
        else:
            return 2

    df = df.assign(label=df['diff'].apply(label_return))

    return df[['label', 'diff']]


def FindBestLags(Inputs, Targets, lagmax=200, nlags=10):
    """
    Selects the best lags for each input feature based on their correlation with the target variable.

    Parameters:
    -----------
    Inputs : array-like, shape (n_samples, n_features)
        The input features.
    Targets : array-like, shape (n_samples, )
        The target variable
    lagmax : int, optional (default=200)
        The maximum lag to consider for the input features.
    nlags : int, optional (default=10)
        The number of lags to select for each input feature.

    Returns:
    --------
    Best_Inputs : array-like, shape (n_samples, nlags, n_features)
        The lagged input features selected based on the best lags.
        
    Forecast_Inputs : array-like, shape (2, nlags, n_features)
        The lagged input features for the last two time steps, used for forecasting.

    """

    nFeature = np.shape(Inputs)[1]
    nSample = np.shape(Inputs)[0]
    BestLags = []


    # Find the best lags for each input feature
    for feature_id in range(nFeature):
        # Compute the lags for the current input feature
        lags = lagmat(Inputs[:, feature_id], lags=list(np.arange(1, lagmax)))
        # Convert the lag matrix to a pandas DataFrame and exclude the first `lagmax` rows (which contain NaNs)
        lags = pd.DataFrame(lags[lagmax+1:])
        # Extract the target variable for the selected lags
        present = Targets[lagmax+1:]

        # # Use correlation to select lags
        corr = [np.abs(np.corrcoef(lags.iloc[:, i], present)[0, 1]) for i in range(lags.shape[1])]
        
        # Calculate the distance correlation between all lags and the target variable

        LagsId = np.argsort(corr)[-nlags:]
        

        # Store the lag indices for the current input feature
        BestLags.append(np.sort(LagsId) + np.repeat(1, len(LagsId)))
        
        
    # Construct the `Best_Inputs` array using the selected lags
    Best_Inputs = np.zeros((nSample, nlags, nFeature))
    for feature_id in range(nFeature): 
        Best_Inputs[:, :, feature_id] = lagmat(Inputs[:, feature_id], lags=BestLags[feature_id])
    # Exclude the first `lagmax` rows (which contain NaNs) from the `Best_Inputs` and `Targets` arrays
    Best_Inputs = Best_Inputs[lagmax+1:, :, :]
    Targets = Targets[lagmax+1:]

    # Construct the `Forecast_Inputs` array for the last two time steps
    ForecastInputs = np.zeros((nSample, nlags, nFeature))
    for feature_id in range(nFeature): 
        ForecastInputs[:,:,feature_id] = lagmat(Inputs[:,feature_id], lags=BestLags[feature_id]-np.repeat(1,nlags))
    Forecast_Inputs = ForecastInputs[-1:,:,:]

    return Best_Inputs, Forecast_Inputs



def find_heikin_ashi_candlestick(df):
    
    """
    Calculates Heikin Ashi candles for a given DataFrame of candlestick data.

    Heikin Ashi candles are a type of financial chart used to represent the price movement of an asset. They are calculated using a modified formula that takes into account the values of the previous candlestick, and can help to filter out noise and identify trends.

    The function takes a DataFrame of candlestick data as input, and returns a DataFrame containing the Heikin Ashi candles.

    Parameters:
        df (pd.DataFrame): A DataFrame containing the candlestick data. The DataFrame should have columns for the open, high, low, and close prices of the asset, with each row representing a single time interval.

    Returns:
        df_HA (pd.DataFrame): A DataFrame containing the Heikin Ashi candles. The DataFrame has the same index as the input DataFrame, and columns for the open, high, low, and close prices of the Heikin Ashi candles.

    Example:
        data = {'Open': [10, 12, 11, 13], 'High': [15, 14, 13, 14], 'Low': [9, 11, 10, 10], 'Close': [14, 13, 12, 11]}
        df = pd.DataFrame(data)
        ha_candles = find_heikin_ashi_candlestick(df)
        print(ha_candles)
    """
    df=df.dropna()
    df_HA = df.copy()
    df_HA['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

    for i in range(len(df)):
        if i == 0:
            df_HA['Open'][i] = (df['Open'][i] + df['Close'][i]) / 2
        else:
            df_HA['Open'][i] = (df_HA['Open'][i-1] + df_HA['Close'][i-1]) / 2

    df_HA['High'] = df[['Open', 'Close', 'High']].max(axis=1)
    df_HA['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)

    df_HA['Mean'] = np.mean(pd.concat((df_HA['Open'], df_HA['Low'], df_HA['High'], df_HA['Close']), axis=1), axis=1)
    df_HA['diff']=df_HA['Close']-df_HA['Open']
    
    return df_HA




