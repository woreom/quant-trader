from utils import (fourier, standardization, VMDdecomposition, FindBestLags,
                   find_heikin_ashi_candlestick, label_returns)
from get_data import (correct_candle, get_country_index_from_mt5, get_data_from_mt5)
from ml_model import lgb_regressor, mlp_regressor
from deep_model import transformer_model
from strategy import (Strategy, Control_Position)

import numpy as np
import pandas as pd


import joblib

def bot(bot_model, initialize, currency='EURUSD', TimeFrame='15m',
        risk=100, max_position_time='3*TimeFrame', max_pending_time='0.1*max_time',
        use_haiken_ashi=False, hyper_tune=False, plot_results=False, force_to_train=False):
    """
    Run a trading bot based on the specified model.

    Parameters:
    - bot_model (str): The type of trading bot model to use. Should be 'quant' or 'advance_quant'.
    - initialize (function): The initialization function for the trading bot.
    - currency (str): The currency pair to trade. Default is 'EURUSD'.
    - TimeFrame (str): The time frame for the trading data. Default is '15m'.
    - risk (float): The risk amount for each trade. Default is 100.
    - max_position_time (str): The maximum time allowed for a position to be open. Default is '3*TimeFrame'.
    - max_pending_time (str): The maximum time allowed for a pending order to be active. Default is '0.1*max_time'.
    - use_haiken_ashi (bool): Whether to use Haiken Ashi candles for the trading bot. Default is False.
    - hyper_tune (bool): Whether to perform hyperparameter tuning for the trading bot. Default is False.
    - plot_results (bool): Whether to plot the trading results. Default is False.
    - force_to_train (bool): Whether to force the trading bot to train even if pre-training data is available. Default is False.

    Returns:
    - info (dict): Information about the trading bot run.
    - quant_outputs (dict): Outputs from the trading bot.

    Raises:
    - ValueError: If an invalid bot model is specified.
    """

    if bot_model == 'quant':
        info, quant_outputs = quant_bot(initialize, currency=currency, TimeFrame=TimeFrame,
                                        risk=risk, max_position_time=max_position_time,
                                        max_pending_time=max_pending_time, 
                                        hyper_tune=hyper_tune, plot_results=plot_results)

    elif bot_model == 'advance_quant':
        if force_to_train:
            info, quant_outputs = advance_quant_bot(initialize, currency=currency, TimeFrame=TimeFrame,
                                                    use_pre_train=False, use_haiken_ashi=use_haiken_ashi,
                                                    hyper_tune=hyper_tune, plot_results=plot_results)
        else:
            try:
                info, quant_outputs = advance_quant_bot(initialize, currency=currency, TimeFrame=TimeFrame,
                                                        use_pre_train=True, use_haiken_ashi=use_haiken_ashi,
                                                        hyper_tune=hyper_tune, plot_results=plot_results)
            except:
                info, quant_outputs = advance_quant_bot(initialize, currency=currency, TimeFrame=TimeFrame,
                                                        use_pre_train=False, use_haiken_ashi=use_haiken_ashi,
                                                        hyper_tune=hyper_tune, plot_results=plot_results)
    else:
        raise ValueError("Invalid bot model, should be quant or advance_quant")

    return info, quant_outputs


def quant_bot(initialize, currency='EURUSD', TimeFrame='15m',
              risk=100, max_position_time='3*timeframe', max_pending_time='0.1*maxtime',
             hyper_tune=False, plot_results=False):
    

    Numimf, n_sample, lagmax, nlags, n_trials=32, int(15e3), 200, 8, 50

    
    
    df=correct_candle(initialize, currency, TimeFrame)
    

    params = joblib.load('params/trade_params.pkl')

    t = df.index[-1]-df.index[-2] # TimeFrame
    params['MaxTime']=int(max_position_time.split(sep='*')[0])*int(t.total_seconds()/60) 

    max_pending_time=int(params['MaxTime']*float(max_pending_time.split(sep='*')[0]))*60
    max_position_time=params['MaxTime']*60

    data = df['Mean'].to_numpy()
    n = len(df)

    fft_forecast = fourier(data, int(0.1 * n))

    y = data - fft_forecast[:n]
    y_diff = np.diff(y)

    Target = y_diff[-n_sample:]

    imfs = VMDdecomposition(Target, Numimf, plot_results)

    X_scaled,_ = standardization(imfs.to_numpy())

    target, scaler = standardization(Target.reshape(-1,1))

    Inputs, X_forecast = FindBestLags(X_scaled, 
                                    target.reshape(-1,),
                                    lagmax=lagmax,
                                    nlags=nlags)
                                    
    nSample, seq_length, n_features = Inputs.shape

    Inputs = Inputs.reshape(nSample, seq_length * n_features)
    X_forecast = X_forecast.reshape(1, seq_length * n_features)

    Targets = target[-len(Inputs):]

    forecast_diff, lgb, rmse = lgb_regressor(Inputs, Targets, X_forecast,
                                            save_name='quant',
                                            hyper_tune=hyper_tune, 
                                            plotResults=plot_results, 
                                            n_trials=n_trials)
    outputs=lgb.predict(Inputs)
    outputs= scaler.inverse_transform(outputs.reshape(-1, 1))
    y_pred =[]
    dd=y[-len(outputs)-1:-1]
    for i in range(len(outputs)):
        y_pred.append(outputs[i]+dd[i])
        
    outputs=np.array(y_pred).reshape(-1, 1)
    temp=np.array(fft_forecast[:n]).reshape(-1, 1)
    outputs=outputs+temp[-len(outputs):]
    
    outputs=pd.DataFrame({'real': df['Mean'].values[-len(outputs):],
                          'quant': outputs.reshape(-1,)}, index=df.index[-len(outputs):])
    
    outputs['error (%)']=100*(outputs['real']-outputs['quant']).abs()/outputs['real']

    forecast_diff = scaler.inverse_transform(np.reshape(forecast_diff, (1, -1)))[0][0]

    Forecast = y[-1] + forecast_diff + fft_forecast[n]

    info={'Forecast': Forecast,
          'EntryPoint': df['Mean'].iloc[-1],
          'EntryTime': df.index[-1]}

    info = Strategy(df, info, params)

    print('--------------Results-------------')
    print(info)
    print('----------------------------------')

    if info['Action']!='Hold':
        # Trade
        Control_Position(initialize, info, max_pending_time, max_position_time)
    
    return info, outputs



def quant_classifier(initialize, currency='EURUSD', TimeFrame='15m',
            use_pre_train=True, use_haiken_ashi=True, hyper_tune=False, plot_results=False):
    

    Numimf, max_nsample, lagmax, nlags, n_trials=32, int(20e3), 200, 8, 50
    
    if len(currency)==3:
        df=get_country_index_from_mt5(initialize, currency, TimeFrame)
    else:
        
        try:
            df=correct_candle(initialize, currency, TimeFrame)
            
        except:
            df=get_data_from_mt5(initialize, currency, TimeFrame)
    
    if use_haiken_ashi:
        save_name=f"{df['info'].iloc[0]}_use_haiken_ashi"
    else:
        save_name=df['info'].iloc[0]

    n_sample = np.min([len(df), max_nsample]) 
    df=df.iloc[-n_sample:]
    
    if not (len(df) % 2 == 0) : df=df[1:]
    
    if use_haiken_ashi:
        y=label_returns(find_heikin_ashi_candlestick(df))
    else:
        y=label_returns(df)
    
    X=df['diff'].to_numpy()
    
    X=X.reshape(-1, 1)
    NumSamples = X.shape[0]
    NumFeature = X.shape[1]
    Inputs = np.zeros((NumSamples, NumFeature, Numimf))
    for i in range(NumFeature):
        Inputs[:, i, :] = VMDdecomposition(X[:, i], Numimf=Numimf, plotResults=plot_results)
    Inputs = Inputs.reshape(NumSamples, NumFeature * Numimf)

    X_scaled,_ = standardization(Inputs)
    
    inputs, inputs_forecast = FindBestLags(X_scaled,
                                                 y['diff'].to_numpy().reshape(-1,),
                                                 lagmax=lagmax,
                                                 nlags=nlags)
    labels = y['label'].iloc[-len(inputs):].to_numpy()

    forecast, prob, net, acc, encoding, label_indices=transformer_model(inputs, labels, inputs_forecast,
                                                               save_name=save_name, use_pre_train=use_pre_train, 
                                                               hyper_tune=hyper_tune, plotResults=plot_results, n_trials=n_trials)
    

    outputs=pd.DataFrame({'quant classifier': label_indices, 'label': labels }, index=y.index[-len(inputs):])
    outputs.replace(0, 'sell', inplace=True)
    outputs.replace(1, 'neutral', inplace=True)
    outputs.replace(2, 'buy', inplace=True)
    
    return  forecast, prob, net, acc, encoding, outputs

def advance_quant_bot(initialize, currency='EURUSD', TimeFrame='15m',
              risk=100, max_position_time='3*timeframe', max_pending_time='0.1*maxtime',
            use_pre_train=True, use_haiken_ashi=False, hyper_tune=False, plot_results=False):
    

    forecast1, prob, net, acc, encoding, classifier_outputs=quant_classifier(initialize, currency=currency, TimeFrame=TimeFrame,
                use_pre_train=use_pre_train, use_haiken_ashi=use_haiken_ashi, hyper_tune=hyper_tune, plot_results=plot_results)

    if forecast1!=1:
        
        df=correct_candle(initialize, currency, TimeFrame)
        targets=np.diff(df['Mean'].values)[-len(encoding)+1:]
        y_scaled,scaler=standardization(targets.reshape(-1, 1))
        encoding = np.nan_to_num(encoding)
        forecast, mlp, rmse=mlp_regressor(encoding[:-1,:], y_scaled, encoding[-1,:], f'{currency}_{TimeFrame}', 
                      hyper_tune=hyper_tune, plotResults=plot_results, n_trials=30)

        forecast_diff=scaler.inverse_transform(forecast.reshape(1, -1))[0][0]
        y_pred=scaler.inverse_transform(mlp.predict(encoding[:-1,:]).reshape(-1, 1)).reshape(-1, )

        Forecast=forecast_diff+df['Mean'].iloc[-1]

        outputs=[]
        dd=df['Mean'].values[-len(y_pred)-1:-1]
        for i in range(len(y_pred)):
            outputs.append(y_pred[i]+dd[i])

        regressor_outputs=pd.DataFrame({'real': df['Mean'].values[-len(y_pred):],
                              'advance-quant': outputs}, index=df.index[-len(y_pred):])
        
        outputs=pd.concat([classifier_outputs, regressor_outputs], axis=1).dropna()
        
        info={'Forecast': Forecast,
              'EntryPoint': df['Mean'].iloc[-1],
              'EntryTime': df.index[-1]}
        
        params = joblib.load('params/trade_params.pkl')
        
        t = df.index[-1]-df.index[-2] # TimeFrame
        params['MaxTime']=int(max_position_time.split(sep='*')[0])*int(t.total_seconds()/60) 

        max_pending_time=int(params['MaxTime']*float(max_pending_time.split(sep='*')[0]))*60
        max_position_time=params['MaxTime']*60

        info = Strategy(df, info, params)

        print('--------------Results-------------')
        print(info)
        print('----------------------------------')

        if info['Action']!='Hold':
            # Trade
            Control_Position(initialize, info, max_pending_time, max_position_time)

    else:
        info={'Forecast': 'Neutral',
              'EntryPoint':None,
              'EntryTime':None,
              'confidence':prob}
        
        print("The market is currently within a trading range, and it may be advisable to avoid trading at this time.")
        outputs=classifier_outputs

    return info, outputs









    



    



