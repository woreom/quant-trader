import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd


def plot_bot_results(df, trade_info):
    
    fig = make_subplots(rows=1, cols=1)
    
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],  name='price'),row=1, col=1)
    
    
    if trade_info['Forecast'] != 'Neutral':
        
        EntryPoint=trade_info['EntryPoint']
        TakeProfit=trade_info['TakeProfit']
        StepLoss=trade_info['StepLoss']
        Forecast=trade_info['Forecast']
        

        t=pd.date_range(start=df.index[-1],freq=df.index[-1]-df.index[-2],periods=2)
        
        
        fig.add_trace(go.Scatter(x=[t[0], t[-1]],y=np.repeat(EntryPoint,2), 
                                 name='Entry Point',line=dict(color="#0000ff", width=1.5)),row=1, col=1)
        fig.add_trace(go.Scatter(x=[t[0], t[-1]],y=np.repeat(StepLoss,2), 
                                 name='Step Loss',line=dict(color="#ff0000", width=1.5)),row=1, col=1)
        fig.add_trace(go.Scatter(x=[t[0], t[-1]],y=np.repeat(TakeProfit,2), 
                                 name='Take Profit',line=dict(color="#008000", width=1.5)),row=1, col=1)  
        fig.add_trace(go.Scatter(x=[t[0], t[-1]],y=[EntryPoint, Forecast], 
                                 name='Forecast',line=dict(color="#9933ff", width=1.5)),row=1, col=1)  
    else:
        
        fig.add_annotation(dict(font=dict(color='black',size=15),
                                        x=0,
                                        y=-0.12,
                                        showarrow=False,
                                        text="The market is currently within a trading range, and it may be advisable to avoid trading at this time.",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
        

    

    layout = dict(
            title="Results for {}".format(df['info'].iloc[0]),
            xaxis_rangebreaks=[dict(bounds=["sat", "mon"])],
            xaxis_rangeslider_visible=False,
            showlegend=True)
    fig.update_layout(layout)
    
    return fig

