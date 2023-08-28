# Imports
from main import bot
from visualization import plot_bot_results
from get_data import get_data_from_mt5
from plotly.offline import plot

# Configuration
LOGIN = "51545562"
PASSWORD = "zop7gsit"
SERVER = "Alpari-MT5-Demo"
initialize = [LOGIN, PASSWORD, SERVER]



## Trading Bot
bot_model='advance_quant' #  quant or advance_quant

currency, TimeFrame = 'EURUSD', '15m' 

info, outputs = bot(bot_model, initialize, currency=currency, TimeFrame=TimeFrame,
                    risk=100, max_position_time='2*TimeFrame', max_pending_time='0.1*max_time',  
                    use_haiken_ashi=False, hyper_tune=False, plot_results=False, force_to_train=False)


df = get_data_from_mt5(initialize, currency, TimeFrame)
fig = plot_bot_results(df.iloc[-50:], info)
plot(fig)


