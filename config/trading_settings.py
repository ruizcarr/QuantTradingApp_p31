from datetime import date
from datetime import timedelta

# Import Settings
from config.utils import retrieve_training_settings
settings =retrieve_training_settings()

#from config.settings import settings


# Update Settings to Trading Settings
settings['trading_app_only']=True
settings['start'] = '2019-01-01' # '2018-01-01'
settings['end'] =(date.today() + timedelta(days=1)).isoformat()
settings['startcash'] = 46000 #56500#210000#35000 #30650 #27800 #54300 #52300 # #EUR
settings['verbose'] = True
settings['qstats'] = True
settings['do_BT']= True
settings['offline'] = False
#settings['tickers_bounds'] ={'ES=F': (-0.0, 0.5), 'NQ=F': (-0, 0.5), 'GC=F': (0.00, 0.5), 'CL=F': (0, 0.2), 'EURUSD=X': (-0.00, 0.00), 'cash': (0, 0.05)}
settings['add_cash'] = True
