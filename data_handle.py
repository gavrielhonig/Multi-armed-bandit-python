import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


#Read Raw file
df = pd.read_csv('raw_data_big.csv')


## Handling multiple rows per ID
df = df.sort_values('installation_date').reset_index(drop=True)
df = df.set_index('id_for_vendor')
df['user_cnt'] = df.index.value_counts()
df = df.groupby('id_for_vendor').first()


#Data handling
date_installed = df['installation_date'].str.split('T', expand=True)[0]
df['installation_date'] = pd.to_datetime(date_installed)
df['continent'] = df['timezone'].str.split("/", expand=True)[0]
df['languange'] = df['preferred_language'].str.split("-", expand=True)[0]
date_subscribed = df['subscription_date'].str.split('T', expand=True)[0]
df['subscription_date'] = pd.to_datetime(date_subscribed)
first_ses = df['first_session'].str.split('T', expand=True)[0]
df['first_session'] = pd.to_datetime(first_ses)
df['delta'] = (df['subscription_date'] - df['installation_date']).apply(lambda delta: delta.days)
df['daily_sessions'] = df['num_sessions'] / df['delta']
df['daily_sessions_bef_sub'] = df['num_sessions_until_subscription'] / df['delta']
df['had_trial'] = df['had_trial'].map({'TRUE':1,'FALSE':0})
df['device_kind'] = df['device_kind'].str[12:]

#Yev Gdp
df['gdpCountry'] = df['app_store_country']
df['gdpCountry']=df['gdpCountry'].replace({'AE':37749,
  'AR':9887,'AT':50002,'AU':53825,'BE':45175,'BR':8796,'CA':46212,'CH':83716,
  'CL':15399,'CZ':23213,'DE':46563,'DK':57975,'EG':3046,'ES':29961,'FR':41760,
  'GB':41030,'HU':17463,'ID':4163,'IL':42823,'IN':2171,'IT':32946,'JP':40846,
  'KH':1620,'KZ':9139,'MX':10118,
  'MY':11136,'NL':52367,'NO':77975,'PH':3294,'PK':1388,'PL':14901,'PT':23408,
  'RO':12301,'RU':11162,'SA':22865,'SE':54608,'TH':7274,'TR':9370,'TW':24827,'UA':3592,'US':65111,'VN':2740,'ZA':11300,'CO':6500})
df.gdpCountry= pd.to_numeric(df.gdpCountry, errors='coerce')
df['gdpCountry']=df['gdpCountry'].fillna(11335)
df['gdpCountry'] = pd.cut(df.gdpCountry, bins=[0,29960,50000,150000], labels=[0,1,2])

#End Yev Gdp

device_map = {'IPhone7': 0, 'IPhone7Plus': 0, 'IPhone8Plus': 0, 'IPhone6S': 0, 'IPhoneSE': 0,
              'IPhone8': 0, 'IPhone6SPlus': 0, 'IPadAir1G': 0, 'IPhone5S': 0, 'IPadMini3G': 0,
              'IPhone6': 0, 'IPadAir2G': 0, 'IPhone6Plus': 0, 'IPadMini2G': 0,
              'IPhoneXR': 1, 'IPhoneX': 1, 'IPhoneXS': 1, 'IPhoneXSMax': 1, 'IPadMini5G': 1,
              'IPhone11ProMax': 1, 'IPad6G': 1, 'IPhone11': 1, 'IPhone11Pro': 1, 'IPadPro12_9G3': 1,
              'IPad5G': 1, 'IPadPro10_5': 1, 'IPadPro9_7': 1, 'IPadPro12_9': 1, 'IPadPro11': 1,
              'IPadPro2G12_9': 1, 'IPadMini4G': 1, 'IPad7G': 1}
df['device_kind'] = df['device_kind'].map(device_map)
df['app_store_country'] = np.where(df['app_store_country'] == 'US', 'US'
                                   , np.where(df['app_store_country'] == 'VN', 'VN'
                                              , np.where(df['app_store_country'] == 'DE', 'DE'
                                                         , np.where(df['app_store_country'] == 'RU', 'RU'
                                                                    , np.where(df['app_store_country'] == 'GB','GB'
                                                                               , np.where(df['app_store_country'] == 'IT', 'IT'
                                                                                          , np.where(df['app_store_country'] == 'FR', 'FR'
                                                                                                           , np.where(df['app_store_country'] == 'BR', 'BR', 'ROW'))))))))
df['languange'] = np.where(df['languange'] == 'en', 'en'
                           , np.where(df['languange'] == 'vi', 'vi'
                                      , np.where(df['languange'] == 'de', 'de'
                                                 , np.where(df['languange'] == 'ru', 'ru'
                                                            , np.where(df['languange'] == 'es', 'es', 'ROW')))))
variant_map = {'Baseline': 1, 'GroupA': 2, 'GroupB': 3, 'GroupC': 4, 'GroupD': 5}
df['variant'] = df['variant'].map(variant_map)



df.drop(['full_refund_date','timezone','is_subscriber_on_popup'
            ,'preferred_language','installation_date','revenue_so_far','num_sessions','delta'
            ,'subscription_date','first_session','languange','continent','seen_popup', 'pressed_button'
            ,'first_auto_renewal_disabling_date','num_export_48h', 'auto_renewal_disabling_date'
            ,'last_subscription_renewal_date','had_trial','trial_length','subscription_duration'
            ,'seen_popup_at','pressed_button_at','last_session_48h'],axis=1,inplace=True)


df = df.dropna(subset=['last_network', 'app_store_country'])
df = df.fillna(0)
#df = df.reset_index(drop=True)
columns = ['variant','is_subscriber','device_kind', 'last_network', 'app_store_country',
           'num_sessions_48h', 'num_sessions_until_subscription',
           'num_successful_export_48h', 'num_sub_screen_presented_48h',
           'avg_session_duration_48h', 'user_cnt','daily_sessions', 'daily_sessions_bef_sub']

df = df[columns]
df = pd.get_dummies(df, columns=['app_store_country','last_network'])


df = df[0:10000]
df.to_csv('work.csv',index=False)
print("DF saved!")

