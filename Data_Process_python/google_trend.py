import pandas as pd                        
from pytrends.request import TrendReq
from dateutil.relativedelta import relativedelta
from datetime import datetime
import time
import concurrent.futures
import os

df_trend = pd.read_csv('/Users/nianyun/Documents/Pictet_IPO_Case/Data_clean/IPO_return.csv')
df_trend = df_trend.loc[ (df_trend['Issuer'].notnull()) & (df_trend['Trade_Date'].notnull())]
df_trend['Issuer'] = df_trend['Issuer'].str.rstrip()

company_date = dict(zip(df_trend.Issuer, df_trend.Trade_Date))

#use company name, to get weekly data, use the 6 month before and after IPO date 
# company_date
# d1 = company_date['Gemplus']
# start = datetime.strptime(d1, '%Y-%m-%d') + relativedelta(months=6)
# end = datetime.strptime(d1, '%Y-%m-%d') - relativedelta(months=6)
# datetime.strptime('2003-01-01', '%Y-%m-%d') > datetime.strptime('2004-01-01', '%Y-%m-%d')



def get_trend(search, date , month_before, month_after):
    pytrends = TrendReq()
    kw_list = [search]
    start = datetime.strptime(date, '%Y-%m-%d') - relativedelta(months=month_before)
    end = datetime.strptime(date, '%Y-%m-%d') + relativedelta(months=month_after)
    if start < datetime.strptime('2004-01-01', '%Y-%m-%d'):
        return None 
    else:
        timeframe = ' '.join([start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')])
        pytrends.build_payload(kw_list, timeframe=timeframe)
        df = pytrends.interest_over_time()
        df = df.reset_index().rename(columns={ search : 'SVI'})
        df['Issuer'] = search
        if len(df) == 0:
            return None 
        else:
            return df


def all_trend(company_date, nchunk):
    df_list = []
    for company, date in company_date.items():
        try:
            df_google = get_trend(search = company, date = date , month_before = 3, month_after = 9)
        except:
            time.sleep(30)
            df_google = get_trend(search = company, date = date , month_before = 3, month_after = 9)
        if df_google is not None:
            df_list.append(df_google)
            df = pd.concat(df_list)
    try:
        df.to_csv(f'../Data/Google_trend/trend_chunk_{nchunk}.csv', index = False)
        print('data frame get')
        return df
    except:
        print('no data found')
        return None

def chunks(data, size):
    from itertools import islice
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in islice(it, size)}


list_chunk = [item for item in chunks(company_date, 5)]
#rate limit in google trend, start from where I left
path = '/Users/nianyun/Documents/Pictet_IPO_Case/Data/Google_trend'
file_path = os.listdir(path)
existing_index = [f.split('_')[-1].split('.')[0] for f in file_path]



with concurrent.futures.ThreadPoolExecutor(max_workers= 5) as executor:
    future_to_ipo = {executor.submit(all_trend,chunk, index): index for index, chunk in enumerate(list_chunk) if str(index) not in existing_index}
    for future in concurrent.futures.as_completed(future_to_ipo):
        letter = future_to_ipo[future]
        try:
            data = future.result()
        except Exception as exc:
            print('%r generated an exception: %s' % (letter, exc))
        else:
            print(f'search {letter} success')



# c = chunkify(iterable = company_date , chunk = 2) 
# type(c[0])
# df_list = all_trend(company_date = company_date )

# i = 0
# for item in chunkify(iterable = company_date , chunk = 2):
#     while i < 10:
#         print(item)
#         i +=1




# x = get_trend(search = 'Hiland Partners LP', date = '2014-07-31' , month_before= 3,month_after=9 )
# x.reset_index().rename(columns={ search : 'SVI'})





# c = [1,2]
# c.append(x)
# len(x)

# for company, date in company_date.items():
#     df_google = get_trend(search = company, date = date , month_before = 3, month_after = 9)
    


# pytrends = TrendReq()
# kw_list = ['Gemplus']
# pytrends.build_payload(kw_list, timeframe='2000-01-01 2001-01-01')
# df_2 = pytrends.interest_over_time()

# pytrends = TrendReq()
# kw_list = ['Apple']
# pytrends.build_payload(kw_list, timeframe='2000-01-01 2001-01-01')
# df_2 = pytrends.interest_over_time()








# pytrend = TrendReq()
# keywords = pytrend.suggestions(keyword='apple')
# x = pd.DataFrame(keywords)
# pytrend.interestOverTime({keyword: '/m/02prfxl'})
# pytrend.interest_over_time(keyword = '/m/02prfxl')

# pytrend = TrendReq()
# kw_list = ['Sprouts Farmers Market']
# pytrend.build_payload(['/m/02prfxl'], timeframe='today 1-m')





# pytrend = TrendReq()
# kw_list = ['Sprouts Farmers Market']
# pytrend.build_payload(kw_list, timeframe='today 1-m')
# df_1= pytrend.interest_over_time()


# pytrend = TrendReq()
# kw_list = ['/m/0k8z']
# pytrend.build_payload(kw_list, timeframe='today 1-m' )
# df_2= pytrend.interest_over_time()
