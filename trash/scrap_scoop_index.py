from bs4 import BeautifulSoup
import pandas as pd
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests
import numpy as np
import string
import time

def get_url_ipo(search):
    url = f'https://www.iposcoop.com/ipo-index/{search}/'
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    li_list = soup.find_all(href = re.compile("https://www.iposcoop.com/ipo/"))
    url_list = []
    for li in li_list:
        url = li.get('href')
        url_list.append(url)
    return url_list


def get_table(url):
    table = pd.read_html(url)
    table = table[0].set_index(0)
    table_new = table.T
    table_new.columns = [str(col).replace(':','') for col in table_new.columns ]
    return table_new

def get_df_one_letter(search):
    df_list = []
    i = 0
    url_list = get_url_ipo(search =search )
    for url in url_list:
        try:
            df = get_table(url =url )
            if 'nan' in df.columns:
                df = df.drop(columns = 'nan').dropna(axis=0, how='all')
        except:
            time.sleep(20)
            df = get_table(url =url )
            if 'nan' in df.columns:
                df = df.drop(columns = 'nan').dropna(axis=0, how='all')
        df_list.append(df)
        i +=1
        print(f'{i} done in search {search}')
    df_final = pd.concat(df_list)
    df_final.reset_index(drop= True)
    df_final.to_csv(f'Data/IPO_info/ipo_info_scoop_{search}.csv', index = False)
    print(f'search {search} is done')
    return df_final

search_letter = list(string.ascii_uppercase)
search_number = list(np.arange(1,10))
search_list  = search_letter + search_number

c_df = get_df_one_letter(search = 'C')
s_df = get_df_one_letter(search = 'S')

import os
path = 'Data/Google_trend'
file_path = os.listdir(path)
existing_index = [f.split('_')[-1].split('.')[0] for f in file_path]
len(existing_index)
[i for i in [1,2,3,4,5] if str(i) not in existing_index ]
'5' in existing_index

list_chunk = [item for item in chunks(company_date, 5)]

chunk = list_chunk[15]
x = all_trend(list_chunk[15], 15)

df_google = get_trend(search = 'Lantern Pharma', date = '2020-06-11' , month_before = 3, month_after = 9)

pytrends = TrendReq()
kw_list = ['Apple']
pytrends.build_payload(kw_list)
df_2 = pytrends.interest_over_time()

# df_c = pd.concat(c_list[65:70])
# c_list[0]

# c_list[69].columns[0]

# c_list[69]['Industry']

# sub = c_list[69]

# sub.drop(columns = 'nan').dropna(axis=0, how='all')

# .dropna(axis=0, how='all')
# url = 'https://www.iposcoop.com/ipo-index/A/'
# r = requests.get(url)
# soup = BeautifulSoup(r.content, 'html.parser')

# get the link, and click into 


# for i in soup.find_all('li' , name = 'A'):
#     print(i)
#     break

# x = soup.find_all( attrs={"name": "A"})
# y = soup.find_all( attrs={"rel": "bookmark"})
# z = soup.find_all(href = re.compile("https://www.iposcoop.com/ipo/"))
# x[0].get('href')

# soup.find_all(href = re.compile("https://www.iposcoop.com/ipo/"))



# url = 'https://www.iposcoop.com/ipo/a-spac-i-acquisition-corp/'
# r = requests.get(url)
# soup = BeautifulSoup(r.content, 'html.parser')


# soup.find_all('table' , {"class": "ipo-table"})
# len(soup.find_all('table' , {"class": "ipo-table"}))
# soup.find_all('table' , {"class": "ipo-table"})

# pd.read_html(soup.find_all('table' , {"class": "ipo-table"}))

# table = pd.read_html('https://www.iposcoop.com/ipo/a-spac-i-acquisition-corp/')
# table = table[0].set_index(0)
# table_new = table.T
# table_new.columns
# #remove : in column 
# [col.replace(':','') for col in table_new.columns ]

# 'Business:'.replace(':','')

# table_new.columns.name = None
# table_new.index


# def get_url_ipo(search):
#     url = f'https://www.iposcoop.com/ipo-index/{search}/'
#     r = requests.get(url)
#     soup = BeautifulSoup(r.content, 'html.parser')
#     li_list = soup.find_all(href = re.compile("https://www.iposcoop.com/ipo/"))
#     url_list = []
#     for li in li_list:
#         url = li.get('href')
#         url_list.append(url)
#     return url_list


# def get_table(url):
#     table = pd.read_html(url)
#     table = table[0].set_index(0)
#     table_new = table.T
#     table_new.columns = [col.replace(':','') for col in table_new.columns ]
#     return table_new

# def get_all_table(search_list ):
#     df_list = []
#     i = 0
#     for search in search_list:
#         url_list = get_url_ipo(search =search )
#         for url in url_list:
#             try:
#                 df = get_table(url =url )
#             except:
#                 time.sleep(20)
#                 df = get_table(url =url )
#             df_list.append(df)
#             i +=1
#             print(f'{i} done')
#     df_final = pd.concat(df_list)
#     df_final.reset_index(drop= True)
#     return df_final

# search_letter = list(string.ascii_uppercase)
# search_number = list(np.arange(1,10))
# search_list  = search_letter + search_number
# df_ipo = get_all_table(search_list = search_list )

# df_ipo.to_csv('../Data/ipo_info_scoop.csv', index = False)



# url_list = get_url_ipo('A')
# url_list[34]

# get_table(url_list[34])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

fig, ax = plt.subplots(figsize=(15,8))
sns.lineplot(data = df , x = 'year' , y = '1st_Day_Return' , ax = ax)
ax2 = ax.twinx()
sns.countplot(data = df , x = 'year' ,ax = ax2)
plt.show()

df['year']= df['year'].astype(int)
x = df['year'].value_counts().reset_index().rename(columns = {'index':  'year', 'year' :'count'})
type(x['year'].iloc[0])


def calculate_ticks(ax, ticks, round_to=0.1, center=False):
    upperbound = np.ceil(ax.get_ybound()[1]/round_to)
    lowerbound = np.floor(ax.get_ybound()[0]/round_to)
    dy = upperbound - lowerbound
    fit = np.floor(dy/(ticks - 1)) + 1
    dy_new = (ticks - 1)*fit
    if center:
        offset = np.floor((dy_new - dy)/2)
        lowerbound = lowerbound - offset
    values = np.linspace(lowerbound, lowerbound + dy_new, ticks)
    return values*round_to

# fig, ax = plt.subplots(figsize=(15,8))
# sns.lineplot(data = df , x = 'year' , y = '1st_Day_Return' , ax = ax)
# ax2 =  ax.twinx().twiny()
# sns.barplot(data = x , x = 'year', y = 'count',ax = ax2)
# ax.set_xticks([2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020])
# ax2.set_xticks(['2004',2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020])
# plt.show()

fig, ax = plt.subplots(figsize=(15,8))
ax2 =  ax.twinx().twiny()
sns.countplot(data = df , x = 'year' ,ax = ax , Color = 'lightgreen')
sns.lineplot(data = df , x = 'year' , y = '1st_Day_Return' , ax = ax2 , color = 'darkred' , marker = 'o' , linewidth=1.5)
ax.set_xticks(['2004',2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020])
ax2.set_xticks([2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020])
ax.grid(None)
plt.show()



data = pd.DataFrame({'Day': [1, 2, 3, 4], 'Value': [3, 7, 4, 2], 'Value 2': [1, 7, 4, 5]})

f, ax = plt.subplots()
sns.barplot(data=df, x='Day', y='Value')
sns.pointplot(data=df, x='Day', y='Value 2')