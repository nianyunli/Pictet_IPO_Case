from bs4 import BeautifulSoup
import pandas as pd
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests
import numpy as np
import string
import time
import concurrent.futures

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
            #shape error, duplicate nan
            if 'nan' in df.columns:
                df = df.drop(columns = 'nan').dropna(axis=0, how='all')
        df_list.append(df)
        i +=1
        print(f'{i} done in search {search}')
    df_final = pd.concat(df_list)
    df_final.reset_index(drop= True)
    df_final.to_csv(f'../Data/IPO_info/ipo_info_scoop_{search}.csv', index = False)
    print(f'search {search} is done')
    return df_final

search_letter = list(string.ascii_uppercase)
search_number = list(np.arange(1,10))
search_list  = search_letter + search_number


with concurrent.futures.ThreadPoolExecutor(max_workers= 35) as executor:
    future_to_ipo = {executor.submit(get_df_one_letter,search): search for search in search_list}
    for future in concurrent.futures.as_completed(future_to_ipo):
        letter = future_to_ipo[future]
        try:
            data = future.result()
        except Exception as exc:
            print('%r generated an exception: %s' % (letter, exc))
        else:
            print(f'search {letter} success')

