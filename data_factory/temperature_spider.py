# -*- coding: utf-8 -*-


import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
import os, sys
import csv
import time
import importlib

importlib.reload(sys)

def get_one_page(url):
    '''
    get the website
    '''
    print('load '+url)
    headers = {'User-Agent':'User-Agent:Mozilla/5.0'}
    try:
        response = requests.get(url,headers=headers)
        if response.status_code == 200:
            return response.content
        return None
    except RequestException:
        return None

def parse_one_page(html):
    '''
    parse the content in the website
    '''
    
    html = (html.decode('gbk').encode('utf-8').decode('utf-8'))
    soup = BeautifulSoup(html,  "lxml")
    info = soup.find('div',  class_='wdetail')
    rows=[]
    tr_list = info.find_all('tr')[1:]
    for index,  tr in enumerate(tr_list):
        td_list = tr.find_all('td')
        date = td_list[0].text.strip().replace("\n", "")
        weather = td_list[1].text.strip().replace("\n", "").split("/")[0].strip()
        temperature_high = td_list[2].text.strip().replace("\n",  "").split("/")[0].strip()
        temperature_low = td_list[2].text.strip().replace("\n",  "").split("/")[1].strip()

        rows.append((date,weather,temperature_high,temperature_low))
    return rows

def getTemperatureData(city_ids, years, months, outdir='./'):
    city_map = {33401: 'hangzhou',
                33402: 'jiaxing',
                33403: 'shaoxing',
                33404: 'jinhua',
                33405: 'ningbo',
                33406: 'taizhou',
                33407: 'lishui',
                33408: 'huzhou',
                33409: 'quzhou',
                33410: 'wenzhou',
                33411: 'zhoushan',}

    for cid in city_ids:
        with open(os.path.join(outdir, 'weather_{}.csv'.format(cid)), 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'weather', 'temperature_high','temperature_low'])
            for j in range(len(years)):
                for month in months[j]:
                    url = 'http://www.tianqihoubao.com/lishi/{}/month/{}{:02d}.html'.format(city_map[cid], years[j], month)
                    html = get_one_page(url)
                    content = parse_one_page(html)
                    writer.writerows(content)
                    print('{} {}{:02d} is OK!'.format(city_map[cid], years[j], month))
                    time.sleep(2)

# test
if __name__ == '__main__':
    city_ids = [33406]
    years = [2019]
    months = [list(range(1, 11))]

    getTemperatureData(city_ids, years, months, '../Repo/data/check/')
