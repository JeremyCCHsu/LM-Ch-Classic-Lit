# coding=utf-8

# This script is best suited for http://ctext.org
# (Notice: the website actually prevents automatic downloading, so plz watch out.)
# 
# I've been banned for downloading it.   = =
# Maybe I can fake a browser?
# 	http://stackoverflow.com/questions/27652543/how-to-use-python-requests-to-fake-a-browser-visit
# 
# For BeatifulSoup, plz see
# http://www.crummy.com/software/BeautifulSoup/bs4/doc.zh/

import re
import urllib
from bs4 import BeautifulSoup
from time import sleep
from random import randint

import requests
import sys
# from random import randint

# import xml.etree.ElementTree as ET

def urlbase(i):
	return 'https://zh.wikisource.org/zh-hant/%E4%B8%89%E5%9C%8B%E6%BC%94%E7%BE%A9/%E7%AC%AC' + ('%03d' % i) +'%E5%9B%9E'

chapFirst = 1
chapLast  = 120
oFile = 'SanGuoYanYi%03d-%03d.txt' % (chapFirst, chapLast)
# oFile     = 'Zizhitongjan%03d-%03d.txt' % (chapFirst, chapLast)

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

# print(response.content)

# import numpy as np
# chs = np.array(range(chapFirst, chapLast))
# np.random.shuffle(chs)

# [TODO] I forgot to save retrieved files in order = =
with open(oFile, 'w') as f:
	for i in xrange(chapFirst, chapLast+1):
	# for i in chs:
		print 'Chapter %03d downloading...' % i
		url = urlbase(i)
		session = requests.Session()
		rsp = session.get(url, headers=headers)
		if rsp.status_code < 400:
			page = rsp.content
			soup = BeautifulSoup(page, 'html.parser')
			title = soup.find_all('td')
			title = title[4].text
			f.write(title.encode('utf-8'))
			f.write('\n')
			td = soup.find_all('p')
			print '  %d elements' % len(td)
			for t in td:
				text = t.get_text()
				f.write(text.encode('utf-8'))
				f.write('\n')
			f.write('\n\n')
			s = randint(50, 150) / 10.0
			print '  Sleep for %.1f sec' % s
			sleep(s)		# I still get caught with [20, 70]
		else:
			print "We're banned! (%d)" % rsp.status_code
			sys.exit(0)

		# url.close()

# tree = ET.fromstring(data)

# print tree
# tags = soup('a')
# for tag in tags:
#     print tag.get('href', None)
