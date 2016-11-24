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
	return 'https://zh.wikisource.org/zh-hant/%E7%B4%85%E6%A8%93%E5%A4%A2/%E7%AC%AC' + ('%03d' % i) + '%E5%9B%9E'
	

chapFirst = 1
chapLast  = 120
oFile = 'HongLouMong%03d-%03d.txt' % (chapFirst, chapLast)
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
			title = soup.find('center').text
			f.write(title.encode('utf-8'))
			f.write('\n')
			td = soup.find_all('p')
			print '  %d elements' % len(td)
			for t in td:
				# if t.has_attr('class'):
				# 	cls = t.get('class')
				# 	cls = ' '.join(cls)
				# 	if cls == 'ctext':
				text = t.get_text()
				f.write(text.encode('utf-8'))
				f.write('\n')
						# print t.get_text()
				# elif t.h2:
				# 	text = t.h2.get_text()
				# 	f.write('\n')
				# 	f.write(text.encode('utf-8'))
			f.write('\n\n')
			s = randint(10, 100) / 10.0
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
