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

# import xml.etree.ElementTree as ET

urlbase   = 'http://ctext.org/sanguo-yanyi/ch%d/zh'
chapFirst = 48
chapLast  = 120
oFile = 'SanGuoYanYi%3d-%3d.txt' % (chapFirst, chapLast)
# oFile     = 'Zizhitongjan%03d-%03d.txt' % (chapFirst, chapLast)

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

# print(response.content)


with open(oFile, 'w') as f:
	for i in xrange(chapFirst, chapLast+1):
		print 'Chapter %03d downloading...' % i
		url = urlbase % i
		rsp = requests.get(url, headers=headers)
		# url  = urllib.urlopen(url)
		# page = url.read()
		page = rsp.content
		soup = BeautifulSoup(page, 'html.parser')
		td = soup.find_all('td')
		print '  %d elements' % len(td)
		for t in td:
			if t.has_attr('class'):
				cls = t.get('class')
				cls = ' '.join(cls)
				if cls == 'ctext':
					text = t.get_text()
					f.write(text.encode('utf-8'))
					# print t.get_text()
			elif t.h2:
				text = t.h2.get_text()
				f.write('\n')
				f.write(text.encode('utf-8'))
		sleep(randint(78, 123) / 10.0)		# I still get caught with [20, 70]
		# url.close()

# tree = ET.fromstring(data)

# print tree
# tags = soup('a')
# for tag in tags:
#     print tag.get('href', None)