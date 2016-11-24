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
	return 'https://zh.wikisource.org/zh-hant/%E8%B3%87%E6%B2%BB%E9%80%9A%E9%91%91/%E5%8D%B7' + ('%03d' % i)

chapFirst = 1
chapLast  = 264
oFile = 'XiYouJi%03d-%03d.txt' % (chapFirst, chapLast)
# oFile     = 'Zizhitongjan%03d-%03d.txt' % (chapFirst, chapLast)

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
# pattern = '(' + u'第' + '\S+' u'回' + '[.\n]+)' + u'作者'

# [\d+]
# ^\d+\s+


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
			content = soup.find('div', id='mw-content-text')
			for c in content:
				if c.name == 'h1' or c.name == 'h2':
					# print c.span.text
					f.write(c.text.encode('utf-8'))
					f.write('\n')
				elif c.name == 'p' or c.name == 'dl':
					# print c.text.encode('utf-8')'
					text = c.text
					text = re.sub('\[\d+\]', '', text)
					text = re.sub('^\d+\s+', '', text)
					# print text.encode('utf-8')
					f.write(c.text.encode('utf-8'))
					f.write('\n')
			f.write('\n')
			s = randint(10, 50) / 10.0
			print '  Sleep for %.1f sec' % s
			sleep(s)
		else:
			print "We're banned! (%d)" % rsp.status_code
			sys.exit(0)


