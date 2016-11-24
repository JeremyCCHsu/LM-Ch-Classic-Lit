# coding=utf-8

# For BeatifulSoup, plz see
# http://www.crummy.com/software/BeautifulSoup/bs4/doc.zh/

import re
import urllib
from bs4 import BeautifulSoup
# import xml.etree.ElementTree as ET

urlbase   = 'https://zh.wikisource.org/wiki/%E8%B3%87%E6%B2%BB%E9%80%9A%E9%91%91/%E5%8D%B7'
chapFirst = 001
chapLast  = 294
oFile     = 'Zizhitongjan%03d-%03d.txt' % (chapFirst, chapLast)

with open(oFile, 'w') as f:
	for i in xrange(chapFirst, chapLast+1):
		print 'Chapter %d downloading...' % i
		url  = '%s%03d' % (urlbase, i)
		url  = urllib.urlopen(url)
		page = url.read()
		url.close()
		# print 'Retrieve', len(data), 'chars'
		# print data
		soup = BeautifulSoup(page, 'html.parser')
		# text = soup.body.find_all('p')[1].text
		prgh = soup.body.find_all('p')
		for p in prgh:
			text = p.text
			if re.search('\d+', text):
				pass
			else:
				text = re.sub('\n', '', text)
				text = re.sub('\s+', '', text)
				f.write(text.encode('utf-8'))

# tree = ET.fromstring(data)

# print tree
# tags = soup('a')
# for tag in tags:
#     print tag.get('href', None)