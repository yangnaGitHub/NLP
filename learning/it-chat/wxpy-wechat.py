# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 17:28:06 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

#pip install wxpy

from __future__ import unicode_literals
from threading import Timer
from wxpy import *

#二维码是用像素的形式打印出来
#bot = Bot(console_qr=2, cache_path='botoo.pkl')
bot=Bot()#windows

def send_news(): 
    try:
        my_friend = bot.friends().search(u'树枝')[0]
        my_friend.send(u'杨娜test')
        t = Timer(86400, send_news)
        #每86400秒（1天），发送1次，不用linux的定时任务是因为每次登陆都需要扫描二维码登陆，很麻烦的一件事，就让他一直挂着吧   
        t.start()
    except:
        my_friend = bot.friends().search('树枝')[0]
        my_friend.send(u"今天消息发送失败了")

if __name__ == "__main__":
    send_news()