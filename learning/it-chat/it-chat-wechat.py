# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:41:28 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

#pip install itchat

import itchat

#login.weixin.qq.com
#webpush.wx.qq.com
itchat.login()

friends = itchat.get_friends(update=True)[0:]#好友相关的信息,是一个列表
#print(friends)
#我的感受,这应该就是web端的wechat的爬虫而已吧,只是打包好了

def get_val(friend, feature):
    return friend[feature]

#friends[0]自己的信息
#<User: {'MemberList': <ContactList: []>, 'UserName': '@3bae1a3902914ce0607dc65271b8466b', 'City': '', 'DisplayName': '', 'PYQuanPin': 'shuzhi', 'RemarkPYInitial': '', 'Province': '', 'KeyWord': 'yan', 'RemarkName': '', 'PYInitial': 'SZ', 'EncryChatRoomId': '', 'Alias': '', 'Signature': '去冒险，去体验，去爱想爱的人，去做想做的事，去活成想要活成的', 'NickName': '树枝', 'RemarkPYQuanPin': '', 'HeadImgUrl': '/cgi-bin/mmwebwx-bin/webwxgeticon?seq=665740665&username=@3bae1a3902914ce0607dc65271b8466b&skey=@crypt_79c76e86_dd8d3abc9bee8287b2efb295ec7ff51f', 'UniFriend': 0, 'Sex': 1, 'AppAccountFlag': 0, 'VerifyFlag': 0, 'ChatRoomId': 0, 'HideInputBarFlag': 0, 'AttrStatus': 168037, 'SnsFlag': 17, 'MemberCount': 0, 'OwnerUin': 0, 'ContactFlag': 2055, 'Uin': 535084920, 'StarFriend': 0, 'Statues': 0, 'WebWxPluginSwitch': 0, 'HeadImgFlag': 1, 'IsOwner': 0}>
sexcout = [0, 0, 0]
#可取Sex, NickName, Province, City, Signature等信息
for friend in friends[1:]:
    if get_val(friend, 'Sex') in [1, 2]:
        sexcout[get_val(friend, 'Sex')] += 1
    else:
        sexcout[0] += 1
total = sum(sexcout)
print('total[{}], male[{}], female[{}]'.format(total, sexcout[1], sexcout[2]))

import re
text_list = []
for friend in friends:
    Signature = get_val(friend, 'Signature').strip().replace('span', '').replace('class', '').replace('emoji', '')
    re_rule = re.compile('1f\d+\w*|[<>/=]')
    Signature = re_rule.sub('', Signature)
    text_list.append(Signature)
text = ''.join(text_list)
print(text)