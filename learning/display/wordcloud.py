# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:22:03 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

#pip install wordcloud

import jieba

text = """去冒险，去体验，去爱想爱的人，去做想做的事，去活成想要活成的劳资这么帅，你算哪块小饼干莫愁前路无知己，天下谁人不识君。生于鞭，死于懒。心有猛虎，细嗅蔷薇。挑灯看剑，不忘初心。皆应是你世事洞明皆学问 人情练达即文章是个好人得意时淡然，失意时坦然勇敢的心为了北方神的荣耀！努力 不是为了超越别人 只为给自己一个交代匪交匪舒，天子所予。天猫志硕小哥石桥铺权志龙你身边的电脑配件咨询顾问 17384065760赠人玫瑰～手有余香～好吃不过饺子小象学院官网- www.chinahadoop.cn一打波改动正在接近！！！书到用时方恨少Always stay out of my comfort zone.fisherzhu君子坦荡荡，小人常戚戚；心胸宽广，思想开朗，遇事拿得起放的.居然没有听到寒山寺的钟声当我也老了……上海直招整租一室户，独门厨卫，精装全配实体墙男\码程序为生\保大\先救你，我妈会游泳热爱生活的人最幸福！！！珍惜一切，就算没有拥有！If you are fine,the sun will always shine.苏州吉他培训  吉他零售批发改装维修，全国包邮发货送配件！一颗求佛的心，一缕骚动的灵魂一脸风轻云淡，一生坚持前行15298837983小三被刮了！心存感恩 自爱爱人不卑不亢，不慌不忙。做最好的自己“请大家成为爱自己并散发光芒的人”做简单的小女人！过有品味的生活！ " 2615"宁死不屈乃真汉子，英雄末路最是悲壮！剑在手，问天下谁是英雄！专业吉他培训，出售，维修，批发，改装，出租，欢迎批发！！！望一切顺利会数据分析的健身伙夫，做一个不失赤子之心的小人小象学院官网-www.chinahadoop.cnVIVA旅行家首席小编，最具体验价值的原味旅行！路会越走越远 但只要你回头 你会发现 我会一直在你身后得意时淡然，失意时坦然过往即寻常。顺从大道，改故鼎新 -- 随算法、机器学习修出身材，挺出内涵见天地，见众生，见自我。他出现在你生命中，绝非偶然，而是一种必然。把你放在兜兜你，带你丈步走天涯。万物皆虚，万事皆允。电信、移动、长城宽带安装维护！优惠办理！做一个专注的吃货。我只愿面朝大海，春暖花开。?_?哈哈向日葵，一朵对着太阳咧嘴傻笑的花穷则独善其身,达则兼济天下Wait for a person to travel around the world.我在拉萨等你悲喜自渡  他人难悟  人难自渡  何以渡人  世间清浅  悲喜自尝信贷 " "房抵 " "车抵  平安小晏 咨询电话：17621909705熙熙宝贝'你健健康康'平平安安是粑粑麻麻最大的心愿居安思危，思则有备，有备无患。幸福不是如何得到你想要的，而是如何与你不想要的和平共处。Qq " "从来就没有什么救世主也不靠神仙皇帝美好的肉体和有趣的灵魂半亩诗田耕岁月，三元股海话谋生。无论你正在经历什么，请相信，一切都是最好的安排 " "What we have to learn,we learn by doing.你放过我我放过你无限大な梦のあとの　何もない世の中じゃ人的感情一旦深厚就会看似淡薄co是我的心头好~笙歌一夜鱼龙舞，终有繁华褪尽时。靜心品茶，用心生活！精彩度过每一天！ " "毋庸置疑，好的事情总会到来。而当它来晚时也不失为一种惊喜头晕～～今天又是充满希望的一天磕磕绊绊常有，一马平川难求买房 " 23e9"卖房 " 23e9"租房18262050025招聘正在进行中，恢复正常面试报道13451614629丁经理活在当下，心无挂碍知行合一愿有岁月可回首，且以真情共白头。cest la vie心情真的不错~啦啦啦Mia San Mia.May I know you further？What's on your face?Beauty.多一点真诚，少一点套路I will be always with you！森米内调减脂，不节食不反弹，42天华丽蜕变想出去旅游的都来找我啊BALANCE！Don't forget.all  feel生活不是抱怨，而是努力争取！！育儿先育己！"""
wordlist = jieba.cut(text, cut_all=True)
word_space_split = ' '.join(wordlist)

import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
import numpy as np
import PIL.Image as Image

coloring = np.array(Image.open('wordcloud.jpg'))
my_wordcloud = WordCloud(background_color='white', max_words=2000, mask=coloring, max_font_size=60, 
                         random_state=42, scale=2, font_path='simheittf.ttf').generate(word_space_split)

image_colors = ImageColorGenerator(coloring)
plt.imshow(my_wordcloud.recolor(color_func=image_colors))
plt.imshow(my_wordcloud)
plt.axis('off')
plt.show()