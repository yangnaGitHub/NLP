# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:13:59 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

from conf import args
from local import Local as Local
import server
#import sys
#from server import Server as Server
#import os

if __name__ == '__main__':
    #切换目录,非必要使用代码,可能报错,有需要的时候在放出来
    #if os.getcwd() != os.path.dirname(__file__):
        #os.chdir(os.path.dirname(__file__))
    
    #os.chdir(os.path.dirname(__file__))
    #os.chdir('E:\\AboutStudy\\code\\python\\natasha_suda')
    
    #update args
    #args.update(sys.argc, sys.argv)
    
    if 1 == args.local_method:
        local_test = Local(args)
        local_test.debug()
    elif 0 == args.local_method:
        server_run = server.Server(debug=False, thread=True)