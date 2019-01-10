# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 14:21:43 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import os
import sys

def usage():
    print('python XX.py param1 [param2] [param3]')
    print('param1 [env|sig|mul]')
    print('param2 [YY.py]==>if param1 is [sig|mul]')
    print('param3 [filename]==>if param1 is [mul]')

if __name__ == '__main__':
    try:
        if 'env' == sys.argv[1]:
            os.system('source activate yangna')
        elif 'sig' == sys.argv[1]:
            os.system('python3.5 %d' % sys.argv[2])
        elif 'mul' == sys.argv[1]:
            os.system('cp Allmodule.cfg Allmodule_bk.cfg')
            os.system('python3.5 %d > %s.txt' % (sys.argv[2], 'Allmodule'))
            
            with open(sys.argv[3], 'r', encoding='utf-8') as fd:
                for line in fd.readlines():
                    line = line.strip()
                    if line.endswith('.cfg'):
                        os.system('cp %s Allmodule.cfg' % line)
                        os.system('python3.5 %d > %s.txt' % (sys.argv[2], line[:line.find('.cfg')]))
            os.system('cp Allmodule_bk.cfg Allmodule.cfg')
    except:
        usage()