# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:13:59 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

from flask import Flask
from flask import request,render_template
from conf import args
from log import log
import json
import Allmodule

app = Flask(__name__)
module = Allmodule.module(args, None)

@app.route('/deep_chat/v2',methods=["POST"])
def chat():
    client_params = request.get_json(force=True)
    server_param = {}
    if client_params['method'] == 'chat':
        server_param['result'] = module.predict(**client_params)
    elif client_params['method'] == 'retrain':#1
        server_param['result'] = module.train(**client_params)
    elif client_params['method'] == 'lookup':
        server_param['result'] = module.lookup(**client_params)
    elif client_params['method'] == 'log':
        server_param['result'] = module.start_log(**client_params)
    elif client_params['method'] == 'live':
        params = {'success':'true','version':args.version}
        server_param['result']=params
        
    server_param['id'] = client_params['id']
    server_param['jsonrpc'] = client_params['jsonrpc']
    server_param['method'] = client_params['method']
    log(server_param)
    return json.dumps(server_param, ensure_ascii=False).encode("utf-8")

class Server():
    def __init__(self, debug=True, thread=True):
        app.run(debug=debug, host=args.http_host, port=args.http_port, threaded=thread)

#class Server():
#    def __init__(self, debug=True, thread=True):
#        app = Flask(__name__)
#        args = get_args()
#        module = Allmodule.module(args, None)
#
#        @app.route('/deep_chat/v2',methods=["POST"])
#        def chat():
#            client_params = request.get_json(force=True)
#            server_param = {}
#            if client_params['method'] == 'chat':
#                server_param['result'] = module.predict(**client_params)
#            elif client_params['method'] == 'retrain':#1
#                server_param['result'] = module.train(**client_params)
#            elif client_params['method'] == 'lookup':
#                server_param['result'] = module.lookup(**client_params)
#            elif client_params['method'] == 'log':
#                server_param['result'] = module.start_log(**client_params)
#            elif client_params['method'] == 'live':
#                params = {'success':'true','version':args.version}
#                server_param['result']=params
#        
#            server_param['id'] = client_params['id']
#            server_param['jsonrpc'] = client_params['jsonrpc']
#            server_param['method'] = client_params['method']
#            log(server_param)
#            return json.dumps(server_param, ensure_ascii=False).encode("utf-8")
#
#        app.run(debug=debug, host=args.http_host, port=args.http_port, threaded=thread)