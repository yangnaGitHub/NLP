# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 17:56:38 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

#区块链是由区块的记录构成的不可变,有序的链结构,记录可以是交易,文件或任何你想要的数据
#它们是通过哈希值链接起来的
#
#创建一个 Blockchain 类,构造函数中创建了两个列表,一个用于储存区块链,一个用于储存交易
#每个block包括:索引index,时间戳,交易列表,工作量证明(挖矿),和前一个block的哈希值(区块链的不可变)
#block = {
#    'index': 1,
#    'timestamp': 1506057125.900785,
#    'transactions': [
#        {
#            'sender': '8527147fe1f5426f9dd545de4b27ee00',
#            'recipient': 'a77f5cdfa2934df3954a5c7c7da5df1f',
#            'amount': 5,
#        }
#    ],
#    'proof': 324984774000,
#    'previous_hash': '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
#}
#新的区块依赖工作量证明算法(PoW)来构造
#找出一个符合特定条件的数字(难计算,容易验证)
#比特币使用的工作量证明算法是Hashcash,为了争夺创建区块的权利而争相计算结果

import hashlib
import json
from time import time
from urllib.parse import urlparse
import requests

class Blockchain(object):
    def __init__(self):
        self.chain = []
        self.current_transactions = []
        
        self.nodes = set()#储存节点
        
        self.new_block_and_add_to_chain(previous_hash=1, proof=100)
    
    #注册节点
    def register_node(self, address):
        parsed_url = urlparse(address)
        self.nodes.add(parsed_url.netloc)
    
    #检查是否是有效链
    def valid_chain(self, chain):
        last_block = chain[0]
        current_index = 1
        while current_index < len(chain):
            block = chain[current_index]
            if block['previous_hash'] != self.my_hash(last_block):#当前保存的是上一个block的hash值
                return False
            if not self.valid_proof(last_block['proof'], block['proof']):#是否符合规则
                return False
            last_block = block
            current_index += 1
        return True
    
    #共识算法解决冲突,最长的chain
    def resolve_conflicts(self):
        neighbours = self.nodes
        new_chain = None
        max_length = len(self.chain)
        for node in neighbours:#所有的节点
            response = requests.get(f'http://{node}/chain')#每个节点的不同的chain
            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']
                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain
        if new_chain:
            self.chain = new_chain
            return True
        return False
    
    def new_block_and_add_to_chain(self, proof, previous_hash=None):
        #创建一个新的区并加到链中
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.my_hash(self.chain[-1]),
        }
        self.current_transactions = []
        self.chain.append(block)
        return block
    
    def new_transaction(self, sender, recipient, amount):
        #将新的事务加到current_transactions
        self.current_transactions.append({
            'sender': sender,
            'recipient': recipient,
            'amount': amount,
        })
        return self.last_block['index'] + 1
    
    @staticmethod
    def my_hash(block):
        #哈希一个区使用sha-256算法,256bit,然后16进制后转化成字符串该是64个字符
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    @property
    def last_block(self):
        #返回最近加入到链中的区
        return self.chain[-1]
    
    def proof_of_work(self, last_proof):
        proof = 0
        while self.valid_proof(last_proof, proof) is False:
            proof += 1
        return proof
    
    def valid_proof(last_proof, proof):
        guess = f'{last_proof}{proof}'.encode()#last_proof=100, proof=55 ==> 10055拼字符串
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"

blockchain = Blockchain()

from uuid import uuid4
from flask import Flask, jsonify, request 
#创建三个接口
app = Flask(__name__)#创建一个节点
node_identifier = str(uuid4()).replace('-', '')
blockchain = Blockchain()

#挖矿
#计算工作量证明
#通过新增一个交易授予矿工一个币
#构造新区块并将其添加到链中
#http://localhost:5000/mine访问
@app.route('/mine', methods=['GET'])
def mine():
    last_block = blockchain.last_block
    last_proof = last_block['proof']
    proof = blockchain.proof_of_work(last_proof)
    blockchain.new_transaction(
        sender='0',
        recipient=node_identifier,
        amount=1,
    )
    block = blockchain.new_block(proof)
    response = {
        'message': 'New Block Forged',
        'index': block['index'],
        'transactions': block['transactions'],
        'proof': block['proof'],
        'previous_hash': block['previous_hash'],
    }
    return jsonify(response), 200

#发送交易
@app.route('/transactions/new', methods=['POST'])
def new_transaction():
    values = request.get_json()
    required = ['sender', 'recipient', 'amount']
    if not all(k in values for k in required):#同时有'sender', 'recipient', 'amount'
        return 'Missing values', 400
    index = blockchain.new_transaction(values['sender'], values['recipient'], values['amount'])
    response = {'message': f'Transaction will be added to Block {index}'}
    return jsonify(response), 201

#返回整个区块链
@app.route('/chain', methods=['GET'])
def full_chain():
    response = {
        'chain': blockchain.chain,
        'length': len(blockchain.chain),
    }
    return jsonify(response), 200

@app.route('/nodes/register', methods=['POST'])
def register_nodes():
    values = request.get_json()
    nodes = values.get('nodes')
    if nodes is None:
        return "Error: Please supply a valid list of nodes", 400
    for node in nodes:
        blockchain.register_node(node)
    response = {
        'message': 'New nodes have been added',
        'total_nodes': list(blockchain.nodes),
    }
    return jsonify(response), 201

@app.route('/nodes/resolve', methods=['GET'])
def consensus():
    replaced = blockchain.resolve_conflicts()
    if replaced:
        response = {
            'message': 'Our chain was replaced',
            'new_chain': blockchain.chain
        }
    else:
        response = {
            'message': 'Our chain is authoritative',
            'chain': blockchain.chain
        }
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

#一致性问题(分布式)
#需要找到一种方式让一个节点知道它相邻的节点
#每个节点都需要保存一份包含网络中其他节点的记录

#tips:
#x=5
#for index in range(5):
#    print(f'{x+index}')