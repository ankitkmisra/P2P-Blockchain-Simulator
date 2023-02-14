import copy
import numpy as np 
import networkx as nx
from random import sample
from numpy.random import default_rng

from utils import verifyBlock, computeLatency, pushq

rng = default_rng(42)

class Event:
    """
    Event class: Stores all types of Events
    
    EVENT_TYPES: "TxnGen", "TxnRecv", "BlockRecv", "BlockMined"
    """

    def __init__(self, time, event_type, txn=None, block=None, sender=None, receiver=None):
        self.time = time
        self.event_type = event_type
        self.txn = txn
        self.block = block
        self.sender = sender
        self.receiver = receiver
    

class Block:
    time=0
    def __init__(self, bid, pbid, txnIncluded, miner, balance=[]):
        self.bid = bid # block id
        self.pbid = pbid #parent block id
        self.size = 1 + len(txnIncluded) #size of block
        if pbid != 0: #to check if it is not genesis block
            # self.txnIncluded = copy.deepcopy(txnIncluded)
            # txnPool stores all the txn mined till now 
            # length shows the length of chain from genesis block till current block
            self.txnIncluded = txnIncluded
            self.txnPool = pbid.txnPool
            self.length = pbid.length+1
            self.balance = copy.deepcopy(pbid.balance)
        else:
            self.txnIncluded = set()
            self.txnPool = set()
            self.length = 1
            self.balance = balance
        self.miner = miner
        
        for i in txnIncluded: #updating balance of all the user 
            if i.case == 1:
                self.balance[i.peerX.nid] -= i.value
            self.txnPool.add(i)
            self.balance[i.peerY.nid] += i.value

class Node:
    peers = set() # neighbours of the node
    blockChain = {} # blockchain of the node
    blockReceived = set() # blocks received till now 
    txnReceived = set() # txn received till now 
    
    g = nx.DiGraph() # graph


    def __init__(self, nid, speed, cpu, genesis, miningTime, blkgen, txngen):
        self.nid = nid # unique id of the node 
        self.speed = speed # 1=fast, 0=slow
        self.cpu = cpu
        self.lbid = genesis.bid
        # self.blockChain[genesis.bid] = copy.deepcopy(genesis)
        self.blockChain[genesis.bid] = genesis
        self.blkid_generator = blkgen
        self.miningTime = miningTime # avg interarrival time/hashpower
        self.txnid_generator = txngen

    # genetares transactions
    def txnSend(self, event):
        event.txn.value = np.random.uniform(0, self.blockChain[self.lbid].balance[self.nid]+1)
        if self.blockChain[self.lbid].balance[self.nid] > event.txn.value:
            self.txnReceived.add(event.txn) #add  recieved into set
            for i in self.peers:
                t = event.time + computeLatency(peerX=self, peerY=i, m=1)
                pushq(Event(t, event_type="TxnRecv", sender=self, receiver=i, txn=event.txn))
            

    # forwards transactions
    def txnRecv(self,event):
        if event.txn not in self.txnReceived:
            self.txnReceived.add(event.txn)
            for i in self.peers:
                t = event.time + computeLatency(peerX=self, peerY=i, m=1)
                pushq(Event(t, event_type="TxnRecv", sender=self.nid, receiver=i, txn=event.txn))
        

    # new block generation
    def mineNewBlock(self, block, lat):
        remaingTxn = self.txnReceived.difference(block.txnPool)
        toBeDeleted = set()

        _= [toBeDeleted.add(i) for i in remaingTxn if i.value > block.balance[i.peerX.nid]]
        remaingTxn = remaingTxn.difference(toBeDeleted)
        numTxn = len(remaingTxn)
        
        if numTxn > 1:
            numTxn = min(np.random.randint(1, numTxn), 1022) # 1 for coinbase txn, 1 for itself

        txnToInclude = set(sample(remaingTxn, numTxn))
        txnId = next(self.txnid_generator)
        coinBaseTxn = Transaction(id=txnId, peerX=-1, peerY=self, value=50, case=2)
        txnToInclude.add(coinBaseTxn)

        newBlockId = next(self.blkid_generator)
        newBlock = Block(bid=newBlockId, pbid=block,
                         txnIncluded=txnToInclude, miner=self)

        lat = lat + rng.exponential(self.miningTime) #takes mean not lambda
        pushq(Event(lat, event_type="BlockMined", block=newBlock))


    #this function is called, if block receives a node from its peers
    #block is verified and if the block is without any errors then its is added to blockchain 
    # and then transmitted to neighbours 
    # If addition of that block creates a primary chain then mining is started over that block
    def verifyAndAddReceivedBlock(self, event):
        if event.block.bid not in self.blockReceived:
            self.blockReceived.add(event.block.bid)
            if verifyBlock(event.block):
                return

            event.block.time = event.time
            self.blockChain[event.block.bid] = event.block
            self.g.add_edge(event.block.bid, event.block.pbid.bid)
            if event.block.length > self.blockChain[self.lbid].length:
                self.lbid = event.block.bid
                self.mineNewBlock(block=event.block, lat=event.time)

            for i in self.peers:
                lat = event.time + computeLatency(peerX=self, peerY=i, m=event.block.size)
                pushq(Event(lat, event_type="BlockRecv", sender=self, receiver=i, block=event.block))
        
    # thsi function is called once the mining of a block is completed, 
    # If after mining the addition of block creates a primary chain then
    # the block is shared with neighbours and mining is continued otherwise 
    # node waits a block whose addition will, create primary chain
    def receiveSelfMinedBlock(self, event):
        event.block.time = event.time

        self.blockChain[event.block.bid] = event.block
        self.g.add_edge(event.block.bid, event.block.pbid.bid)
        
        self.blockReceived.add(event.block.bid)

        if event.block.length > self.blockChain[self.lbid].length:
            self.lbid = event.block.bid
            for i in self.peers:
                lat = event.time + computeLatency(peerX=self, peerY=i, m=event.block.size)
                pushq(Event(lat, event_type="BlockRecv", sender=self, receiver=i, block=event.block))
                self.mineNewBlock(block=event.block, lat=event.time)


class Transaction:
    #TxnID: IDx pays IDy C coins
    def __init__(self, id, peerX, peerY, value, case) -> None:
        self.txnId = id
        self.peerX = peerX
        self.peerY = peerY
        self.value = value
        self.case = case
        self.size = 1

    def __repr__(self) -> str:
        if(self.case==1):
            return str(self.txnId)+": "+str(self.peerX)+" pays "+str(self.peerY)+" "+str(self.coins)+" coins"
        if(self.case==2):
           return str(self.txnId)+": "+str(self.peerX)+" mines "+str(self.coins)+" coins"