import copy
import networkx as nx
from collections import deque

from utils.seed import rng
from utils import verifyBlock, computeLatency, pushq, incr_blocks


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
    def __init__(self, bid, pb, txnIncluded, miner, balance=[]):
        self.bid = bid # block id
        self.pb = pb #parent block id
        self.size = 1 + len(txnIncluded) #size of block
        if pb != 0: #to check if it is not genesis block
            # self.txnIncluded = copy.deepcopy(txnIncluded)
            # txnPool stores all the txn mined till now 
            # length shows the length of chain from genesis block till current block
            self.txnIncluded = txnIncluded
            self.txnPool = pb.txnPool
            self.length = pb.length+1
            self.balance = copy.deepcopy(pb.balance)
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
    def __init__(self, nid, speed, cpu, genesis, miningTime, blkgen, txngen):
        self.peers = set() # neighbours of the node
        self.blockChain = dict() # blockchain of the node
        self.blockReceived = set() # blocks received till now 
        self.blockTime = dict() # time at which each block arrived
        self.blockTime[1] = 0
        self.orphanBlocks = set() # blocks received whose parents have not been received yet
        self.txnReceived = set() # txn received till now 
        self.g = nx.DiGraph() # graph
        self.nid = nid # unique id of the node 
        self.speed = speed # 1=fast, 0=slow
        self.cpu = cpu
        self.lbid = genesis.bid
        # self.blockChain[genesis.bid] = copy.deepcopy(genesis)
        self.blockChain[genesis.bid] = genesis
        self.blkid_generator = blkgen
        self.miningTime = miningTime # avg interarrival time/hashpower
        self.txnid_generator = txngen
        self.created_blocks_own = 0

    # generates transactions
    def txnSend(self, event):
        if self.blockChain[self.lbid].balance[self.nid] < 1e-8: 
            #if balance is less than 1e-8 then no txn is generated
            return
        event.txn.value = rng.uniform(0, self.blockChain[self.lbid].balance[self.nid])
        if self.blockChain[self.lbid].balance[self.nid] > event.txn.value:
            self.txnReceived.add(event.txn) #add  recieved into set
            for i in self.peers:
                t = event.time + computeLatency(peerX=self, peerY=i, m=1)
                pushq(Event(t, event_type="TxnRecv", sender=self, receiver=i, txn=event.txn))
            

    # forwards transactions
    def txnRecv(self, event):
        if event.txn not in self.txnReceived:
            self.txnReceived.add(event.txn)
            for i in self.peers:
                t = event.time + computeLatency(peerX=self, peerY=i, m=1)
                pushq(Event(t, event_type="TxnRecv", sender=self.nid, receiver=i, txn=event.txn))
        

    # new block generation
    def mineNewBlock(self, block, lat):
        while True:
            remaingTxn = self.txnReceived.difference(block.txnPool)
            toBeDeleted = set([i for i in remaingTxn if i.value > block.balance[i.peerX.nid]])

            remaingTxn = remaingTxn.difference(toBeDeleted)
            numTxn = len(remaingTxn)
            
            if numTxn > 1:
                numTxn = min(rng.integers(1, numTxn), 1023) # 1 for coinbase txn, 1 for itself

            txnToInclude = set(rng.choice(list(remaingTxn), numTxn))
            txnId = next(self.txnid_generator)
            coinBaseTxn = Transaction(id=txnId, peerX=-1, peerY=self, value=50, case=2)
            txnToInclude.add(coinBaseTxn)

            newBlockId = next(self.blkid_generator)
            # print(newBlockId)
            newBlock = Block(bid=newBlockId, pb=block,
                            txnIncluded=txnToInclude, miner=self)
            
            if verifyBlock(newBlock):
                break

        lat = lat + rng.exponential(self.miningTime) #takes mean not lambda
        pushq(Event(lat, event_type="BlockMined", block=newBlock))


    #this function is called, if block receives a node from its peers
    #block is verified and if the block is without any errors then its is added to blockchain 
    # and then transmitted to neighbours 
    # If addition of that block creates a primary chain then mining is started over that block
    def verifyAndAddReceivedBlock(self, event):
        if event.block.bid not in self.blockReceived:
            self.blockReceived.add(event.block.bid)
            if not verifyBlock(event.block):
                return
            if event.block.pb.bid not in self.blockChain:
                self.orphanBlocks.add(event.block)
                return

            orphanProcessing = deque()
            orphanProcessing.append(event.block)
            last_block = event.block
            while len(orphanProcessing) > 0:
                curr_block = orphanProcessing.popleft()
                self.blockTime[curr_block.bid] = event.time
                self.blockChain[curr_block.bid] = curr_block
                self.g.add_edge(curr_block.bid, curr_block.pb.bid)
                if curr_block.length > last_block.length:
                    last_block = curr_block
                for i in self.peers:
                    lat = event.time + computeLatency(peerX=self, peerY=i, m=curr_block.size)
                    pushq(Event(lat, event_type="BlockRecv", sender=self, receiver=i, block=curr_block))
                for orphanBlock in self.orphanBlocks.copy():
                    if orphanBlock.pb.bid == curr_block.bid:
                        self.orphanBlocks.remove(orphanBlock)
                        orphanProcessing.append(orphanBlock)

            if last_block.length > self.blockChain[self.lbid].length:
                self.lbid = last_block.bid
                self.mineNewBlock(block=last_block, lat=event.time)
        
    # thsi function is called once the mining of a block is completed, 
    # If after mining the addition of block creates a primary chain then
    # the block is shared with neighbours and mining is continued otherwise 
    # node waits a block whose addition will, create primary chain
    def receiveSelfMinedBlock(self, event):
        if event.block.length <= self.blockChain[self.lbid].length:
            return

        incr_blocks()
        self.created_blocks_own += 1

        self.blockTime[event.block.bid] = event.time
        self.blockChain[event.block.bid] = event.block
        self.g.add_edge(event.block.bid, event.block.pb.bid)
        self.blockReceived.add(event.block.bid)
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