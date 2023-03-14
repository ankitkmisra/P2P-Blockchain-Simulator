import heapq
from utils.seed import rng

__all__ = ["blkIdGen",
           "txnIdGen",
           "verifyBlock",
           "initLatency",
           "computeLatency",
           "pushq",
           "eventq",
           "rho",
           "incr_blocks",
           "get_blocks"]


rho = None # rho is the global latency matrix
eventq = [] # eventq is the global event queue
created_blocks = 0

## Generators for block and transaction ids ##
def blkIdGen():
    i = 2 # genesis block is 1, and parent block of genesis is 0
    while True:
        yield i
        i += 1

def txnIdGen():
    i = 0
    while True:
        yield i
        i += 1

## Block Verification ##
def verifyBlock(cblock):
    for i in cblock.txnIncluded:
        if i.peerX == -1:
            return True

        if cblock.pb.balance[i.peerY.nid] + i.value != cblock.balance[i.peerY.nid] or \
            cblock.pb.balance[i.peerX.nid] - i.value < 0 or \
            cblock.pb.balance[i.peerX.nid] - i.value != cblock.balance[i.peerX.nid]:
            return False
            
    return True

## Latency Matrix ##
def initLatency(n):         # initializing a 2d array for rho
    global rho
    rho = rng.uniform(10, 500, [n, n])

def computeLatency(peerX, peerY, m):  # computing the latency by taking both nodes and size of message
    if(peerX.speed=="fast" and peerY.speed=="fast"):
        cij = 100 #mb 
    else:
        cij = 5
    dij = rng.exponential(96/cij) #msec
    return rho[peerX.nid][peerY.nid] + m/cij + dij

## a helper function to push an event into the event queue ##
def pushq(event):
    heapq.heappush(eventq, (event.time, event))

def incr_blocks():
    global created_blocks
    created_blocks += 1

def get_blocks():
    return created_blocks