import heapq, random
from numpy.random import default_rng

rng = default_rng(42)

rho = None # rho is the global latency matrix
eventq = [] # eventq is the global event queue


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

def verifyBlock(cblock):
    for a in cblock.txnIncluded:
        if a.sender == -1:
            return True

        if cblock.pbid.balance[a.receiver.nid] + a.value != cblock.balance[a.receiver.nid] or \
            cblock.pbid.balance[a.sender.nid] - a.value < 0 or \
            cblock.pbid.balance[a.sender.nid] - a.value != cblock.balance[a.sender.nid]:
            return False
            
    return True


def initLatency(n):         # initializing a 2d array for rho
    global rho
    rho = rng.uniform(10, 500, [n, n])

def computeLatency(peerX, peerY, m):  # computing the latency by taking both nodes and size of message
    if(peerX.speed=="fast" and peerY.speed=="fast"):
            cij = 100 #mb 
    else:
        cij = 5
    dij = random.expovariate(cij/96)#msec
    return rho[peerX.nid][peerY.nid]+m/cij+dij

# a helper function to push an event into the event queue
def pushq(event):
    heapq.heappush(eventq, (event.time, event))
