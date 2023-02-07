import heapq
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
    pblock = cblock.pbid
    for a in cblock.txnIncluded:
        cb = pblock.balance[a.receiver.nid] + a.value
        if cb != cblock.balance[a.receiver.nid]:
            return False
        if a.sender == -1:
            continue
        sb = pblock.balance[a.sender.nid] - a.value
        if sb < 0:
            return False
        if sb != cblock.balance[a.sender.nid]:
            return False
    return True


def initLatency(n):         # initializing a 2d array for rho
    global rho
    rho = rng.uniform(10, 500, [n, n])

def computeLatency(i, j, m):  # computing the latency by taking both nodes and size of message
    lat = rho[i.nid][j.nid]
    c = 5.0
    if i.speed == 1 and j.speed == 1:
        c = 100.0
    lat += m/c + rng.exponential(96.0/c)
    return lat

# a helper function to push an event into the event queue
def pushq(event):
    heapq.heappush(eventq, (event.time, event))
