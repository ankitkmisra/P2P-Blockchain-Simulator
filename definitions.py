import random

class Peer():
    def __init__(self, id) -> None:
        self.speed = random.choice([0, 1]) #0->slow, 1->fast
        self.cpu = random.choice([0,1]) #0->low, 1->high
        self.id = id
        self.txns = [] #generated txn

class Txn():
    #TxnID: IDx pays IDy C coins
    def __init__(self, id, peerX, peerY, coins) -> None:
        self.txnId = id
        self.peerX = peerX
        self.peerY = peerY
        self.coins = coins

    def __repr__(self) -> str:
        return self.txnId+": "+self.peerX+" pays "+self.peerY+" "+self.coins+" coins"
        
class Network():
    def __init__(self) -> None:
        pass

    