import random

class Peer():
    def __init__(self, id) -> None:
        self.speed = random.choice([0, 1]) #0->slow, 1->fast
        self.cpu = random.choice([0,1]) #0->low, 1->high
        self.id = id
        self.txns = [] #generated txn+sender+reciever data needed!! {txn:{"sender":id, "reciever":[ids]})} or [[txn, sender, [reciever]]]
        self.neighbors = [] #adjacent peers
        self.coins = 0
        self.blockchain = [] #tree+arrival this is for each node, might vary node to node
    
    def sendTxn(self): #send txn that it has recieved (need to avoid repeated sending and sending to peer whom recieved from)
        pass

    def recieveTxn(self): #add recieved txn into txn queue
        pass

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
    def __init__(self, n) -> None:
        self.peers = [Peer(i) for i in range(n)] #creating peers
        self.graph = [self.peers[i].neighbors for i in range(n)]
        if(not self.isConnected()):
            self.createGraph()

        for p in self.peers:
            connectCount = random.randint(4,8)
            for i in range(connectCount):
                pass
    
    def isConnected() -> bool:
        return False

    def createGraph():
        pass

    def connection(self, peerX, peerY) -> None:
        self.peers[peerX].neighbors.append(peerY)
        self.peers[peerY].neighbors.append(peerX) #connecting 2 peers

    