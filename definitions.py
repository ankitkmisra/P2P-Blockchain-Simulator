import random, numpy as np

class Peer():
    def __init__(self, id, speed, cpu) -> None:
        self.speed = speed
        self.cpu = cpu
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
        self.sizeKB = 1

    def __repr__(self) -> str:
        return self.txnId+": "+self.peerX+" pays "+self.peerY+" "+self.coins+" coins"
        
class Network():
    def __init__(self, n, speed, cpu) -> None:
        self.n = n #next valid peer ID will be n, ie the no of peers that exist as 0 to n-1 ids are already given
        self.peers = [Peer(i, speed[i], cpu[i]) for i in range(n)] #creating peers unique ID - 0 to n-1
        self.graph = [self.peers[i].neighbors for i in range(n)]
        temp_tries = 100000
        while(not self.isConnected() or temp_tries<0): #untill connected graph is made or timed out
            self.createGraph()
            temp_tries-=1
            #break

        for p in self.peers:
            connectCount = random.randint(4,8)
            for i in range(connectCount):
                pass
    
    def isValidPeer(self, x) -> bool:
        if(x>=0 and x<self.n and self.peers[x]):
            return True
        return False

    def generateTxn(self, peerX, peerY, coins): #generate single txn at random time by a peer 
        if(self.isValidPeer(peerX) and self.peers[peerX].coins>coins and self.isValidPeer(peerY)):
            return Txn(peerX, peerY, coins)

    def isConnected(self) -> bool:
        visited = [False for i in range(self.n)]
        self.dfs(0, visited)
        #print(visited)
        return np.all(visited, axis=0)
    
    def dfs(self, v, visited):
        visited[v] = True
        for i in self.graph[v]:
            if visited[i] == False:
                self.dfs(i, visited)

    
    def createGraph(self): #create neighbors 4 to 8 for each peer
        for peerX in range(self.n):
            l = random.randint(4,8) #4 to 8 neighbor length
            while(len(self.peers[peerX].neighbors)<l): 
                self.connection(peerX, random.choice([j for j in range(self.n) if j != peerX and j not in self.peers[peerX].neighbors]))
        self.graph = [self.peers[i].neighbors for i in range(self.n)]


    def connection(self, peerX, peerY) -> None: #if x and y not connected then connect
        if(peerY not in self.peers[peerX].neighbors and peerX not in self.peers[peerY].neighbors):
            self.peers[peerX].neighbors.append(peerY)
            self.peers[peerY].neighbors.append(peerX) #connecting 2 peers

net = Network(100, [1 for i in range(100)], [1 for i in range(100)])

