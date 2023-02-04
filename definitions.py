import random, numpy as np

Ttx=10
samples = 10
Tk = 3
interarrival = random.expovariate(Ttx)

class Peer():
    def __init__(self, id, speed, cpu) -> None:
        self.speed = speed
        self.cpu = cpu
        self.id = id
        self.txns = [] #seen txn+sender+reciever data needed!! {txn:{"sender":id, "reciever":[ids]})} or [[txn, sender, [reciever]]]
        self.neighbors = [] #adjacent peers
        self.coins = 0
        self.hashpower = 0
        self.blockchain = Blockchain() #init blobkchain with genesis, stores data such as arrival time of a block, block data, the tree, write to file func


class Txn():
    #TxnID: IDx pays IDy C coins
    def __init__(self, id, peerX, peerY, coins, case) -> None:
        self.txnId = id
        self.peerX = peerX
        self.peerY = peerY
        self.coins = coins
        self.case = case
        self.size = 1

    def __repr__(self) -> str:
        if(self.case==1):
            return str(self.txnId)+": "+str(self.peerX)+" pays "+str(self.peerY)+" "+str(self.coins)+" coins"
        if(self.case==2):
           return str(self.txnId)+": "+str(self.peerX)+" mines "+str(self.coins)+" coins" 

#consists of no of peers and peers array
class Network():
    def __init__(self, n, z0, z1) -> None:
        self.n = n #next valid peer ID will be n, ie the no of peers that exist as 0 to n-1 ids are already given
        speed = ["slow" for i in range(int(n*z0))]+["fast" for i in range(n-int(n*z0))]
        cpu = ["low CPU" for i in range(int(n*z1))]+["high CPU" for i in range(n-int(n*z1))]
        np.random.shuffle(speed);np.random.shuffle(cpu)
        self.peers = [Peer(i, speed[i], cpu[i]) for i in range(n)] #creating peers unique ID - 0 to n-1
        graph = [self.peers[i].neighbors for i in range(n)] #temp graph
        temp_tries = 100000
        while(not self.isConnected(graph) or temp_tries<0): #untill connected graph is made or timed out
            graph = self.createGraph()
            temp_tries-=1
            #break
        #print(self.graph)

    def isConnected(self, graph) -> bool:
        visited = [False for i in range(self.n)]
        self.dfs(0, visited, graph)
        #print(visited)
        return np.all(visited, axis=0)
    
    def dfs(self, v, visited, graph):
        visited[v] = True
        for i in graph[v]:
            if visited[i] == False:
                self.dfs(i, visited, graph)
    
    def createGraph(self): #create neighbors 4 to 8 for each peer
        for peerX in range(self.n):
            l = random.randint(4,8) #4 to 8 neighbor length
            while(len(self.peers[peerX].neighbors)<l): 
                self.connection(peerX, random.choice([j for j in range(self.n) if j != peerX and j not in self.peers[peerX].neighbors]))
        return [self.peers[i].neighbors for i in range(self.n)]

    def connection(self, peerX, peerY) -> None: #if x and y not connected then connect
        if(peerY not in self.peers[peerX].neighbors and peerX not in self.peers[peerY].neighbors):
            self.peers[peerX].neighbors.append(peerY)
            self.peers[peerY].neighbors.append(peerX) #connecting 2 peers

class Block():
    def __init__(self, id, peer) -> None:
        self.id = id
        self.peer = peer
        self.size = 1 #kb min when empty
        self.txn = []

class Blockchain():
    def __init__(self) -> None:
        pass

class PoW():
    def __init__(self) -> None:
        pass

class Simulate():
    def __init__(self, n, z0, z1, Ttx) -> None:
        self.net = Network(n, z0, z1) #n and peers
        self.Bqueue = [] #blockqueue sorted by timestamp
        self.eventQ = [] #event queue sorted by timestamp, of txns
        self.pij = [[0 for i in range(n)] for j in range(n)] #0 if self to self prop
        self.txnId=0 #start with 0
        self.time=0 #at simulation start time=0
        self.n = n
        for i in range(n): #prop delay
            for j in range(i+1, n):
                self.pij[i][j] = random.uniform(10, 500)*0.001 #s
                self.pij[j][i] = random.uniform(10, 500)*0.001 #s

    def isValidPeer(self, x) -> bool: #can be in the class calling network
        if(x>=0 and x<self.net.n and self.net.peers[x]):
            return True
        return False

    #call at interarrival time////need to simulate peerY and coins randomly
    #2 types of txn event: generation and forwarded
    #loopless how??
    def actionTxn(self, event):
        if(event[4]=="gen"): #generate single txn at random time by a peer 
            _, peerX, peerY, coins, _ = event
            if(self.isValidPeer(peerX) and self.net.peers[peerX].coins>=coins and self.isValidPeer(peerY)):
                txn = Txn(self.txnId, peerX, peerY, coins, 1) #txn generated to put into recieve queue
                self.txnId+=1
                self.net.peers[peerX].coins-=coins #sender
                self.net.peers[peerY].coins+=coins #reciever
                for i in self.net.peers[peerX].neighbors: #sending my generated txn to my neighbors
                    self.eventQ.append([self.time+self.latency(peerX, i, len(repr(txn))), peerX, i, txn, "fowd"]) #should it be len(msg) or 1kb of txn???
                
                #next txn to be generated
                peerY = random.randint(0, self.n)
                coins = random.uniform(0, self.net.peers[peerX].coins) #coz float
                self.eventQ.append([self.time, peerX, peerY, coins, "gen"]) #txn to send
                self.sortQ()
        elif(event[4]=="recv"):
            pass #make sure not to send back or send again
    
    def actionBlk(self, event):
        if(event[4]=="gen"): #generate single txn at random time by a peer 
            pass
        elif(event[4]=="recv"):
            pass
        
    def sortQ(self):
        self.eventQ = sorted(self.eventQ)
        print(self.eventQ)

    def latency(self, peerX, peerY, m): # hold for time t before sending a block -> representing network latency
        if(self.net.peers[peerX].speed=="fast" and self.net.peers[peerY].speed=="fast"):
            cij = 100*1000 #kb 
        else:
            cij = 5000
        dij = random.expovariate(96/cij)
        return self.pij[peerX][peerY]+m/cij+dij

sim = Simulate(10, 0.5, 0.5, 10)
sim.actionTxn([0,1,0,0,"gen"])
#print(np.average([sim.latency(0, 1, 300000) for i in range(1000)]))