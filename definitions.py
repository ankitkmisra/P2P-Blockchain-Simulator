import random, numpy as np

class Peer():
    def __init__(self, id, speed, cpu) -> None:
        self.speed = speed
        self.cpu = cpu
        self.id = id
        self.txns = [] #seen txn+sender+reciever data needed!! {txn:{"sender":id, "reciever":[ids]})} or [[txn, sender, [reciever]]]
        self.neighbors = [] #adjacent peers
        self.coins = 0
        self.hashpower = 0
        self.blockchain = self.longestChain() #init blobkchain with genesis, stores data such as arrival time of a block, block data, the tree, write to file func
        self.tree = []
    
    def longestChain(self):#how will we get the blockchain here? extend the self.blockchain here
        pass
    
#txn and coinbase txn both
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
    def __init__(self, id, peer, prev, txn, createdAt, coinbase) -> None:
        self.id = id
        self.peer = peer
        self.size = 1 #kb min when empty
        self.txn = txn
        self.coins = 0
        self.coinbase = coinbase
        self.createdAt = createdAt
        self.prev = prev #prev block

class PoW():
    def __init__(self) -> None:
        pass

class Simulate():
    def __init__(self, n, z0, z1, Ttx, I, N) -> None:
        self.net = Network(n, z0, z1) #n and peers
        self.eventQ = [] #event queue sorted by timestamp, of txns
        self.pij = [[0 for i in range(n)] for j in range(n)] #0 if self to self prop
        self.txnId=0 #start with 0
        self.blkId=0 #start with 0
        self.time=0.0 #at simulation start time=0
        self.n = n
        self.Ttx = Ttx
        self.I = I
        self.N = N
        H=0
        for i in range(n): #prop delay
            for j in range(i+1, n):
                self.pij[i][j] = random.uniform(10, 500)*0.001 #s
                self.pij[j][i] = random.uniform(10, 500)*0.001 #s
            if(self.net.peers[i].cpu=="high CPU"):
                self.net.peers[i].hashpower = 10;H+=10
            else:
                self.net.peers[i].hashpower = 10;H+=1
        for i in range(n):
            self.net.peers[i].hashpower/=H
        self.seedEvent()
        self.runEvent()

    def isValidPeer(self, x) -> bool: #can be in the class calling network
        if(x>=0 and x<self.net.n and self.net.peers[x]):
            return True
        return False

    #4 types of event: generation and forwarded for blocks and transactions
    def pushEvent(self, event):
        if(event[4]=="txnGen"): #generate single txn at random time by a peer 
            time, peerX, peerY, coins, _ = event #array of items
            #global track of coins here but if we keep it block wise the validation will be consistant with fork
            if(self.isValidPeer(peerX) and self.net.peers[peerX].coins>=coins and self.isValidPeer(peerY)):
                txn = Txn(self.txnId, peerX, peerY, coins, 1) #txn generated to put into recieve queue
                self.txnId+=1
                for i in self.net.peers[peerX].neighbors: #sending my generated txn to my neighbors
                    self.eventQ.append([time+self.latency(peerX, i, 1024), peerX, i, txn, "txnFowd"]) #size of txn is 1kb hence m=1kb
                
                #next txn to be generated
                peerY = random.randint(0, self.n-1)
                coins = random.uniform(0, self.net.peers[peerX].coins) #coz float
                self.eventQ.append([time+random.expovariate(1/self.Ttx), peerX, peerY, coins, "txnGen"]) #txn to send Ttx

        elif(event[4]=="txnFowd"):
            time, peerX, peerY, txn, _ = event
            for i in self.net.peers[peerX].neighbors: #sending my generated txn to my neighbors
                    if(i!=peerY):
                        self.eventQ.append([time+self.latency(peerX, i, 1024), peerX, i, txn, "txnFowd"]) #size of txn is 1kb hence m=1kb
            pass #make sure not to send back or send again - loopless

        elif(event[4]=="blkGen"):
            time, peerX, prev, txn, _ = event #items needed add here
            coinbase = Txn(self.txnId, peerX, None, 50, 2)
            self.txnId+=1
            blk = Block(self.blkId, peerX, prev, txn, time, coinbase) #txn generated to put into recieve queue
            self.blkId+=1
            for i in self.net.peers[peerX].neighbors: #sending my generated txn to my neighbors
                self.eventQ.append([time+self.latency(peerX, i, 1024), peerX, i, blk, "blkFowd"]) #size of txn is 1kb hence m=1kb
            
            #next txn to be generated
            Tk = I/self.net.peers[peerX].hashpower
            self.eventQ.append([time+random.expovariate(1/Tk), peerX, blk, txn, "blkGen"]) #txn to send

        elif(event[4]=="blkFowd"):
            time, peerX, peerY, blk, _ = event
            for i in self.net.peers[peerX].neighbors: #sending my generated txn to my neighbors
                    if(i!=peerY):
                        self.eventQ.append([time+self.latency(peerX, i, 1024), peerX, i, blk, "blkFowd"]) #size of txn is 1kb hence m=1kb
            pass

    def runEvent(self):
        count = 1
        self.sortQ()
        
        while count <= self.N:
            event = self.eventQ.pop(0)
            self.time = event[0]
            print(event)
            self.pushEvent(event)
            self.sortQ()
            count += 1
        pass

    def seedEvent(self):
        peerX = random.randint(0,self.n-1)
        peerY = random.randint(0, self.n-1)
        coins = random.uniform(0, self.net.peers[peerX].coins)
        event1 = [self.time, peerX, peerY, coins, "txnGen"]

        genesis = Block(self.blkId, peerX, 0, None, self.time, None)
        event2 = [self.time, peerX, genesis, None, "blkGen"]
        self.eventQ += [event1, event2]
        pass
    
    def sortQ(self):
        self.eventQ = sorted(self.eventQ, key=lambda x:x[0])

    def latency(self, peerX, peerY, m): # hold for time t before sending a block -> representing network latency
        if(self.net.peers[peerX].speed=="fast" and self.net.peers[peerY].speed=="fast"):
            cij = 100*1000 #kb 
        else:
            cij = 5000
        dij = random.expovariate(cij/96)
        return self.pij[peerX][peerY]+m/cij+dij


Ttx=0.1 #interarrival time of txns
peers = 10
I = 1 #interarrival time of blocks
z0=0.5
z1=0.5
N = 100 #no of events

sim = Simulate(peers, z0, z1, Ttx, I, N)
#sim.pushEvent([0,1,0,0,"txnGen"])
#print(np.average([sim.latency(0, 1, 300000) for i in range(1000)]))


#how to make sure blockchain is within peer ie blocks arrival time is for particular peer and not centralised
#make sure simulation is in perspective of each peer and not global

#todo
"""
loopless block transmit
block execution point 7 properly
utxo per block event to be maintained
visualisation tree yada yada
report overleaf :')
chain adding to peer and writing that to file
adding txn in block and its limit
broadcast condition on block
fork resolution - longest chain?
around 45 done 11 remaining (+ report 44)
"""