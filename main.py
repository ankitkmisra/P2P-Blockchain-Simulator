import heapq
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.random import default_rng

from utils.definitions import Node, Block, Event, Transaction
from utils.utils import blkIdGen, txnIdGen, initLatency, eventq

rng = default_rng(42)

class Simulation:
    def __init__(self, n, ttx, z0, z1, I=600):
        """
        Initializes the simulation object
            n: Number of Nodes
            ttx: Mean inter-arrival time of transactions
            z0: Percentage of slow nodes
            z1: Percentage of low CPU nodes
            I: Average block mining time
        """
        self.G = nx.Graph()
        self.G.add_nodes_from(range(n))

        self.gblock = Block(pbid=0, bid=1, txnIncluded=set(), miner=-1)
        self.gblock.balance = [0]*n

        self.blkid_generator = blkIdGen()
        self.txnid_generator = txnIdGen()
        
        speed = [0]*int(n*z0) + [1]*(n - int(n*z0))
        cpu = [0]*int(n*z1) + [1]*(n - int(n*z1))
        np.random.shuffle(speed)
        np.random.shuffle(cpu)

        invh0 = n*(10 - 9*z1)
        invh1 = invh0/10
        miningTime = [I*invh0 if cpu[i] == 0 else I*invh1 for i in range(n)]

        print(miningTime)
        
        self.nodes = [None]*n
        for i in range(n):
            self.nodes[i] = Node(nid=i, speed=speed[i], cpu=cpu[i],
                                 genesis=self.gblock, miningTime=miningTime[i],
                                 blkgen=self.blkid_generator,
                                 txngen=self.txnid_generator)

        self.ttx = ttx
        initLatency(n)

    def generate_network(self):
        """
        Generates an appropriate network by connecting nodes
        """
        n = len(self.nodes)
        while not nx.is_connected(self.G):
            self.G = nx.Graph()
            self.G.add_nodes_from(range(n))
            # reset nodes
            for node in self.nodes:
                node.peers = set()
            # generate random connections
            for nodeX in range(n):
                l = random.randint(4, 8)
                while len(self.nodes[nodeX].peers) < l:
                    nodeY = random.randint(0, n-1)
                    if nodeY != nodeX:
                        self.connection(nodeX, nodeY)
            print(self.G.edges)

    def connection(self, nodeX, nodeY): #if x and y not connected then connect
        if(nodeY not in self.nodes[nodeX].peers and nodeX not in self.nodes[nodeY].peers):
            self.G.add_edge(nodeX, nodeY)
            self.nodes[nodeX].peers.add(self.nodes[nodeY])
            self.nodes[nodeY].peers.add(self.nodes[nodeX])

    def print_graph(self):        
        """
        print the graph to visualize the network
        """
        nx.draw(self.G)
        plt.show()

    def gen_all_txn(self, max_time):
        """
        Add all transaction events and start mining on the genesis
        block in the simulation queue (including only the coinbase txn)
        """
        for p in self.nodes:
            # generate block on genesis
            minetime = rng.exponential(p.miningTime)
            block2mine = Block(
                pbid=self.gblock,
                bid=next(self.blkid_generator),
                txnIncluded=set([Transaction(
                    sender=-1,
                    tid=next(self.txnid_generator),
                    receiver=p,
                    value=50
                )]),
                miner=p
            )
            # add mining finish times to event queue
            heapq.heappush(eventq, (minetime, Event(minetime, "BlockMined", block=block2mine)))
            
            # generate transactions based on the mean interarrival time
            t = rng.exponential(self.ttx)
            while(t < max_time):
                elem = Transaction(
                    sender=p,
                    tid = next(self.txnid_generator),
                    receiver = self.nodes[np.random.randint(0,len(self.nodes))],
                    value = 0
                )
                heapq.heappush(eventq, (t, Event(t, "TxnGen", txn=elem)))
                t = t + rng.exponential(self.ttx)

    def run(self, max_time):
        """
        Runs simulation for `max_time`
        """
        t = 0
        while(t < max_time and len(eventq)!=0):
            t, event = heapq.heappop(eventq)
            self.handle(event)
        
        #! make a new folder
        #! and separate the log trees for each node
        file=open("log_tree.txt","w+")
        for a in self.nodes:
            heading="*"*100+f"Id:{a.nid}"+"*"*100+"\n"
            file.write(heading)
            for _,block in a.blockChain.items():
                if block.pbid == 0: 
                    log_to_write=f"Id:{block.bid},Parent:{-1}, Miner:{block.miner}, Txns:{len(block.txnIncluded)}, Time:{block.time}\n"
                else:
                    log_to_write=f"Id:{block.bid},Parent:{block.pbid.bid}, Miner:{block.miner}, Txns:{len(block.txnIncluded)}, Time:{block.time}\n"
                file.write(log_to_write)
            

    def handle(self, event):
        """
        Calls the appropriate event handler for latest event
        """
        if event.event_type == "TxnGen":
            event.txn.sender.txnSend(event)
        elif event.event_type == "TxnRecv":
            event.receiver.txnRecv(event)
        elif event.event_type == "BlockRecv":
            event.receiver.verifyAndAddReceivedBlock(event)
        elif event.event_type == "BlockMined":
            event.block.miner.receiveSelfMinedBlock(event)
        else:
            print("bug in simulation.handle()")

    def draw_bc(self, nid):
        # draw network with planar layout
        nx.draw(self.nodes[nid].g, pos=nx.planar_layout(self.nodes[nid].g), node_size=10, node_color='red')
        # nx.draw(self.nodes[nid].g, node_size=10, node_color='red')
        plt.show()
            

if __name__ == "__main__":
    num_nodes=10
    percentage_slow=0.5
    percentage_lowcpu=0.5 
    mean_inter_arrival=10
    average_block_mining_time=600

    simulation_time=10000
    simulator = Simulation(num_nodes, mean_inter_arrival, percentage_slow,
                           percentage_lowcpu, average_block_mining_time)
    
    simulator.generate_network()
    simulator.print_graph()
    simulator.gen_all_txn(simulation_time)
    simulator.run(simulation_time)

    # draw blockchain
    for i in range(num_nodes):
        simulator.draw_bc(i)