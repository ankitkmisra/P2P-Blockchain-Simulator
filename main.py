import os
import heapq
import random
import shutil
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils.seed import rng, seeds
np.random.seed(seeds["np.random.seed"])
random.seed(seeds["random.seed"])

from utils.definitions import Node, Block, Event, Transaction
from utils import blkIdGen, txnIdGen, initLatency, eventq


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

        self.genesis = Block(pb=0, bid=1, txnIncluded=set(), miner=None, balance = [0]*n)

        self.blkid_generator = blkIdGen()
        self.txnid_generator = txnIdGen()
        
        speed = ["slow" for i in range(int(n*z0))]+["fast" for i in range(n-int(n*z0))]
        cpu = ["low CPU" for i in range(int(n*z1))]+["high CPU" for i in range(n-int(n*z1))]
        rng.shuffle(speed)
        rng.shuffle(cpu)

        #hashing power
        invh0 = n*(10 - 9*z1)
        invh1 = invh0/10
        miningTime = [I*invh0 if cpu[i] == "low CPU" else I*invh1 for i in range(n)]

        print(miningTime)
        
        self.nodes = [None]*n
        for i in range(n):
            self.nodes[i] = Node(nid=i, speed=speed[i], cpu=cpu[i],
                                 genesis=self.genesis, miningTime=miningTime[i],
                                 blkgen=self.blkid_generator,
                                 txngen=self.txnid_generator)

        self.ttx = ttx
        initLatency(n)

    def generate_network(self): #p2p network connection
        n = len(self.nodes)
        while not nx.is_connected(self.G):
            self.G = nx.Graph()
            self.G.add_nodes_from(range(n))
            # reset nodes
            for node in self.nodes:
                node.peers = set()
            # generate random connections
            for nodeX in range(n):
                l = rng.integers(4, 9)
                #print(l)
                while len(self.nodes[nodeX].peers) < l:
                    nodes_choice = [j for j in range(n) if j != nodeX and j not in self.nodes[nodeX].peers] 
                    rng.shuffle(nodes_choice)
                    nodeY = nodes_choice[0]
                    if nodeY != nodeX:
                        self.connection(nodeX, nodeY)
                #print(len(self.nodes[nodeX].peers))
            #print(self.G.edges)

    def connection(self, nodeX, nodeY): #if x and y not connected then connect
        if(nodeY not in self.nodes[nodeX].peers and nodeX not in self.nodes[nodeY].peers):
            self.G.add_edge(nodeX, nodeY)
            self.nodes[nodeX].peers.add(self.nodes[nodeY])
            self.nodes[nodeY].peers.add(self.nodes[nodeX])

    def gen_all_txn(self, max_time): #generate event
        for p in self.nodes:
            minetime = rng.exponential(p.miningTime)
            block2mine = Block( #genesis block
                pb=self.genesis,
                bid=next(self.blkid_generator),
                txnIncluded=set([Transaction(
                    peerX=-1,
                    id=next(self.txnid_generator),
                    peerY=p,
                    value=50,
                    case=2
                )]),
                miner=p
            )
            # add mining finish times to event queue
            heapq.heappush(eventq, (minetime, Event(minetime, "BlockMined", block=block2mine)))
            
            # generate transactions based on the mean interarrival time
            t = rng.exponential(self.ttx)
            while(t < max_time):
                elem = Transaction(
                    peerX=p,
                    id = next(self.txnid_generator),
                    peerY = self.nodes[rng.integers(0, len(self.nodes))],
                    value = 0, case=1
                )
                heapq.heappush(eventq, (t, Event(t, "TxnGen", txn=elem)))
                t = t + rng.exponential(self.ttx)

    def run(self, untill): #simulate untill
        time = 0
        while(time < untill and len(eventq) > 0):
            time, event = heapq.heappop(eventq)
            self.handle(event)
        
        for i in self.nodes: #each node
            file = open(f'./logs/log_tree_{i.nid}.txt', 'w+') #store in file
            heading = f'Data For Node Id: {i.nid}\n'
            file.write(heading)
            for _, block in i.blockChain.items(): #each block
                if block.pb == 0: #genesis
                    log_to_write = f"Block Id:{block.bid}, Parent ID:{None}, Miner ID:{None}, Txns:{len(block.txnIncluded)}, Time:{block.time}\n"
                else:
                    log_to_write = f"Block Id:{block.bid}, Parent ID:{block.pb.bid}, Miner ID:{block.miner.nid}, Txns:{len(block.txnIncluded)}, Time:{block.time}\n"
                file.write(log_to_write)
            file.close()
            

    def handle(self, event): #event handler - push to queue
        if event.event_type == "TxnGen":
            event.txn.peerX.txnSend(event)
        elif event.event_type == "TxnRecv":
            event.receiver.txnRecv(event)
        elif event.event_type == "BlockRecv":
            event.receiver.verifyAndAddReceivedBlock(event)
        elif event.event_type == "BlockMined":
            event.block.miner.receiveSelfMinedBlock(event)

    def draw_bc(self, nid, save=False):
        plt.figure()
        # draw network with Kamada Kawai layout
        nx.draw(self.nodes[nid].g, with_labels=True, pos=nx.kamada_kawai_layout(self.nodes[nid].g), node_size=10, node_color='red')
        if save:
            plt.savefig(f'./figures/blockchain_{nid}.png')
        else:
            plt.show()

    def print_graph(self, save=False): #print graph     
        plt.figure()
        nx.draw(self.G, with_labels=True)
        if save:
            plt.savefig('./figures/network_graph.png')
        else:
            plt.show()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='a P2P network blockchain simulator')
    parser.add_argument('-n', '--num_nodes', default=10, type=int, help='number of nodes in the P2P network')
    parser.add_argument('-z0', '--percentage_slow', default=0.5, type=float, help='percentage of slow nodes')
    parser.add_argument('-z1', '--percentage_lowcpu', default=0.5, type=float, help='percentage of nodes having low CPU power')
    parser.add_argument('-ttx', '--mean_inter_arrival', default=10, type=float, help='mean inter-arrival time between transactions')
    parser.add_argument('-I', '--average_block_mining_time', default=600, type=float, help='average time taken to mine a block')
    parser.add_argument('-T', '--simulation_time', default=10000, type=float, help='total time for which the P2P network is simulated')
    parser.add_argument('-s', '--save_figures', default=False, action='store_true', help='use this flag to save all figures generated in ./figures')

    args = parser.parse_args()

    num_nodes = args.num_nodes
    percentage_slow = args.percentage_slow
    percentage_lowcpu = args.percentage_lowcpu
    mean_inter_arrival = args.mean_inter_arrival
    average_block_mining_time = args.average_block_mining_time
    simulation_time = args.simulation_time
    save_figures = args.save_figures

    if os.path.exists('./logs'):
        shutil.rmtree('./logs')
    os.mkdir('./logs')

    if save_figures:
        if os.path.exists('./figures'):
            shutil.rmtree('./figures')
        os.mkdir('./figures')

    simulator = Simulation(num_nodes, mean_inter_arrival, percentage_slow,
                           percentage_lowcpu, average_block_mining_time)
    
    simulator.generate_network()
    simulator.print_graph(save=save_figures)
    simulator.gen_all_txn(simulation_time)
    simulator.run(simulation_time)

    # draw blockchain
    for i in range(num_nodes):
        simulator.draw_bc(i, save=save_figures)