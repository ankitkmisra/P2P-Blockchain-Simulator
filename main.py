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
from utils import blkIdGen, txnIdGen, initLatency, eventq, get_blocks

#The discrete event simulator
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
        #building the connection graph of peers
        self.G = nx.Graph()
        self.G.add_nodes_from(range(n))

        #genesis block
        self.genesis = Block(pb=0, bid=1, txnIncluded=set(), miner=None, balance = [0]*n)

        #id generation
        self.blkid_generator = blkIdGen()
        self.txnid_generator = txnIdGen()
        
        #speed and cpu as mentioned in the question
        speed = ["slow" for i in range(int(n*z0))]+["fast" for i in range(n-int(n*z0))]
        cpu = ["low" for i in range(int(n*z1))]+["high" for i in range(n-int(n*z1))]
        rng.shuffle(speed)
        rng.shuffle(cpu)

        #hashing power
        invh0 = n*(10 - 9*z1)
        invh1 = invh0/10
        miningTime = [I*invh0 if cpu[i] == "low" else I*invh1 for i in range(n)]
        
        self.nodes = [None]*n
        for i in range(n):
            self.nodes[i] = Node(nid=i, speed=speed[i], cpu=cpu[i],
                                 genesis=self.genesis, miningTime=miningTime[i],
                                 blkgen=self.blkid_generator,
                                 txngen=self.txnid_generator)

        self.ttx = ttx
        self.I = I
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
                    nodeY = rng.choice([j for j in range(n) if j != nodeX and j not in self.nodes[nodeX].peers]) 
                    if nodeY != nodeX:
                        self.connection(nodeX, nodeY)
                #print(len(self.nodes[nodeX].peers))
            #print(self.G.edges)

    #adding edges of the peer graph
    def connection(self, nodeX, nodeY): #if x and y not connected then connect
        if(nodeY not in self.nodes[nodeX].peers and nodeX not in self.nodes[nodeY].peers):
            self.G.add_edge(nodeX, nodeY)
            self.nodes[nodeX].peers.add(self.nodes[nodeY])
            self.nodes[nodeY].peers.add(self.nodes[nodeX])

    # initializing events and pushing to the queue - block and txn
    def gen_all_txn(self, max_time): #generate event
        for p in self.nodes:
            minetime = rng.exponential(p.miningTime)
            block2mine = Block( #first mined block
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
            # adding block event to the queue
            heapq.heappush(eventq, (minetime, Event(minetime, "BlockMined", block=block2mine)))
            
            # generate txn and add to the queue
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

    #running the simulation, running events from the queue and writing to file
    def run(self, untill): #simulate untill
        time = 0
        while(time < untill and len(eventq) > 0):
            time, event = heapq.heappop(eventq)
            self.handle(event)
        while(len(eventq) > 0):
            time, event = heapq.heappop(eventq)
            if event.event_type in ["TxnRecv", "BlockRecv"]:
                self.handle(event)
        
        for i in self.nodes: #each node
            file = open(f'./logs/log_tree_{i.nid}.txt', 'w+') #store in file
            heading = f'Data For Node Id: {i.nid}\n'
            file.write(heading)
            for _, block in i.blockChain.items(): #each block
                parent = None
                miner = None
                if block.pb != 0: #if parent and miner exists
                    parent = block.pb.bid
                    miner = block.miner.nid
                log_to_write = f"Block Id:{block.bid}, Parent ID:{parent}, Miner ID:{miner}, Txns:{len(block.txnIncluded)}, Time:{block.time}\n"
                file.write(log_to_write)
            file.close()
            
    #just a switch case to handle events
    def handle(self, event): #event handler 
        if event.event_type == "TxnGen":
            event.txn.peerX.txnSend(event)
        elif event.event_type == "TxnRecv":
            event.receiver.txnRecv(event)
        elif event.event_type == "BlockRecv":
            event.receiver.verifyAndAddReceivedBlock(event)
        elif event.event_type == "BlockMined":
            event.block.miner.receiveSelfMinedBlock(event)

    #plot
    def draw_bc(self, nid, save=False):
        plt.figure()
        # draw network with Kamada Kawai layout
        nx.draw(self.nodes[nid].g, pos=nx.kamada_kawai_layout(self.nodes[nid].g), node_size=10, node_color='red')
        if save:
            plt.savefig(f'./figures/blockchain_{nid}.png')
        else:
            plt.show()

    #save plot
    def print_graph(self, save=False): #print graph     
        plt.figure()
        nx.draw(self.G, with_labels=True)
        if save:
            plt.savefig('./figures/network_graph.png')
        else:
            plt.show()

    def print_stats(self):
        nd = self.nodes[0]
        genesis = 1

        g_rev = nd.g.reverse()
        succ = nx.dfs_successors(g_rev, source=genesis)
        depth_from_root = {}
        max_depth = {}
        parent = {}
        deepest_node = genesis

        def dfs(u, par = None, dep = 0):
            nonlocal g_rev, succ, depth_from_root, max_depth, parent, deepest_node
            depth_from_root[u] = dep
            max_depth[u] = dep
            parent[u] = par
            if dep > depth_from_root[deepest_node]:
                deepest_node = u
            if u not in succ:
                return
            for v in succ[u]:
                dfs(v, u, dep + 1)
                max_depth[u] = max(max_depth[u], max_depth[v])
        dfs(genesis)

        branches = []
        while deepest_node != genesis:
            child = deepest_node
            deepest_node = parent[deepest_node]
            for u in succ[deepest_node]:
                if u != child:
                    branches.append(max_depth[u] - depth_from_root[deepest_node])

        node_type_successful = {}
        node_type_blocks_mined = {}
        for type in ['slow_low', 'slow_high', 'fast_low', 'fast_high']:
            node_type_successful[type] = 0
            node_type_blocks_mined[type] = 0
        mined_in_longest_chain = {}
        for node in self.nodes:
            mined_in_longest_chain[node.nid] = 0
            node_type_blocks_mined[node.speed + '_' + node.cpu] += node.created_blocks_own
        block = nd.blockChain[nd.lbid]
        while block.bid != genesis:
            mined_in_longest_chain[block.miner.nid] += 1
            node_type_successful[block.miner.speed + '_' + block.miner.cpu] += 1
            block = block.pb

        print("Length of longest chain (including genesis block):", nd.blockChain[nd.lbid].length)
        print("Total number of blocks mined:", get_blocks())
        print("Fraction of mined blocks present in longest chain:", round((nd.blockChain[nd.lbid].length-1) / get_blocks(), 3))
        print()
        for type in ['slow_low', 'slow_high', 'fast_low', 'fast_high']:
            print(f"% blocks in longest chain mined by {type} node:", round(node_type_successful[type] / (nd.blockChain[nd.lbid].length-1), 2))
        print()
        for type in ['slow_low', 'slow_high', 'fast_low', 'fast_high']:
            if node_type_blocks_mined[type] == 0:
                print(f"% blocks mined by {type} node that made it to longest chain: 0.0")
            else:
                print(f"% blocks mined by {type} node that made it to longest chain:", round(node_type_successful[type] / node_type_blocks_mined[type], 2))
        print()
        if len(branches) > 0:
            print("Lengths of branches:", branches)
            print("Average length of branch:", round(np.average(branches), 3))
        else:
            print("No branches were formed!")


if __name__ == "__main__":
    #parse args
    parser = argparse.ArgumentParser(description='a P2P network blockchain simulator')
    parser.add_argument('-n', '--num_nodes', default=10, type=int, help='number of nodes in the P2P network')
    parser.add_argument('-z0', '--fraction_slow', default=0.5, type=float, help='fraction of slow nodes')
    parser.add_argument('-z1', '--fraction_lowcpu', default=0.5, type=float, help='fraction of nodes having low CPU power')
    parser.add_argument('-ttx', '--mean_inter_arrival', default=10, type=float, help='mean inter-arrival time between transactions')
    parser.add_argument('-I', '--average_block_mining_time', default=100000, type=float, help='average time taken to mine a block')
    parser.add_argument('-T', '--simulation_time', default=1000000, type=float, help='total time for which the P2P network is simulated')
    parser.add_argument('-s', '--save_figures', default=False, action='store_true', help='use this flag to save all figures generated in ./figures')

    args = parser.parse_args()

    num_nodes = args.num_nodes
    percentage_slow = args.fraction_slow
    percentage_lowcpu = args.fraction_lowcpu
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
    simulator.print_stats()

    # draw blockchain
    for i in range(num_nodes):
        simulator.draw_bc(i, save=save_figures)