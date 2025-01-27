# P2P-Blockchain-Simulator
Simulation of a P2P Cryptocurrency Network - CS 765, Spring 2023

# Installation
To extract the file from ubuntu, extract the zip folder using `unzip RollNo1_RollNo2_RollNo3.zip`

To clone from github repo `git clone https://github.com/ankitkmisra/P2P-Blockchain-Simulator.git`

The code for selfish mining and stubborn mining is inside the folders `selfish` and `stubborn`. To run them we need to change our directory into those folders as given below


# Instruction to Run
* Step 1: Open the folder RollNo1_RollNo2_RollNo3 using `cd RollNo1_RollNo2_RollNo3`
	* Step 1.1: To run the selfish mining code open the folder `selfish` as: `cd selfish`
	* Step 1.2: To run the stubborn mining code open the folder `stubborn` as: `cd stubborn`
* Step 2: To install required libraries `pip install networkx numpy matplotlib`
* Step 3: To run as default parameters, use the command as mentioned below `python3 main.py`

The results would get written down inside the figures and logs folders.

To change various parameters usage as mentioned below
* number of nodes in the P2P network: `-n` or `--num_nodes`
* percentage of slow nodes: `-z0` or `--fraction_slow`
* the adversary mining power: `-a` or `--alpha`
* percentage of nodes adversary is connected to: `-zeta` or `--zeta`
* percentage of nodes having low CPU power: `-z1` or `--fraction_lowcpu`
* mean inter-arrival time between transactions: `-ttx` or `--mean_inter_arrival`
* average time taken to mine a block: `-I` or `--average_block_mining_time`
* total time for which the P2P network is simulated: `-T` or `--simulation_time`
* use this flag to save all figures generated in ./figures: `-s` or `--save_figures`


For example `python3 main.py -n 10 -z0 0.5 -z1 0.5 -ttx 10 -I 1000 -T 10000 -s`

To get help regarding any paramenter: `python3 main.py -h` will give the description on usage

# Usage
usage: `python3 main.py [-h] [-n NUM_NODES] [-z0 PERCENTAGE_SLOW] [-z1 PERCENTAGE_LOWCPU] [-a ALPHA] [-zeta ZETA] [-ttx MEAN_INTER_ARRIVAL]
               [-I AVERAGE_BLOCK_MINING_TIME] [-T SIMULATION_TIME] [-s]`
