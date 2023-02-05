from definitions import Simulate
import sys

def main():
    args = sys.argv[1:] #arguments
    peers, z0, z1, Ttx, I, N = args
    f = open("output.txt", "w")
    Simulate(int(peers), float(z0), float(z1), float(Ttx), float(I), int(N))
    print("bye!")
    f.write("bye!")
    f.close()

if __name__ == "__main__":
    main()
