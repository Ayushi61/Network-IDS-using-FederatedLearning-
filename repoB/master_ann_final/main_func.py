'''
Author:Ayushi
'''
from data_ana import KDD_data_ana
from eval import run_train_test
import getopt
import os
import sys

def main():
    argv=sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hw:p:i:", ["workers=", "ports=","ids="])
    except getopt.GetoptError:
        print( 'fede.py -w <worker1,worker2> -p <port1,port2> -i <id1,id2>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print( 'fede.py -w <worker1,worker2> -p <port1,port2> -i <id1,id2>')
            sys.exit()
        elif opt in ("-w", "--workers"):
            workers = arg.split(",")
        elif opt in ("-p", "--ports"):
            ports = arg.split(",")
        elif opt in ("-i", "--ids"):
            ids = arg.split(",")
    X_train, X_test, y_train, y_test,encoder=KDD_data_ana()
    run_train_test(workers,ports,ids,X_train, X_test, y_train, y_test,encoder)
    


if __name__=="__main__":
    main()
