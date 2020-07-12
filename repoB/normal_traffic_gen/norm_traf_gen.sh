#!/bin/bash
#Author :Ayushi
ip1=$1
ip2=$2
#ip3=$3
ping $ip1 -c 2
ping $ip2 -c 2
#ping $ip3 -c 2
#sudo scp -i /home/ayush/.ssh/id_rsa test.txt ayush@10.128.0.16:/home/ayush/ADS/test
wget https://ars.els-cdn.com/content/image/1-s2.0-S1055790306000893-mmc2.txt
