#!/bin/bash

ping 10.128.0.8 -c 2
ping 10.128.0.16 -c 2
ping 10.128.0.11 -c 2
#sudo scp -i /home/ayush/.ssh/id_rsa test.txt ayush@10.128.0.16:/home/ayush/ADS/test
wget https://ars.els-cdn.com/content/image/1-s2.0-S1055790306000893-mmc2.txt
