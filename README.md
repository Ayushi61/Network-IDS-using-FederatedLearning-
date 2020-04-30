# ADS
Intrusion detection 

Setup- 
instance-1 - 10.128.0.2  ---> master
instance-2 - 10.128.0.3  ---> worker1
Instance-3 - 10.128.0.4 ---> worker2 
instance-1 - 10.128.0.2  --- > attacker


 Modules in setup 

instance-1 - 10.128.0.2  ---> master  ---- > 
-----------------------###########################
trains data   
			-- fede.py 
			--BENZIL
-----------------------###########################


instance-2 - 10.128.0.3  ---> worker1 ---- > 
-----------------------###########################
runs websocket server
			--run websocker.py
			--AYUSHI
collects live data and converts it to expected format -- similar to database,  
			--auto_dataset_conv.py -- 
			--AYUSHI
generates normal traffic,
			-- norm_traf_gen.py
			-- BENZIL
makes predictions
			-- make.py
			-- AYUSHI AND BENZIL
-----------------------###########################


Instance-3 - 10.128.0.4 ---> worker2 ---- > runs websocket server, collects live data,  generates normal traffic,  makes predictions


instance-1 - 10.128.0.2  --- > attacker --- >  
-----------------------###########################
ddos attack scripts
			--port_scan.py
			--sock_syn.py
			-- AYUSHI
-----------------------###########################			








Training:
instance-3, 5 and 6
step1 run websocket on instance 3 and 6 
	instance3   python3 run_websocket_server.py --host 10.128.0.3 --port 8777 --id alice
	instance6	python3 run_websocket_server.py --host 10.128.0.4 --port 8778 --id bob
step2 run fede.py on instance 5 
			python3 fede.py
			

	

Testing normal traffic -- 
Instance- 3, and 6
step0: start background tshark script on 3 and 6 -- /home/ayush/ADS/tcpdump2gureKDDCup99-- ./auto_dataset_conv.sh 
step1 : start ids_detect.py 
step2 - to generate normal traffic - run script- norm_traf_gen.sh -- root@instance-6:/home/ayush/ADS/normal_traffic_gen



Testing attack -- 
Instance 1,3 and 6
step0: start background tshark script on 3 and 6 -- /home/ayush/ADS/tcpdump2gureKDDCup99-- ./auto_dataset_conv.sh
step1 start make.py
step2 : trigger attack script1 /home/ayush/attacks/final_attack -- sock_syn.py  -- instance1
step3 : trigger attack script2 /home/ayush/attacks/final_attack 	port_scan.py -- instance1





















Training:
instance-3, 5 and 6
instance 5- master -- /home/ayush/ADS/ann  -- fede.py -- run this to train
instance 3 and 6 -workers --- /home/ayush/ADS/PySyft -- run_websocket_server.py --- run this to open websocket

step1 run websocket on instance 3 and 6 
	instance3   python3 run_websocket_server.py --host 10.128.0.12 --port 8777 --id alice
	instance6	python3 run_websocket_server.py --host 10.128.0.16 --port 8778 --id bob
step2 run fede.py on instance 5 
			python3 fede.py
			
Result -- generates binaize-threat-model.pt in /home/ayush/ADS/ann instance 3,5 and 6

fede.py #todo scp model to 3 and 6

Testing normal-- 
Instance- 3, and 6

step0 : start ids_detect.py  -## todo
step1: start background tshark script on 3 and 6 -- /home/ayush/ADS/tcpdump2gureKDDCup99-- ./auto_dataset_conv.sh # todo -- write cleanup script for tcpdump    #todo -- increase iterations , and 
step2 - to generate normal traffic - run script- norm_traf_gen.sh -- root@instance-6:/home/ayush/ADS/normal_traffic_gen

Testing attack-- 
Instance 1,3 and 6
step0 start ids_detect.py  -## todo
step1: start background tshark script on 3 and 6 -- /home/ayush/ADS/tcpdump2gureKDDCup99-- ./auto_dataset_conv.sh
step2 : trigger attack script1 /home/ayush/attacks/final_attack -- sock_syn.py  -- instance1
step3 : trigger attack script2 /home/ayush/attacks/final_attack 	port_scan.py -- instance1




###todo ids_detect.py

1) start polling for csv starting from seed_0.csv -- iterate upto 100 and reset
2) detect - predict for each csv- each row-- group with count, if count of normal > 3 - then normal, else max of threats 
3) only print detected threat if not normal 


##todo - beautify- print statements 

##todo morning- iterations - prediction - to 20

