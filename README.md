Github repo- https://github.ncsu.edu/arajend4/ADS
# ADS
Intrusion detection 

Setup details- 

instance-1 - 10.128.0.2  ---> master

instance-2 - 10.128.0.3  ---> worker1

Instance-3 - 10.128.0.4 ---> worker2 

instance-1 - 10.128.0.2  --- > attacker


# Steps to Run:
##############TRAINING  WITH DATASET TESTING##############################


Step1: Worker nodes   Instance-2 and Instance-3:

Run the websockets from ADS/PySyft: 

Instance-2

python3 run_websocket_server.py --host 10.128.0.3 --port 8777 --id alice

Instance-3

python3 run_websocket_server.py --host 10.128.0.4 --port 8778 --id bob


Step 2 Instance-1: 

On master run main_func.py -- ADS/master_ann_final

Run

python3 main_func.py -w 10.128.0.3,10.128.0.4 -p 8777,8778 -i alice,bob

##############TRAINING COMPLETED WITH DATASET TESTING##############################

##############REAL TIME TESTING##########################################

Step3: Once the training is completed, start packet analyser for real time testing -- ADS/data_pre_process/- Instance-2 & Instance-3

Run on instance-2 and 3

./auto_dataset_conv.sh



Step4: make predictions instance-2 and instance-3  ADS/predict_workers

run: 

python3 make_prediction.py



Step5: normal traffic gen on instance-2 and instance-3 /ADS/normal_traffic_gen

run :

./norm_traf_gen.sh <ip to ping1> <ip to ping2>

./norm_traf_gen.sh 10.128.0.2 10.128.0.3





Step 6:make prediction instance-2 and instance-3  ADS/predict_workers

run: 

python3 make_prediction.py




step 7: Detect intrusion

Repeat step4, then on attacker node, instance-1  ADS/final_attack

run: 

python3 port_scan.py 

python3 sock_scan.py <ip> 22

python3 sock_scan.py 10.128.0.3 22


Work distribution and progress:

|Modules	                         			           |     Progress	          |             Owner	              |            Due   |
|-----------------------------------------------|------------------------|---------------------------------|------------------|
|Try Approach 1 using Dianne	     			           |      Failed	           |         Ayushi & Benzil	        |        15-Mar    |
|Attack Modules									                        |		    100%			           |				        Ayushi			            |		      31-Mar    |
|Train Basic Model With KDD dataset				         |		    100%			           |				        Benzil			            |		      31-Mar    | 	 
|Test Basic Model							            	           |		    100%			           |				        Benzil			            |		      31-Mar    |	 
|Train and Test Model with Multiple Nodes		     |		    100%			           |				        Ayushi			            |		      12-Apr    |	 
|DataSet Parser from Tshark Output				          |		    100%			           |				        Ayushi			            |		      31-Mar    |	 
|Prediction With whole system using live data	  |		    100%			           |			      Ayushi & Benzil		       |		      20-Apr    |	 
|Complete System Integration					               |		    100%			           |			      Ayushi &Benzil		        |		      26-Apr    |	 
|Paper Writing									                         |		    100%		            |			      Ayushi &Benzil		        |		      2-May     |	 

