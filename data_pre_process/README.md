# Data pre processing
./auto_dataset_conv.sh
Step1. run ./auto_dataset_conv.sh
This starts capturing packerts, in the interval of 10 seconds, total of 100 seconds. Interval and iterations can be increased by altering the line- "tshark  -b duration:10 -a files:10 -w test.pcap &" in the script

It generated the pcap every 10 seconds, bro script parses the pcap, and genrates the parsed file. 

Uses bro-ids script to generate the parsed traffic from tshark output- from pcap files/ 
ref- https://github.com/inigoperona/tcpdump2gureKDDCup99/blob/master/darpa2gurekddcup.bro

The  parsed file is fed to parse_tcp_bro.py, in order to calculate few other parameters like :
'duration'	'src_bytes'	'dst_bytes'	'wrong_fragment'	'urgent'	'hot'	'num_failed_logins'	'num_compromised'	'root_shell'	'su_attempted'	'num_root'	'num_file_creations'	'num_shells'	'num_access_files'	'num_outbound_cmds' 'count'	'srv_count'	'serror_rate'	'srv_serror_rate'	'rerror_rate'	'srv_rerror_rate'	'same_srv_rate'	'diff_srv_rate'	'srv_diff_host_rate'	'dst_host_count'	'dst_host_srv_count'	'dst_host_same_srv_rate'	'dst_host_diff_srv_rate'	'dst_host_same_src_port_rate'	'dst_host_srv_diff_host_rate'	'dst_host_serror_rate'	'dst_host_srv_serror_rate'	'dst_host_rerror_rate'	'dst_host_srv_rerror_rate'		


This python script generates a seed csv file, which acts as an input to ../predict_workers/make_prediction.py to detect intrusions live. 
