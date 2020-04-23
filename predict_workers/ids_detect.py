import pandas as pd
pd.set_option("display.max_columns", 10)
import plotly.graph_objects as go
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import syft as sy
import numpy as np
import torch.nn.functional as F
from syft.frameworks.torch.fl import utils
import torch.nn as nn
from letters import print_letters 
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        input_dim: number of input features.
        output_dim: number of labels.
        """
        super(Net, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        #self.fc1 = torch.nn.Linear(input_dim, 1024)
        #self.dropout1 = torch.nn.Dropout(0.01)
        #self.fc2 = torch.nn.Linear(1024, 768)
        #self.dropout2 = torch.nn.Dropout(0.01)
        #self.fc4 = torch.nn.Linear(768, 512)
        #self.dropout3 = torch.nn.Dropout(0.01)
        #self.fc3 = torch.nn.Linear(512, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        #x = x.view(-1, 33)
        #x = F.relu(self.fc1(x))
        #x = self.dropout1(x)
        #x = F.relu(self.fc2(x))
        #x = self.dropout2(x)
        #x = F.relu(self.fc4(x))
        #x = self.dropout3(x)
        #x = F.sigmoid(self.fc3(x))
        return outputs


model_new = Net(33,23)

colnames = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
            'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate']

df_dta= pd.read_csv("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",
        names=colnames+["threat_type"])[:100000]
numerical_colmanes = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
                      'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                      'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'count',
                      'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                      'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                      'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                      'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                      'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

numdf1=df_dta[numerical_colmanes].copy()
n_col=numdf1.loc[:,(numdf1 != numdf1.iloc[0]).any()]
numerical_df1 = df_dta[n_col.columns].copy()

threat_types = df_dta["threat_type"].values
encoder = LabelEncoder()
y = encoder.fit_transform(threat_types)

PATH = "/home/ayush/ADS/tcpdump2gureKDDCup99/"
files = 99


try:
	print("Predicting Live Captured Data")
	while True:
		if os.path.isfile("binaize-threat-model.pt"):
	
			model_new.load_state_dict(torch.load("binaize-threat-model.pt"))
			model_new.eval()
			
			for i in range(files):
				if os.path.isfile(PATH + "seed_" + str(i+2) + ".csv""):
					df = pd.read_csv((PATH + "seed_" str(i+1) + ".csv", names=colnames)
					numerical_df = df[n_col.columns].copy()
					final_df = numerical_df/numerical_df1.max()
					X_test = final_df.values
					test_inputs = torch.tensor(X_test,dtype=torch.float)			

					size=len(test_inputs)
					predict_labels = {}

					for j in range(size):
						data = test_inputs[j]
						pred = model_new(data)
						pred_label = int(pred.argmax().data.cpu().numpy())
						pred_threat = encoder.inverse_transform([pred_label])[0]
						
						if pred_threat == 'normal':
							predict_labels[pred_threat] += 1
						else:
							predict_labels[pred_threat] += 1

			if predict_labels['normal'] >= 3:
				print ("--- NORMAL STATE ---")
			else:
				print ("\n!!! ALERT !!!")
				print ("!!! PC UNDER ATTACK !!!")
				print("\nPredicted threat type : ", max(predict_labels.items(), key=operator.itemgetter(1))[0])

except KeyboardInterrupt:
	print("Quitting the program.")
except:
	print("Unexpected error: "+sys.exc_info()[0])
	raise


