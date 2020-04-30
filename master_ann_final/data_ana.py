import pandas as pd
pd.set_option("display.max_columns", 10)
import plotly.graph_objects as go
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def KDD_data_ana():
    #matplotlib.use('GTK')
    # We use the KDD CUP 1999 data (https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
    # 41 column names can be found at https://kdd.ics.uci.edu/databases/kddcup99/kddcup.names
    colnames = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
                'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
                'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                'dst_host_srv_rerror_rate']
    
    # We take 10% of the original data which can be found at 
    # http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz
    # We select the first 100K records from this data
    df = pd.read_csv("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",
            names=colnames+["threat_type"])[:100000]
    
    print(df.head(3))
    threat_count_dict = Counter(df["threat_type"])
    threat_types = list(threat_count_dict.keys())
    threat_counts = [threat_count_dict[threat_type] for threat_type in threat_types]
    print("Total distinct number of threat types : ",len(threat_types))
    #fig = go.Figure([go.Bar(x=threat_types, y=threat_counts,text=threat_counts,textposition='auto')])
    #fig.show()
    numerical_colmanes = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
                        'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'count',
                        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
    numerical_df = df[numerical_colmanes].copy()
    # Lets remove the numerical columns with constant value
    numerical_df = numerical_df.loc[:, (numerical_df != numerical_df.iloc[0]).any()]
    # lets scale the values for each column from [0,1]
    # N.B. we dont have any negative values]
    final_df = numerical_df/numerical_df.max()
    X = final_df.values
    # final dataframe has 33 features
    print("Shape of feature matrix : ",X.shape)
    threat_types = df["threat_type"].values
    encoder = LabelEncoder()
    # use LabelEncoder to encode the threat types in numeric values
    y = encoder.fit_transform(threat_types)
    print("Shape of target vector : ",y.shape)
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=42, stratify=y)
    print("Number of records in training data : ", X_train.shape[0])
    print("Number of records in test data : ", X_test.shape[0])
    print("Total distinct number of threat types in training data : ",len(set(y_train)))
    return X_train, X_test, y_train, y_test,encoder

