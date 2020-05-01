import pandas as pd

pd.set_option("display.max_columns", 10)
import plotly.graph_objects as go
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt; plt.rcdefaults()


def KDD_data_ana():
    # KDD CUP 1999 data (https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
    # 41 column names - https://kdd.ics.uci.edu/databases/kddcup99/kddcup.names
    column = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
              'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
              'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
              'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
              'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
              'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
              'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
              'dst_host_srv_rerror_rate']

    # Use 10% of data http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz
    data_frame = pd.read_csv("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",
            names=column + ["threat_type"])
    total_each_threat_type = Counter(data_frame["threat_type"])
    threat_types = list(total_each_threat_type.keys())
    threat_counts=[]
    dos=["back.","land.","neptune.","pod.","smurf.","teardrop."]
    probe=["satan.","ipsweep.","nmap.","portsweep."]
    R2L=["guess_passwd.","ftp_write.","imap.","phf.","multihop.","warezmaster.","warezclient.","spy."]
    U2R=["buffer_overflow.","loadmodule.","rootkit.","perl."]
    for threat_type in threat_types:
        threat_counts.append(total_each_threat_type[threat_type])
    # threat_counts = [total_each_threat_type[threat_type] for threat_type in threat_types]
    print("Total distinct number of threat types : ", len(threat_types))
    plt.figure()
    plt.bar(threat_types, threat_counts)
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.title('Threat Types')
    plt.savefig("all_threats.png")
    dos_c=[]
    dos_tot=0
    for dos_count in dos:
        dos_c.append(total_each_threat_type[dos_count])
        dos_tot+=total_each_threat_type[dos_count]
    plt.figure()
    plt.bar(dos, dos_c)
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.title('DOS attacks')
    plt.savefig("dos.png")
    probe_c = []
    probe_tot=0
    for probe_count in probe:
        probe_c.append(total_each_threat_type[probe_count])
        probe_tot+=total_each_threat_type[probe_count]
    plt.figure()
    plt.bar(probe, probe_c)
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.title('probe attacks')
    plt.savefig("probe.png")
    R2L_c = []
    R2L_tot=0
    for R2L_count in R2L:
        R2L_c.append(total_each_threat_type[R2L_count])
        R2L_tot += total_each_threat_type[R2L_count]
    plt.figure()
    plt.bar(R2L, R2L_c)
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.title('R2L attacks')
    plt.savefig("R2L.png")
    U2R_c = []
    U2R_tot=0
    for U2R_count in U2R:
        U2R_c.append(total_each_threat_type[U2R_count])
        U2R_tot+=total_each_threat_type[U2R_count]
    plt.figure()
    plt.bar(U2R, U2R_c)
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.title('U2R attacks')
    plt.savefig("U2R.png")
    #plt.show()
    threat_major_types=["DOS","PROBE","R2L","U2R"]
    threat_major_count=[dos_tot,probe_tot,R2L_tot,U2R_tot]
    plt.figure()
    plt.pie(threat_major_count,labels=threat_major_types)
    plt.savefig("major_threat.png")


    rate_coolumns = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
                          'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                          'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'count',
                          'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                          'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                          'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                          'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                          'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
    rate_data_frame = data_frame[rate_coolumns].copy()
    # delete columns with constant value
    rate_data_frame = rate_data_frame.loc[:, (rate_data_frame != rate_data_frame.iloc[0]).any()]
    # scale columns
    final_data_frame = rate_data_frame / rate_data_frame.max()
    X = final_data_frame.values
    # final dataframe
    print("Shape of feature matrix : ", X.shape)
    threat_types = data_frame["threat_type"].values
    encoder = LabelEncoder()
    # use LabelEncoder to encode the threat types in numeric values
    y = encoder.fit_transform(threat_types)
    print("Shape of target vector : ", y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    print("Number of records in training data : ", X_train.shape[0])
    print("Number of records in test data : ", X_test.shape[0])
    print("Total distinct number of threat types in training data : ", len(set(y_train)))
    return X_train, X_test, y_train, y_test,encoder

