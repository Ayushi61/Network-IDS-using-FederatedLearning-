
import sys, getopt
import math
import copy
def parse(inputfile,outputfile):
    f=open(inputfile,'r')
    line1=f.readlines()
    #print(line1)
    f.close()
    startpt1=0
    for i in range(0,len(line1)):

        line1[i]=line1[i].strip("\n")
        #print(type(line1[i]))
        arr=copy.deepcopy(line1[i].split(" "))
        #print(type(line1[i]))
        #print("dfsdf")
        #print(arr)
        num_conn1=int(arr[0])
        startTimet1=arr[1]
        orig_pt1=arr[2]
        resp_pt1=arr[3]
        orig_ht1=arr[4]
        resp_ht1=arr[5]
        duration1=arr[6]
        protocol1=arr[7]
        service1=arr[8]
        flag1=arr[9]
        startT1=int(startTimet1.split(".")[0])
        startT2=int(startTimet1.split(".")[1])
        #init
        start=0
        count=0
        serror=0
        rerror=0
        same_srv=0
        diff_srv=0
        srv_count=0
        srv_serror=0
        srv_error=0
        srv_diff_host=0
        same_src_port=0
        for j in range(startpt1,i):
            #print(line1)
            arr2=copy.deepcopy(line1[j].split(" "))
            num_conn2 = int(arr2[0])
            startTimet2 = arr2[1]
            orig_pt2 = arr2[2]
            resp_pt2 = arr2[3]
            orig_ht2 = arr2[4]
            resp_ht2 = arr2[5]
            duration2 = arr2[6]
            protocol2 = arr2[7]
            service2 = arr2[8]
            flag2 = arr2[9]
            startT21 = int(startTimet1.split(".")[0])
            startT22 = int(startTimet1.split(".")[1])
            if (((startT1-2)<=startT2) and (startT2)<=startT1):
                if(start==0):
                    startpt1=j
                    start=1;
                if(resp_ht1==resp_ht2):
                    count+=1
                    if(flag2=="S0" or flag2=="S1" or flag2=="S2" or flag2=="S3"):
                        serror+=1
                    if(flag2=="REJ"):
                        rerror+=1
                    if ((service2!="other") and (service1==service2)):
                        same_srv+=1
                    if (service1!=service2):
                        diff_srv+=1
                if(resp_pt1==resp_pt2):
                    srv_count+=1
                    if (flag2 == "S0" or flag2 == "S1" or flag2 == "S2" or flag2 == "S3"):
                        srv_serror += 1
                    if (flag2 == "REJ"):
                        srv_error += 1
                    if (resp_ht1 != resp_ht2):
                        srv_diff_host += 1
        if(count!=0):
            serror_rate=serror/count
            rerror_rate=rerror/count
            same_srv_rate=same_srv/count
            diff_srv_rate=diff_srv/count
        else:
            serror_rate=0.0
            rerror_rate=0.0
            same_srv_rate=0.0
            diff_srv_rate=0.0
        if(srv_count!=0):
            srv_serror_rate=srv_serror/srv_count
            srv_error_rate=srv_error/srv_count
            srv_diff_host_rate=srv_diff_host/srv_count
        else:
            srv_serror_rate=0.0
            srv_error_rate=0.0
            srv_diff_host_rate=0.0

        line=arr
        line.append(str(count)+" ")
        line.append(str(srv_count)+" ")
        line.append(str(serror_rate)+" ")
        line.append(str(srv_serror_rate)+" ")
        line.append(str(rerror_rate)+" ")
        line.append(str(srv_error_rate)+" ")
        line.append(str(same_srv_rate)+" ")
        line.append(str(diff_srv_rate)+" ")
        line.append(str(srv_diff_host_rate)+" ")
        #line1[i]=arr
        line1[i]=' '.join(map(str,line))

        if(i<100):
            ctr100=0
        else:
            ctr100=i-100
        count=0
        serror = 0
        rerror = 0
        same_srv = 0
        diff_srv = 0
        srv_count = 0
        srv_serror = 0
        srv_error = 0
        srv_diff_host = 0
        same_src_port=0
        for j in range(ctr100,i):
            arr2 = copy.deepcopy(line1[j].split(" "))
            num_conn2 = arr2[0]
            startTimet2 = arr2[1]
            orig_pt2 = arr2[2]
            resp_pt2 = arr2[3]
            orig_ht2 = arr2[4]
            resp_ht2 = arr2[5]
            duration2 = arr2[6]
            protocol2 = arr2[7]
            service2 = arr2[8]
            flag2 = arr2[9]
            startT21 = int(startTimet1.split(".")[0])
            startT22 = int(startTimet1.split(".")[1])
            if(resp_ht1==resp_ht2):
                count+=1
                if (flag2 == "S0" or flag2 == "S1" or flag2 == "S2" or flag2 == "S3"):
                    serror += 1
                if (flag2 == "REJ"):
                    rerror += 1
                if ((service2 != "other") and (service1 == service2)):
                    same_srv += 1
                if (service1 != service2):
                    diff_srv += 1
            if (resp_pt1 == resp_pt2):
                srv_count += 1
                if (flag2 == "S0" or flag2 == "S1" or flag2 == "S2" or flag2 == "S3"):
                    srv_serror += 1
                if (flag2 == "REJ"):
                    srv_error += 1
                if (resp_ht1 != resp_ht2):
                    srv_diff_host += 1
            if(orig_pt1==orig_pt2):
                same_src_port+=1

        if (count != 0):
            serror_rate = serror / count
            rerror_rate = rerror / count
            same_srv_rate = same_srv / count
            diff_srv_rate = diff_srv / count
        else:
            serror_rate = 0.0
            rerror_rate = 0.0
            same_srv_rate = 0.0
            diff_srv_rate = 0.0
        if (srv_count != 0):
            srv_serror_rate = srv_serror / srv_count
            srv_rerror_rate = srv_error / srv_count
            srv_diff_host_rate = srv_diff_host / srv_count
        else:
            srv_serror_rate = 0.0
            srv_rerror_rate = 0.0
            srv_diff_host_rate = 0.0
        if(i-ctr100!=0):
            same_src_port_rate=same_src_port/(i-ctr100)
        else:
            same_src_port_rate=0.0
        arr=[]
        arr.append(line1[i])
        arr.append(str(count) + " ")
        arr.append(str(srv_count) + " ")
        arr.append(str(same_srv_rate) + " ")
        arr.append(str(diff_srv_rate) + " ")
        arr.append(str(same_src_port_rate) + " ")
        arr.append(str(srv_diff_host_rate) + " ")
        arr.append(str(serror_rate) + " ")
        arr.append(str(srv_serror_rate) + " ")
        arr.append(str(rerror_rate) + " ")
        arr.append(str(srv_rerror_rate) + " ")
        line1[i]=' '.join(map(str,arr))
    f2=open(outputfile,'w')
    for i in range(0,len(line1)):
        f2.write(str(line1[i])+"\n")
    f2.close()




    print( 'Input file is "', inputfile)
    print('Output file is "', outputfile)

argv=sys.argv[1:]
if (len(argv)!=4):
    print('test.py -i <inputfile> -o <outputfile>')
    sys.exit()
try:
    opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
except getopt.GetoptError:
    print( 'test.py -i <inputfile> -o <outputfile>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print( 'test.py -i <inputfile> -o <outputfile>')
        sys.exit()
    elif opt in ("-i", "--ifile"):
        inputfile = arg
    elif opt in ("-o", "--ofile"):
        outputfile = arg
parse(inputfile,outputfile)

