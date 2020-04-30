#!/bin/bash
#echo "HELLO World"
rm -rf *pcap
rm -rf *list
rm -rf *csv
tshark  -b duration:10 -a files:10 -w test.pcap &
cnt=1
#filename="text.pcap"
#grepname="test_000$cnt"
#filename=$(ls | grep "test_000${cnt}_*")
#echo $filename
while [ $cnt -lt 10 ]
do
	if [ $cnt -lt 10 ]
	then
		filename=$(ls | grep "test_0000${cnt}_*")
		if [ "$filename" != "" ]
		then
			cnt1=$((cnt + 1))
			f2=$(ls | grep "test_0000${cnt1}_*")
			if [ "$f2" == "" ]
			then
				if [ $cnt1 -eq 10 ]
				then
					f2=$(ls | grep "test_000${cnt1}_*")
				else
					f2=$(ls | grep "test_0000${cnt1}_*")
				fi
			fi

		fi
	elif [ $cnt -lt 100 ]
	then
		filename=$(ls | grep "test_000${cnt}_*")
		if [ "$filename" != "" ]
                then
        		cnt1=$((cnt + 1))
                        f2=$(ls | grep "test_000${cnt1}_*")
                        if [ "$f2" == "" ]
                        then
                        	if [ $cnt1 -eq 100 ]
                                then
                                        f2=$(ls | grep "test_00${cnt1}_*")
                                else
                                        f2=$(ls | grep "test_000${cnt1}_*")
                                fi
			fi
		fi
	elif [ $cnt -lt 1000 ]
	then
		filename=$(ls | grep "test_00${cnt}_*")
		if [ "$filename" != "" ]
		then
                        cnt1=$((cnt + 1))
                        f2=$(ls | grep "test_00${cnt1}_*")
                        if [ "$f2" == "" ]
                        then
                                f2=$(ls | grep "test_00${cnt1}_*")
                        fi
		fi

	elif [ $cnt -lt 10000 ]
	then
		filename=$(ls | grep "test_0${cnt}_*")
		if [ "$filename" != "" ]
		then
                       cnt1=$((cnt + 1))
                        f2=$(ls | grep "test_0${cnt1}_*")
                        if [ "$f2" == "" ]
                        then
                                f2=$(ls | grep "test_0${cnt1}_*")
                        fi
 
		fi

	elif [ $cnt -lt 100000 ]
	then
		filename=$(ls | grep "test_${cnt}_*")
		if [ "$filename" != "" ]
                then
                        cnt1=$((cnt + 1))
                        f2=$(ls | grep "test_${cnt1}_*")
                        if [ "$f2" == "" ]
                        then
                                f2=$(ls | grep "test_${cnt1}_*")
                        fi
                fi

	fi
	
	if [ "$filename" != "" ]
	then
		if [ "$f2" != "" ]
		then

			echo $filename
			tshark -r $filename -Y "(ip.src==10.128.0.0/24 && ip.dst==10.128.0.0/24) || (ip.src==104.17.142.0/24 || ip.dst==104.17.142.0/24) " -w test_${cnt}.pcap
			bro -r test_${cnt}.pcap darpa2gurekddcup.bro > conn${cnt}.list
			sort -n conn${cnt}.list > conn${cnt}_sort.list
			./trafAld.out conn${cnt}_sort.list trafAid_${cnt}.list
			cat trafAid_${cnt}.list | awk '{ for(i=7;i<47;i++) {printf $i;printf ",";} printf $47;print "" }' > seed_${cnt}.csv
			cnt=$((cnt + 1))
			if [ $cnt -eq 10 ]
			then
				touch seed_${cnt}.csv
				#cnt=1
			fi
		fi
	fi
	#echo $cnt
done
