#!/bin/bash
#echo "HELLO World"
tshark -Y "ip.src==10.128.0.0/24 && ip.dst==10.128.0.0/24" -b duration:10 -a files:6 -w test.pcap &
cnt=1
#filename="text.pcap"
#grepname="test_000$cnt"
#filename=$(ls | grep "test_000${cnt}_*")
#echo $filename
while [ $cnt -lt 6 ]
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
				f2=$(ls | grep "test_0000${cnt1}_*")
			fi

		fi
	elif [ $cnt -lt 100 ]
	then
		filename=$(ls | grep "test_000${cnt}_*")
		if [ "$filename" != "" ]
                then
        		cnt1=$((cnt + 1))
                        f2=$(ls | grep "test_0000${cnt1}_*")
                        if [ "$f2" == "" ]
                        then
                                f2=$(ls | grep "test_0000${cnt1}_*")
                        fi
		fi
	elif [ $cnt -lt 1000 ]
	then
		filename=$(ls | grep "test_00${cnt}_*")
		if [ "$filename" != "" ]
		then
                        cnt1=$((cnt + 1))
                        f2=$(ls | grep "test_0000${cnt1}_*")
                        if [ "$f2" == "" ]
                        then
                                f2=$(ls | grep "test_0000${cnt1}_*")
                        fi
		fi

	elif [ $cnt -lt 10000 ]
	then
		filename=$(ls | grep "test_0${cnt}_*")
		if [ "$filename" != "" ]
		then
                       cnt1=$((cnt + 1))
                        f2=$(ls | grep "test_0000${cnt1}_*")
                        if [ "$f2" == "" ]
                        then
                                f2=$(ls | grep "test_0000${cnt1}_*")
                        fi
 
		fi

	elif [ $cnt -lt 100000 ]
	then
		filename=$(ls | grep "test_${cnt}_*")
		if [ "$filename" != "" ]
                then
                        cnt1=$((cnt + 1))
                        f2=$(ls | grep "test_0000${cnt1}_*")
                        if [ "$f2" == "" ]
                        then
                                f2=$(ls | grep "test_0000${cnt1}_*")
                        fi
                fi

	fi
	
	if [ "$filename" != "" ]
	then
		if [ "$f2" != "" ]
		then

			echo $filename
			bro -r $filename darpa2gurekddcup.bro > conn${cnt}.list
			sort -n conn${cnt}.list > conn${cnt}_sort.list
			./trafAld.out conn${cnt}_sort.list trafAid_${cnt}.list
			cat trafAid_${cnt}.list | awk '{ for(i=7;i<47;i++) {printf $i;printf ",";} printf $47;print "" }' > seed_${cnt}.csv
			cnt=$((cnt + 1))
		fi
	fi
	#echo $cnt
done
