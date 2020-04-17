#!/bin/sh
#echo "HELLO World"
#tshark -b duration:60 -w test.pcap -b files:5 &
cnt=0
#filename="text.pcap"
#grepname="test_000$cnt"
#filename=$(ls | grep "test_000${cnt}_*")
#echo $filename
while [ $cnt -lt 100 ]
do
	if [ $cnt -lt 10 ]
	then
		filename=$(ls | grep "test_0000${cnt}_*")
		if [ "$filename" != "" ]
		then
			echo $filename
		fi
	elif [ $cnt -lt 100 ]
	then
		filename=$(ls | grep "test_000${cnt}_*")
		if [ "$filename" != "" ]
                then
	                echo $filename
		fi
	elif [ $cnt -lt 1000 ]
	then
		filename=$(ls | grep "test_00${cnt}_*")
		if [ "$filename" != "" ]
		then
                        echo $filename
                fi

	elif [ $cnt -lt 10000 ]
	then
		filename=$(ls | grep "test_0${cnt}_*")
		if [ "$filename" != "" ]
		then
                        echo $filename
                fi

	elif [ $cnt -lt 100000 ]
	then
		filename=$(ls | grep "test_${cnt}_*")
		if [ "$filename" != "" ]
                then
                        echo $filename
                fi

	fi

	bro -r $filename darpa2gurekddcup.bro > conn${cnt}.list
	sort -n conn${cnt}.list > conn${cnt}_sort.list
	./trafAld.out conn${cnt}_sort.list trafAid_${cnt}.list	
	cnt=$((cnt + 1))
	#echo $cnt
done
