#!/bin/bash

if [ $# -ne 1 ]; then
    echo "./download_data.sh <path-to-set>"
    echo "Example: ./download_data.sh ../data/FullVSOURL/CCURL"
    echo "Example: ./download_data.sh ../data/URL1553"
    echo "Example: ./download_data.sh ../data/FullVSOURL/NonCCURL"
    exit 1
fi

cd $1
# mkdir -p images
# cd images
mkdir -p images 

counter=0
##for each file 
for f in *.txt; do 
	dir_n="./images/"
	while IFS=' ' read junk1 url junk2 junk3;
	do
		# echo $url
		
		file_number=${url##*/}
		suffix=".txt"
		file_prefix=${f%$suffix}
		filename="$dir_n$file_prefix$file_number"

		# echo $filename
		# if (($counter == 30))
		# then 
		# 	sleep 5;
		# 	ps aux | grep wget | wc -l;
		# 	counter=0;
		# fi
	    wget -nc --read-timeout=10 -t 2 -O $filename $url
	    echo $filename
	    # counter=$(($counter+1))

	done < $f
done;