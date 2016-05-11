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
	    wget -nc --read-timeout=10 -t 2 -O $filename $url;

	done < $f
done;