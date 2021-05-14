#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

url=$1
dir=$2
downloads=$3
images=0
delay=30

if [ -z $url ] || [ -z $dir ] || [ -z $downloads ];
then
    echo -e "${RED}Invalid Parameters ${NC}"
    echo -e "${GREEN}\nUsage:${NC}\n\t./download_captcha.sh captcha_curl_url image_directory numbers_of_downloads\n"
    exit
fi

if [ -d $dir ];
then
    count=$(ls $dir| wc -l)
    echo "$count images found"
    ((count++))
else
    echo "create new directory"
    mkdir $dir
    count=1
fi

((downloads+=count-1))

while [ $count -le $downloads ]
do
    if [ $images -eq 30 ];
    then
        echo "sleep $delay s"
        sleep $delay
        images=0
    fi

    echo downloading $count.png ... 
    curl $url --output $dir/$count.png --no-progress-meter --no-keepalive --tcp-fastopen
    if [ $downloads -gt 20 ];
    then
        sleep 5
    fi
    ((count++))
    ((images++))
done
exit
