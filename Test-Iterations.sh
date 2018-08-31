#!/bin/bash


for ii in {1..5..1}
do
    var=$((10**$ii))


    foo=$(bc <<< "scale=6; 1*10^-06")


    ans=$(echo "${foo}*${var}" |bc)    

    echo ${ans}
    
done

