#!/bin/bash


for ii in {1..6..1}
do

    var=$((10**$ii))


    foo=$(bc <<< "scale=6; 1*10^-06")


    ans=$(echo "${foo}*${var}" |bc)    

    

    echo "Running Detections on Conf${ans} Visual"
    python detector.py --det NMS$ii-Olive --weights yolov3-608.weights --cfg cfg/yolov3-608.cfg --nms_thresh ${ans} --confidence 0.5
    
done

