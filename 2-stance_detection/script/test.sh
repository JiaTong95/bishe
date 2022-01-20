#!bin/bash

a=("0.1" "0.2" "0.3" "0.5")

for i in ${a[*]}
do
    echo $i
done

for i in ${a[@]}
do
    echo $i
done
