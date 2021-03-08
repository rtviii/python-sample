#!/usr/bin/bash


for i in {1..10};
do
	awk -F ',' "NR==1{print\$$i}" batch3_\#20.csv 
done;


		

