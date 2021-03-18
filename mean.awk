#!/usr/bin/awk  -f

BEGIN {FS=" ";}
# Processes every line: summing record "i"
{ for (i=1;i<=NF;i++) 
	total[i]+=$i ; 
} 
END { 
for(i=1; i<=NF; i++)
printf "%.4f ",total[i]/NR;print "\0"
}

	
