#!/bin/sh
./scripts/floorit
PID=`pidof jackd`
[ ! -z $PID ] && kill $PID
/usr/bin/jackd --silent -t2000 -dalsa -r48000 -p512 -n2 -Xseq -D -P95 -Chw:USB -Phw:PCH &
sleep 1
./bin/program
PID=`pidof jackd`
[ ! -z $PID ] && kill $PID
