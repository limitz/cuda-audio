#!/bin/sh
./scripts/jfloorit
PID=`pidof jackd`
[ -z $PID ] && /usr/bin/jackd --silent -t2000 -dalsa -r44100 -p1024 -n2 -Xseq -D -P95 -Chw:USB -Phw:USB &
sleep 2
./bin/program $1 $2
