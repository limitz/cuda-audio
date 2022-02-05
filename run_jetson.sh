#!/bin/sh
./scripts/jfloorit
PID=`pidof jackd`
[ -z $PID ] && /usr/bin/jackd -t2000 -dalsa -r44100 -p1024 -n2 -Xraw -D -P95 -Chw:USB -Phw:USB &
sleep 2
./bin/program $1 $2
