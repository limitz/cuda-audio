#!/bin/sh
./scripts/floorit
PID=`pidof jackd`
[ -z $PID ] && /usr/bin/jackd --silent -t2000 -dalsa -r48000 -p512 -n2 -Xseq -D -P95 -Chw:TR6S -Phw:TR6S &
sleep 2
./bin/program $1 $2
