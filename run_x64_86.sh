#!/bin/sh
./scripts/floorit
PID=`pidof jackd`
[ ! -z $PID ] && kill $PID
/usr/bin/jackd --silent -t2000 -dalsa -r48000 -p512 -n2 -Xseq -D -P95 -Chw:TR6S -Phw:TR6S 1>/dev/null 2>/dev/null &
sleep 2
./bin/program
PID=`pidof jackd`
[ ! -z $PID ] && kill $PID
