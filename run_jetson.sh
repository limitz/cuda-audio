#!/bin/sh
./scripts/jfloorit
PID=`pidof jackd`
[ -z $PID ] && /usr/bin/jackd -t2000 -dalsa -r44100 -p1024 -n2 -Xnone -D -P95 -Chw:TR6S -Phw:TR6S &
sleep 2
PID=`pidof jackd`
./bin/program || kill $PID
