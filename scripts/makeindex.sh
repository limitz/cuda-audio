#!/bin/bash

FILES=`find $1 -name *.wav`
for f in $FILES; do
	echo $f
done
