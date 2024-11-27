#!/bin/bash

./master_photref_collector.py &
PID="$!"
for((i=0; i<1;)); do
    ps -e -o pid,rss,vsz|grep "$PID"
    sleep 0.5
done
