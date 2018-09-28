#!/bin/bash
camera="$1"
if [ -z "$camera" ]; then camera=9; fi
seq 0 998 | xargs -n10 -P4 -- ./run-opticalflow.sh -c $camera
cd results
ls *.png | sed -nf "ren.sed" | xargs -n2 -- mv
tar cf "results-c${camera}-f1000.tar" *.png
rm *.png
cd ..
