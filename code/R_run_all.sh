#!/bin/bash

# Second set of instances
for inst in RD2b RD2 RD3 RD4; do
#   for dist in precalced haversine; do
  for dist in precalced; do
    echo "▶ Running $inst with $dist..."
    python run_dynamic_instance.py --inst "$inst" --dist_method "$dist"
  done
done