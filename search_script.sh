#/bin/bash

export CUDA_VISIBLE_DEVICES=-1

for bs in 32 64 128 256
do
  python search.py --bs $bs --tv 0 --vv 1 > log.$bs-0-1 &
  python search.py --bs $bs --tv 1 --vv 0 > log.$bs-1-0 &
done
