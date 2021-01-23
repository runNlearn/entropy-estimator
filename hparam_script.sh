#/bin/bash

export CUDA_VISIBLE_DEVICES=-1

#python make_hparam_parent_dir.py

for tv in 0 1
do
  for vv in 0 1
  do
#    python hparam_tuning.py --tv $tv --vv $vv > log.$tv-$vv &
    python hparam_py_script.py --tv $tv --vv $vv &
  done
done
