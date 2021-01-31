import sys
from time import sleep

from np_utils import *

def print_sample_block(n):
  gen = mp_get_batch_random_sample_data(1, 1)
  for i in range(n):
    sleep(0.5)
    block, ent = next(gen)
    print(block)
    print(ent)
    print()


if __name__ == '__main__':
  args = sys.argv
  if len(args) == 1:
    num = 1
  else:
    num = int(args[1])
  print_sample_block(num)
