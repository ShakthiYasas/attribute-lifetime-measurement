from multiprocessing import Process

# Parallelizes processes
def run_in_parallel(*fns):
  proc = []
  for fn in fns:
    p = Process(target=fn)
    p.start()
    proc.append(p)
  for p in proc:
    p.join()

# Removed key-value pairs from JSON given a list a keys
def without_keys(d, keys):
  return {x: d[x] for x in d if x not in keys}