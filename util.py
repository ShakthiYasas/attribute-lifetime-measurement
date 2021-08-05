from multiprocessing import Process

def run_in_parallel(*fns):
  proc = []
  for fn in fns:
    p = Process(target=fn)
    p.start()
    proc.append(p)
  for p in proc:
    p.join()

def without_keys(d, keys):
  return {x: d[x] for x in d if x not in keys}