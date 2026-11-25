from prof import *
from profiling import categorise

tot, grand, USED_DEVICE_TIME = categorise(prof)

if __name__ == "__main__":
    print(f"{'category':22s} {'ms':>10s} {'share':>8s}   "
        f"({'device' if USED_DEVICE_TIME else 'host'} self time)")
    print("-" * 54)
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        if grand:
            print(f"{k:22s} {v/1e3:10.2f} {100*v/grand:7.1f}%")
    print("-" * 54)
    print(f"{'TOTAL':22s} {grand/1e3:10.2f}")
