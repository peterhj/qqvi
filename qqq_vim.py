#!/usr/bin/env python3

from datetime import datetime
import json
import os

HOME = os.environ["HOME"]
LOG_DIR = os.path.join(HOME, ".qq", "log")
CACHE_DIR = os.path.join(HOME, ".qq", "cache")

QQ_PAT = f"{chr(0x11)}{chr(0x11)}"

def main():
    t0 = datetime.utcnow().isoformat()
    ctrp = os.path.join(CACHE_DIR, ".ctr")
    try:
        with open(ctrp, "r") as f:
            item = json.loads(f.read())
        ctr = item["ctr"]
        if not isinstance(ctr, int):
            ctr = 0
    except OSError:
        ctr = 0
    ctr += 1
    txtp = os.path.join(CACHE_DIR, f"{ctr}.txt")
    with open(txtp, "w") as f:
        print(f"{QQ_PAT} ", file=f, flush=True)
    with open(ctrp, "w") as f:
        print(json.dumps({"t0": t0, "ctr": ctr}), file=f, flush=True)
    print(txtp)

if __name__ == "__main__":
    main()
