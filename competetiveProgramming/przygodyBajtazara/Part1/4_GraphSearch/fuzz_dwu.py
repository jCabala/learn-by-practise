import random
import subprocess
import sys

random.seed(0)

def brute(n, a, b):
    best = None
    for mask in range(1<<n):
        r0 = [None]*n
        r1 = [None]*n
        for i in range(n):
            if (mask>>i)&1:
                r0[i] = b[i]
                r1[i] = a[i]
            else:
                r0[i] = a[i]
                r1[i] = b[i]
        if len(set(r0))==n and len(set(r1))==n:
            cnt = bin(mask).count('1')
            if best is None or cnt < best:
                best = cnt
    return best if best is not None else -1

# Compile the dwu.cpp program
compilation = subprocess.run(['g++', '-o', 'dwu', 'dwu.cpp'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if compilation.returncode != 0:
    print('Compilation failed:')
    print(compilation.stderr.decode())
    sys.exit(1)

# run many random tests
for t in range(5000):
    n = random.randint(1,10)
    # build a multiset of heights where each height appears 1 or 2 times
    pool = []
    nexth = 1
    while len(pool) < 2*n:
        # decide 1 or 2 occurrences for this new height, but don't overshoot
        remaining = 2*n - len(pool)
        cnt = random.choice([1,2])
        if cnt > remaining:
            cnt = remaining
        for _ in range(cnt):
            pool.append(nexth)
        nexth += 1
    random.shuffle(pool)
    a = pool[:n]
    b = pool[n:]
    # ensure solvable (it should be if every height appears twice)
    true = brute(n, a, b)
    if true == -1:
        continue
    inp = str(n) + '\n' + ' '.join(map(str,a)) + '\n' + ' '.join(map(str,b)) + '\n'
    try:
        p = subprocess.run(['./dwu'], input=inp.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1)
        out = p.stdout.decode().strip()
    except Exception as e:
        print('Execution error', e)
        sys.exit(1)
    try:
        got = int(out)
    except:
        print('Bad output parsing:', out)
        sys.exit(1)
    if got != true:
        print('Mismatch found on test', t)
        print('n=', n)
        print('a=', a)
        print('b=', b)
        print('expected=', true, 'got=', got)
        # print full input in the required format
        print('\nFull input:')
        print(inp)
        sys.exit(0)

print('No mismatches found in 2000 random tests')
