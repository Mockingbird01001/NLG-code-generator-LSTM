import os
import time
import multiprocessing as mp
from multiprocessing import Pool, Process, Pipe, Lock, Value, Array, Manager
def func(x):
    print("||| Sleep time:", x)
    time.sleep(x)
    print("|||", __name__, os.getppid(), os.getpid())
    return x ** 2
def foo(q):
    q.put('hello')
def func_conn(conn):
    conn.send([42, None])
    conn.close()
def func_lock(l, i):
    l.acquire()
    try:
        sleep_time = i * 5 % 3
        time.sleep(sleep_time)
        print("|||", i, sleep_time)
    finally:
        l.release()
        pass
def func_memo(n, a):
    n.value += 1
    for i in range(len(a)):
        a[i] = a[i] + 1
def func_mana(d, l):
    d[1] = '1'
    d['two'] = 2
    d[0.25] = None
    l.reverse()
timer = time.time()
if __name__ == '__main__':
    pass
    with Pool(2) as p0:
        print(p0.map(func, [3, 1, 2]))
print("||| Total Time:", time.time() - timer)
pass
"""
Learning how to use [multiprocessing] in Python. (Official)
https://docs.python.org/3.6/library/multiprocessing.html
multiprocessing Process join run, 李皮筋的技术博客
https://www.cnblogs.com/lipijin/p/3709903.html
"""