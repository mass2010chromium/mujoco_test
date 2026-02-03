"""
Simple self-contained profiling tools.
Warning: This profiling can be a significant burden on very small functions.
It's at least a python function call and some dictionary ops of overhead.
Someday I'll write a C version

Author: mass2010chromium
"""

# Flag to disable profiling globally
DO_PROFILE = True

import atexit
from functools import wraps
import threading
import time

PROF_STARTED = False
PROF_START_TIME = -1
PROF_OUTFILE = None
_prof_map = []
class ProfileInfo:
    def __init__(self, thread_name):
        self.thread_name = thread_name
        self.min_time = 1e18
        self.max_time = 0
        self.time = 0
        self.calls = 0

_prof_id = 0

if DO_PROFILE:
    def profiled(*args, **kwargs):
        """
        Decorator that sets a function to be profiled.
        """
        
        if len(args) == 0:
            return lambda x: profiled(x, **kwargs)
        if len(args) == 1:
            f = args[0]
        else:
            raise ValueError('Invalid arguments to profiling func')

        @wraps(f)
        def wrapper(*args, **kwargs):
            if not PROF_STARTED:
                return f(*args, **kwargs)
            tid = threading.get_ident()
            prof_map = wrapper.prof_map
            if tid not in prof_map:
                prof_map[tid] = ProfileInfo(threading.current_thread().name)
            prof_info = prof_map[tid]
            stime = time.monotonic_ns()
            retval = f(*args, **kwargs)
            dt = time.monotonic_ns() - stime
            prof_info.min_time = min(prof_info.min_time, dt)
            prof_info.max_time = max(prof_info.max_time, dt)
            prof_info.time += dt
            prof_info.calls += 1
            return retval
        wrapper.prof_map = dict()

        if 'name' in kwargs:
            name = kwargs['name']
        else:
            name = f.__name__
        _prof_map.append((name, wrapper.prof_map))
        return wrapper
else:
    def profiled(*args, **kwargs):
        if len(args) == 0:
            return lambda x: profiled(x, **kwargs)
        if len(args) == 1:
            f = args[0]
        else:
            raise ValueError('Invalid arguments to profiling func')
        return f

def prof_start(outfile=None):
    global PROF_START_TIME
    global PROF_OUTFILE
    global PROF_STARTED
    if not DO_PROFILE:
        return
    PROF_START_TIME = time.time()
    if outfile is not None:
        PROF_OUTFILE = outfile.format(PROF_START_TIME)
    PROF_STARTED = True
    atexit.register(prof_dump_info)

def prof_dump_info():
    if not DO_PROFILE:
        return
    cur_time = time.time()
    print_buf = [f'[{cur_time}]-----------------------------']
    def _print(s):
        print(s)
        print_buf.append(s)
    _print(f"Total time: {cur_time - PROF_START_TIME}")
    for fname, data in _prof_map:
        _print(f"> Function {fname}")
        total_time = 0
        total_calls = 0
        max_time = 0
        min_time = 1e18
        _print(f"           |       Time |    Calls |     Average |         Min |         Max | Thread Info")
        for tid, prof_info in data.items():
            if prof_info.calls == 0:
                avg_time = 0
            else:
                avg_time = prof_info.time/1e9/prof_info.calls
            _print(f"           | {prof_info.time/1e9: >11.6f} | {prof_info.calls: >7} | {avg_time: >11.8f} | {prof_info.min_time/1e9: >11.8f} | {prof_info.max_time/1e9: >11.8f} | {prof_info.thread_name}")
            total_time += prof_info.time
            total_calls += prof_info.calls
            max_time = max(max_time, prof_info.max_time)
            min_time = min(min_time, prof_info.min_time)
        if total_calls == 0:
            avg_time = 0
        else:
            avg_time = total_time/1e9/total_calls
        _print(f"<< Totals: | {total_time/1e9: >11.8f} | {total_calls: >7} | {avg_time: >11.8f} | {min_time/1e9: >11.8f} | {max_time/1e9: >11.8f} | {len(data)} threads")
    print_buf.append('')

    if PROF_OUTFILE is not None:
        with open(PROF_OUTFILE, 'a') as outfile:
            outfile.write('\n'.join(print_buf))

if __name__ == "__main__":
    @profiled
    def a(arg1, arg2):
        print(arg1, arg2)
        return arg1 + arg2

    @profiled
    def b(arg1, arg2):
        print(arg1)
        return arg1 + arg2
    
    prof_start()
    for i in range(100000):
        a(i, i+1)
        b(i, i+1)
    prof_dump_info()

