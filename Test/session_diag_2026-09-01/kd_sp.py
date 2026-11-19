import sys, os, ctypes, resource
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
libc = ctypes.CDLL(None)
libc.pthread_self.restype = ctypes.c_void_p
libc.pthread_get_stackaddr_np.restype = ctypes.c_void_p
libc.pthread_get_stackaddr_np.argtypes = [ctypes.c_void_p]
libc.pthread_get_stacksize_np.restype = ctypes.c_size_t
libc.pthread_get_stacksize_np.argtypes = [ctypes.c_void_p]
def report(tag):
    s = libc.pthread_self()
    addr = libc.pthread_get_stackaddr_np(s)
    size = libc.pthread_get_stacksize_np(s)
    print(f"{tag}: stackaddr=0x{addr:x} size={size/1e6:.1f}MB bottom=0x{addr-size:x} "
          f"rlimit={resource.getrlimit(resource.RLIMIT_STACK)}", flush=True)
report("start")
if os.environ.get("PRE"):
    __import__(os.environ["PRE"])
report("after import")
