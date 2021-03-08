import ctypes
import mmap

addr= mmap.mmap(
None, 4096,
mmap.PROT_READ   | mmap.PROT_WRITE | mmap.PROT_EXEC|
mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,        -1, 0)


# x86-64 enconding of 
#     mov rax, 42
#     ret
# 
code = b"\x48\xC7\xC0\x2A\x00\x00\x00\xC3"
ctypes.memmove(addr, code,len(code))

meaning = ctypes.cast(addr,ctypes.CFUNCTYPE(ctypes.c_long))