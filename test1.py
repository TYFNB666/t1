# import torch
#
# if torch.cuda.is_available():
#     print("CUDA is available on this device.")
# else:
#     print("CUDA is not available on this device.")
# import torch
# print(torch.__version__)
import psutil

# 获取当前系统的内存使用情况
memory = psutil.virtual_memory()
print("Total memory: ", memory.total)
print("Available memory: ", memory.available)

# 获取当前进程的内存使用情况
process = psutil.Process()
process_memory = process.memory_info().rss
print("Current process memory usage: ", process_memory)
