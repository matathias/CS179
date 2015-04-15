./transpose output:

Size 512 naive CPU: 0.280448 ms
Size 512 GPU memcpy: 0.032448 ms
Size 512 naive GPU: 0.090528 ms
Size 512 shmem GPU: 0.049504 ms
Size 512 optimal GPU: 0.048032 ms

Size 1024 naive CPU: 2.081952 ms
Size 1024 GPU memcpy: 0.081184 ms
Size 1024 naive GPU: 0.303968 ms
Size 1024 shmem GPU: 0.154752 ms
Size 1024 optimal GPU: 0.152672 ms

Size 2048 naive CPU: 37.262913 ms
Size 2048 GPU memcpy: 0.265344 ms
Size 2048 naive GPU: 1.158048 ms
Size 2048 shmem GPU: 0.515616 ms
Size 2048 optimal GPU: 0.516544 ms

Size 4096 naive CPU: 155.895493 ms
Size 4096 GPU memcpy: 0.999456 ms
Size 4096 naive GPU: 4.124736 ms
Size 4096 shmem GPU: 1.984224 ms
Size 4096 optimal GPU: 1.925184 ms


Bank conflicts were avoided by using a shared memory stride length of 1 when
loading into the memory, and by using a stride length of 65 when writing to
global memory. This length of 65 required padding, and thus there are 64 unused
slots in the shared memory array.

Time to complete:
Part 1: 2 hours
Part 2: 7 hours
