IO and parsing account for a very large portion of the runtime: one run takes
about 26 seconds, and IO time takes about 25.5 seconds, meaning that IO and
parsing account for nearly 98% of the total runtime.

Step size has been set to 5.

Kernal latency and throughput:
(Latency is in seconds, throughput is in reviews / s)
|------------|---------|---------|---------|---------|----------|----------|
| Batch Size |       1 |      32 |    1024 |    2048 |    16384 |    65536 |
|============|=========|=========|=========|=========|==========|==========|
| Latency    | .000109 | .000109 | .000306 | .000317 |  .000693 | segfault |
|------------|---------|---------|---------|---------|----------|----------|
| Throughput |    9213 |  292997 | 3344132 | 6462688 | 23654422 | segfault |
|------------|---------|---------|---------|---------|----------|----------|

When run with valgrind, the segfault for batch size = 65536 seems to be caused
by a lack of good memory. The exact error is:
==2561== Process terminating with default action of signal 11 (SIGSEGV)
==2561==  Bad permissions for mapped region at address 0x504600200
==2561==    at 0x4035E6: cudaClassify(float*, int, float, float*) 
                    (in /home/dwarrick/Documents/CS179/cs179_set6/classify)
==2561==    by 0x402B27: classify(std::istream&, int) 
                    (in /home/dwarrick/Documents/CS179/cs179_set6/classify)
==2561==    by 0x401E6E: main 
                    (in /home/dwarrick/Documents/CS179/cs179_set6/classify)

I am unsure of how to fix this.
