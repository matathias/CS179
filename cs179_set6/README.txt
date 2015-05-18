IO and parsing account for a very large portion of the runtime: one run takes
about 26 seconds, and IO time takes about 25.5 seconds, meaning that IO and
parsing account for nearly 98% of the total runtime.

Step size has been set to 5.

Kernal latency and throughput:
(Latency is in seconds, throughput is in reviews / s)
|-------------------------------------------------------------------------
| Batch Size |       1 |      32 |    1024 |    2048 |   16384 |   65536 |
|============|=========|=========|=========|=========|=========|=========|
| Latency    |
|------------|---------|---------|---------|---------|---------|---------|
| Throughput |
|------------------------------------------------------------------------|
