IO and parsing account for a very large portion of the runtime: one run takes
about 26 seconds, and IO time takes about 25.5 seconds, meaning that IO and
parsing account for nearly 98% of the total runtime.

Step size has been set to 5.

Kernal latency and throughput:
(Latency is in seconds, throughput is in reviews / s)
|------------|---------|---------|---------|---------|----------|----------|
| Batch Size |       1 |      32 |    1024 |    2048 |    16384 |    65536 |
|============|=========|=========|=========|=========|==========|==========|
| Latency    | .000109 | .000109 | .000306 | .000317 |  .000693 |  .001647 |
|------------|---------|---------|---------|---------|----------|----------|
| Throughput |    9213 |  292997 | 3344132 | 6462688 | 23654422 | 39792488 |
|------------|---------|---------|---------|---------|----------|----------|

Interestingly, when batch size is 16384 or 65536 the kernal seems to skip the
final four lines, resulting in error rates that are much larger than they should
be (this is /not/ because the algorithm is incorrect, but rather because a 
single division operation at the very end of the kernal is being skipped). I am
not sure why this is the case.
