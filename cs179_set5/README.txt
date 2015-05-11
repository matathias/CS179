Time spent on set: ~7 Hours

For a batch size of 2048, the single batch latency was 2.789 milliseconds.
This latency decreases as batch size decreases.

For a batch size of 2048, the throughput is 12389.3 reviews/second.
The throughput decreases as batch size decreases, but does not seem to change
significantly if increased past 2048.
Attempting to use the loader.py script on minuteman failed with an
ImportError: No modeul named sklearn.feature_extraction.text
That said, with these kind of throughputs use a pre-compute LSA should be more
efficient as that is simply less overhead.

For k=50, clusters uses a bandwidth of .435844 GB/s. This does not seem to
saturate the PCI-E interface (there are only two streams after all). The 
bottleneck switches between the host-device IO to the kernal at above k=800, 
which has a bandwidth of 1.471887 GB/s. Notably, while trying different values 
of k = 50 * 2^x, the bandwidth increased somewhat rapidly until around k = 200 
and then began increasing more slowly; for instance the difference in bandwidth
between k=50 and k=100 was nearly .3 GB/s, but the difference in bandwidth
between k=400 and k=800 was only about .2 GB/s, despite the larger difference in
k. Additionally any value of k much greater than 800 (such as 825) yeilded a 
lower bandwidth than at k=800.

I think performance could definitely be improved with multiple GPUs. The IO
bottleneck is not reached until a rather high value of k, so more GPUs could be
added without worrying about the IO bottleneck - and more GPUs means more
threads and streams, which would help to hide the batch latency.
