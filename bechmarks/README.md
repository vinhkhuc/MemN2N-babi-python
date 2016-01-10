This page contains benchmark results to compare this Python implementation and the original Matlab code on
the [bAbI tasks](http://fb.ai/babi). 

These results are computed using default configuration, i.e. 3 hops, position encoding (PE), 
linear start training (LS), random noise (RN) and adjacent weight tying.

The values are test error rate (%).

|      |            MemN2N-babi-matlab           ||      MemN2N-babi-python (this repo)      ||
| Task | 3 hops PE LS RN  | 3 hops PE LS RN JOINT | 3 hops PE LS RN   | 3 hops PE LS RN JOINT | 		
|:----:|:----------------:|:---------------------:|:-----------------:|:---------------------:|
|  1   |       0.1        |           0.0         |       0.5         |         0.1           |
|  2   |       8.2        |          13.1         |       8.9         |        16.6           |
|  3   |      41.8        |          23.4         |      41.4         |        26.3           |
|  4   |       4.4        |           5.9         |       7.3         |        11.3           |
|  5   |      13.7        |          12.9         |      12.2         |        14.4           |
|  6   |       7.9        |           3.7         |       7.7         |         2.8           |
|  7   |      20.3        |          22.9         |      19.8         |        16.0           |
|  8   |      11.7        |           9.1         |      12.9         |        10.1           |
|  9   |      13.6        |           2.7         |      13.9         |         2.3           |
|  10  |       9.7        |           7.2         |      18.9         |         6.5           |
|  11  |       0.7        |           0.8         |       0.5         |         1.2           |
|  12  |       0.0        |           0.1         |       0.2         |         0.2           |
|  13  |       0.3        |           0.1         |       0.9         |         0.5           |
|  14  |       0.9        |           4.2         |       9.0         |         5.5           |
|  15  |       0.0        |           0.0         |       0.0         |         0.3           |
|  16  |       0.4        |           1.2         |       0.6         |         2.1           |
|  17  |      51.0        |          43.6         |      49.1         |        42.6           |
|  18  |      10.5        |          10.5         |      10.4         |         9.0           |
|  19  |      81.4        |          86.7         |      90.8         |        90.2           |
|  20  |      0.0         |           0.0         |       0.0         |         0.2           |
|:----:|:----------------:|:---------------------:|:-----------------:|:---------------------:|
| Mean |      13.8        |          12.4         |      15.2         |        12.9           |

