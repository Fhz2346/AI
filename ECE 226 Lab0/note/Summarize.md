# Roofline plot

|     | There are four dtypes available: <br>`ALL_DTYPES = [fp64, fp32, bf16, fp16]`<br>all of them could be measured |
| --- | ------------------------------------------------------------------------------------------------------------- |
| 1   | measure `ROOF_TFLOPS` based on `GEMM`                                                                         |
| 2   | measure bandwidth based on `copy`                                                                             |
|     | plot the roofline according to above 2 measured properties                                                    |
| 3   | measure GEMM of different size $n$ and plot them on roofline                                                  |
|     | ![[roofline_bf16.png\|500]]                                                                                   |
|     | The above part is based on measuring specific kernel operations(GEMM)                                         |
# Evaluate on network(ResNet)

| 1   | profile train step                                                                                                                                                            |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2   | categorise according to operation type:<br>`[optimizer, GEMM / conv, elementwise / norm, reduction, data movement]`<br>and tell which type of operation cost the most of time |
| 3   | calculate `grand` time used by GPU kernel and compare it with measured time on train step                                                                                     |
|     | tell if it is kernel-bound or gap-bound based on unaccounted: <br>unaccounted high means gap-bound                                                                            |
## 2
```
category                       ms    share   (device self time)
------------------------------------------------------
GEMM / conv                105.74    72.3%
elementwise / norm          16.07    11.0%
data movement               15.47    10.6%
reduction                    4.65     3.2%
optimizer                    3.65     2.5%
other                        0.71     0.5%
------------------------------------------------------
TOTAL                      146.28
```
## 3
```
wall clock per step :    86.16 ms   (p5 83.77, p95 97.95)
profiled self time  :    29.29 ms (device)
unaccounted         :    56.87 ms (66%)

FLOPs per step      :    214.0 GFLOP
achieved (wall)     :     2.48 TFLOP/s   = 26.9% of fp32 roof 9.23
achieved (kernels)  :     7.31 TFLOP/s   = 79.1% of fp32 roof 9.23
```
