
| Abbr | bw     | band width                                |
| ---- | ------ | ----------------------------------------- |
| Abbr | tflops | Tera Floating-point Operations Per Second |
| Abbr | ai     | arithmetic intensity                      |

|     |        | Unit      | Rpst                         |
| --- | ------ | --------- | ---------------------------- |
|     | volume | bytes     | volume of data move          |
|     | bw     | bytes/s   | speed of data move           |
|     | ai     | FLOP/byte | 每从内存搬运 1 Byte 数据，平均完成多少次浮点运算 |

| Abbr | GEMM              | General Matrix Multiply                                                                                                                       |
| ---- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| When | $\mathrm{size}$   | $n$                                                                                                                                           |
| Then | $\mathrm{flop}$   | $n^2\times2n=2n^3$                                                                                                                            |
| expl |                   | 每个输出元素做 $n$ 次乘加, 乘加算 $2$ 次                                                                                                                    |
| Then | $\mathrm{volume}$ | $3n^2\times\text{bytes\_per\_elem}$                                                                                                           |
| expl |                   | 3 = 2 input(read) + 1 output(write)                                                                                                           |
|      | $\mathrm{ai}$     | $\displaystyle\frac{\mathrm{flop}}{\mathrm{volume}}=\frac{2n^3}{3n^2\times\text{bytes\_per\_elem}}=\frac{2n}{3\times\text{bytes\_per\_elem}}$ |
