![[roofline_bf16.png|700]]
1. 左边的斜线是带宽上限：算力最多等于带宽乘以算术强度，即 $0.204\ \mathrm{TB/s}\times\mathrm{AI}$。数据搬运速度不够时，算力就无法继续提高。
2. 右边的平线是计算上限，实测 BF16 GEMM 峰值约为 $28.9\ \mathrm{TFLOP/s}$。超过拐点后，继续增加算术强度也不会突破这条计算上限。

ridge: $28.9/0.204\approx142\ \mathrm{FLOP/byte}$。

当 AI 低于 ridge 时，在算术强度不变的情况下，带宽会先于计算峰值成为限制；高于 ridge 后，瓶颈才转向计算能力。Softmax 还可能受指数/归约单元吞吐影响，因此分类需要结合 profiler 时间线看。

