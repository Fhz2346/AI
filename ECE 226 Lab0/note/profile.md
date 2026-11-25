
| Abbr | macs | Multiply-Accumulate                            |
| ---- | ---- | ---------------------------------------------- |
|      |      | 1 MAC = 2 FLOP                                 |
|      |      | One forward+backward is ~3x the forward FLOPs. |
# Self&Total
一个算子内部往往还会调用其他算子。以卷积前向为例，层级关系大致如下：
```
aten::conv2d <-父算子
 └─ aten::convolution
     └─ aten::_convolution
         └─ aten::cudnn_convolution <-子算子
```

| Self  | 只算这个算子自己的代码花的时间, 扣掉它调用的子算子 |
| ----- | -------------------------- |
| Total | 从进入这个算子到离开为止的全部时间, 包括所有子算子 |

| Self CPU     | 这个算子自身花的 CPU 时间，所有调用累加             |
| ------------ | ---------------------------------- |
| Self CPU %   | Self CPU 占全部 Self CPU 总时间的比例       |
| CPU total    | 包含子算子在内的 CPU 总时间，所有调用累加            |
| CPU total %  | CPU total 占 Self CPU 总时间的比例        |
| CPU time avg | 每次调用的平均 CPU 时间，等于 CPU total 除以调用次数 |

If 'unaccounted' is large, the fix is host-side (dataloader workers, pinned memory, launch overhead, CUDA/HIP graphs) -- not a faster kernel.

On a CPU-only profile the two figures can also exceed each other, because host operator time nests; say so rather than reporting a negative gap as a result.
