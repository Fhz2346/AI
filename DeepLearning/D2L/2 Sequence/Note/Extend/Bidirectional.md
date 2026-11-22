
![[{5C8FC835-77B8-4296-BFF7-A47922283655}.png|400]]
使用来自序列两端 (过去和未来) 的信息来估计输出
不适合解决预测问题, 因为没有未来信息
$$\begin{aligned}
&\begin{cases}\begin{aligned}
\overrightarrow{\mathbf{H}}_t&=\phi(\mathbf{X}_t\mathbf{W}_{xh}^{(f)}+\overrightarrow{\mathbf{H}}_{t-1}\mathbf{W}_{hh}^{(f)}+\mathbf{b}_h^{(f)})\\
\overleftarrow{\mathbf{H}}_t&=\phi(\mathbf{X}_t\mathbf{W}_{xh}^{(b)}+\overleftarrow{\mathbf{H}}_{t+1}\mathbf{W}_{hh}^{(b)}+\mathbf{b}_h^{(b)})
\end{aligned}\end{cases}\\
&\begin{cases}\begin{aligned}
\mathbf{H}_t &= \mathbf{hstack}\left[ \overrightarrow{\mathbf{H}}_t,\overleftarrow{\mathbf{H}}_t\right]\\
\mathbf{O}_t&=\mathbf{H}_t\mathbf{W}_{hq}+\mathbf{b}_q
\end{aligned}\end{cases}
\end{aligned}$$
