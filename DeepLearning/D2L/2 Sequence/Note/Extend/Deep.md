
$$\begin{aligned}
\mathrm{H}_t^{l}&=\tanh(\mathrm{H}_t^{l-1}\mathrm{W}_{xh}^{l}+\mathrm{H}_{t-1}^{l}\mathrm{W}_{hh}^{l}+\mathbf{b}_h^{l})\\
\mathrm{O}_t&=\mathrm{H}_t^{(L)}\mathrm{W}_{hq}+\mathbf{b}_q
\end{aligned}$$

![[DeepRNN.png|300]]
