
| 1   | $\tilde{\mathrm{C}}_t$ | $\tanh(\mathrm{X}_t\mathrm{W}_{xc}+\mathrm{H}_{t-1}\mathrm{W}_{hc}+\mathbf{b}_c)$  |
| --- | ---------------------- | ---------------------------------------------------------------------------------- |
| 1.  | $\mathrm{I}_{t}$       | $\sigma(\mathrm{X}_t\mathrm{W}_{xi}+\mathrm{H}_{t-1}\mathrm{W}_{hi}+\mathbf{b}_i)$ |
| 1.  | $\mathrm{F}_t$         | $\sigma(\mathrm{X}_t\mathrm{W}_{xf}+\mathrm{H}_{t-1}\mathrm{W}_{hf}+\mathbf{b}_f)$ |
| 1.  | $\mathrm{O}_{t}$       | $\sigma(\mathrm{X}_t\mathrm{W}_{xo}+\mathrm{H}_{t-1}\mathrm{W}_{ho}+\mathbf{b}_o)$ |
| 2   | $\mathrm{C}_t$         | $\mathrm{F}_t\odot\mathrm{C}_{t-1}+\mathrm{I}_t\odot\tilde{\mathrm{C}}_t$          |
| 3   | $\mathrm{H}_t$         | $\mathrm{O}_t\odot\tanh(\mathrm{C}_t)$                                             |

| [bs,dH] | H   | $\mathrm{H}_{t-1}$ |
| ------- | --- | ------------------ |
| [bs,dH] | I   | $\mathrm{I}_{t}$   |
| [bs,dH] | F   | $\mathrm{F}_t$     |
| [bs,dH] | O   | $\mathrm{O}_{t}$   |
| [bs,dH] | C   | $\mathrm{C}_t$     |
| [bs,dH] | H   | $\mathrm{H}_t$     |
