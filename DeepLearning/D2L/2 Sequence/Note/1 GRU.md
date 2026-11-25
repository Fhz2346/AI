
| 1   | $\mathrm{R}_t$         | $\sigma(\mathrm{X}_t\mathrm{W}_{xr}+\mathrm{H}_{t-1}\mathrm{W}_{hr}+\mathbf{b}_r)$                   |
| --- | ---------------------- | ---------------------------------------------------------------------------------------------------- |
| 2   | $\tilde{\mathrm{H}}_t$ | $\tanh(\mathrm{X}_t\mathrm{W}_{xh}+(\mathrm{R}_t\odot\mathrm{H}_{t-1})\mathrm{W}_{hh}+\mathbf{b}_h)$ |
| 2   | $\mathrm{Z}_t$         | $\sigma(\mathrm{X}_t\mathrm{W}_{xz}+\mathrm{H}_{t-1}\mathrm{W}_{hz}+\mathbf{b}_z)$                   |
| 3   | $\mathrm{H}_t$         | $\mathrm{Z}_t\odot\mathrm{H}_{t-1}+(1-\mathrm{Z}_t)\odot\tilde{\mathrm{H}}_t$                        |
