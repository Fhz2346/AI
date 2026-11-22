
|     | $\mathcal{B}$                               | minibatch                                                                                                                                             |
| --- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
|     | $\hat{\boldsymbol{\mu}}_{\mathcal{B}}$      | $\displaystyle\frac{1}{\vert\mathcal{B}\vert}\sum_{\mathbf{x}\in\mathcal{B}}\mathbf{x}$                                                               |
|     | $\hat{\boldsymbol{\sigma}}_{\mathcal{B}}^2$ | $\displaystyle\frac{1}{\vert\mathcal{B}\vert}\sum_{\mathbf{x}\in\mathcal{B}}(\mathbf{x}-\hat{\boldsymbol{\mu}}_{\mathcal{B}})^2+\epsilon$             |
|     | $\mathrm{BN}(\mathbf{x})$                   | $\displaystyle\boldsymbol{\gamma}\odot\frac{\mathbf{x}-\hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}}+\boldsymbol{\beta}$ |
Two data points with only the difference of mean and variance but same shape should be deem as the same distribution
$\boldsymbol{\gamma},\boldsymbol{\beta}$ to nullify the effect of affine transformation
