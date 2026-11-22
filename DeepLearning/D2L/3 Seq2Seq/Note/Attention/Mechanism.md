
| Let      | $\mathcal{D}$                     | $\{(\mathbf{k}_1,\mathbf{v}_1),\ldots(\mathbf{k}_m,\mathbf{v}_m)\}$                                  |
| -------- | --------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Let      | $\mathbf{q}$                      | Query                                                                                                |
| which    |                                   | is the same kind as $\mathbf{k}$                                                                     |
| **Want** | $\mathbf{v}$                      | map of $\mathbf{q}$ in $\mathcal{D}$                                                                 |
| Def      | $a(\mathbf{q},\mathbf{k}_i)$      | attention **score**(scalar)                                                                          |
| which    |                                   | represent the similarity of $\mathbf{q}$ and $\mathbf{k}$                                            |
|          | $\alpha(\mathbf{q},\mathbf{k}_i)$ | $\mathrm{softmax}(a(\mathbf{q},\mathbf{k}_i))$                                                       |
| Call     | $\alpha(\mathbf{q},\mathbf{k}_i)$ | attention **weight**(scalar)                                                                         |
| with     |                                   | $\alpha(\mathbf{q},\mathbf{k}_i)\geq0\quad\forall i$                                                 |
| with     |                                   | $\sum_i\alpha(\mathbf{q},\mathbf{k}_i)=1$                                                            |
| ==Conc== | $\mathbf{v}$                      | $\mathrm{Attention}(\mathbf{q},\mathcal{D})=\sum_{i=1}^m\alpha(\mathbf{q},\mathbf{k}_i)\mathbf{v}_i$ |
