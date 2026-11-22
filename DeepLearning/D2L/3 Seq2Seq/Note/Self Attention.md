
| Let  | $[\mathbf{x}_i]_1^n$                                                   | sequence of **input** tokens                                                                                                          |
| ---- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| with | $n$                                                                    | sequence length                                                                                                                       |
| with | $\mathbf{x}_i$                                                         | $\in\mathbb{R}^d$                                                                                                                     |
|      | $\begin{bmatrix}\mathbf{q}_i\\\mathbf{k}_i\\\mathbf{v}_i\end{bmatrix}$ | $\begin{bmatrix}\mathrm{W}_\mathbf{q}\mathbf{x}_i\\\mathrm{W}_\mathbf{k}\mathbf{x}_i\\\mathrm{W}_\mathbf{v}\mathbf{x}_i\end{bmatrix}$ |
| Cstr | $\mathcal{D}$                                                          | $\{(\mathbf{k}_1,\mathbf{v}_1),\ldots(\mathbf{k}_m,\mathbf{v}_m)\}$                                                                   |
| Let  | $\mathbf{y}_i$                                                         | sequence of **output** tokens                                                                                                         |
|      | $\mathbf{y}_i$                                                         | $\mathrm{Attention}(\mathbf{q}_i,\mathcal{D})$                                                                                        |
