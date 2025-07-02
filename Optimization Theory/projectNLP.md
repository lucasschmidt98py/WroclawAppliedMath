# Data

|  Plant     |  1 |  2 |  3 |
|-------|---------|---------|---------|
| $c_j$ | 0.00006 | 0.00002 | 0.00003 |
|$p_j$ T/d| 220 | 160 | 180 |

### Definitions

* $i$ - Day Index $i \in [1,2,...,20]$
* $j$ - Plant Index $j \in [1,2,3]$
* $p_j$ - Process capacity of plant $j$
* $x_i^j$ - Ammount of beets processed at day i by Plant j
* $s_i$ - Ammount of beets stored at day i
* $L_j(x^j_i)$ - Losses from the storage of beets accepted for processing on day $i$ at plant $j$

### Minimize

$$ \sum_{i=1}^{20} \sum_{j=1}^{3} L_j(x^j_i) $$

Where

$$  L_j(x^j_i) = c_j x^j_i \frac{ t^j_i + t^j_{i+1}- 2i}{2} $$

### Constrains

$$ \sum_{j=1}^3 x_i^j + s_{i} = s_{i-1} + 800$$

$$\sum_{j=1}^3 x_i^j \leq 560$$

$$ \sum_{i=1}^{20} \sum_{j=1}^{3} x^j_i = 160000 $$

$$ 0 \leq x_i \leq p_i \quad \forall i,j$$

$$ 0 \leq s_i \quad \forall i$$

$$ s_0 = 0 $$

