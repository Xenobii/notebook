# Self Supervised Multi-Pitch Estimation using Note Vector Quantization

I'm cooking with forbidden ingredients 

## Pitch equivariance

Given $x^{(k)}$ the pitch shift of $x$ by $k$ semitones:

$$
    y^{(k)} = f_\theta(x^{(k)}) \newline
    z^{(k)} = \phi(y^{(k)}) \newline
$$

where

$$
    \phi: \mathbb{R}^N \rightarrow \mathbb{R}^2 \newline
$$

I therefore define pitch equivariance as:

$$
    L_{pitch\_equiv} = \min \left(\phi ( y^{(k)}), \phi (roll_k(y^{(0)})) \right)
$$

where $roll_k$ denotes the bin shift across the channel/notebook dimension by $k$.

This criterion enforces that each notebook vector represents a singular pitch class. $\phi$ is an octave invariant deterministic projection to a 2 dimensional plane, mapping each bin along the points of a circle. Since each pitch shift is known, w


## Notebook equivariance

After projecting $y$ to $z$, we can derive the centroid of this projection $c$. This essentialy creates a representative 2 dimensional feature, skewed towards the dominant pitch class index on the circle.

$$
    c = \sum_i^d z
$$

Converting this center $c$ to circular coordinates creates a radius r and angle $\theta_{z_i}$. Specifically, the angle should be directed to the dominant pitch class along the circle, located at index k. Therefore, we can define the following criterion for assigning specific pitch classes to each codebook:

$$
    L_{notebook} = \min(\theta_{z_i}, \theta_k)
$$


## Transformation invariance

Regular ass transformation invariance with gain and noise moving on

$$
    L_{invariance} = \min(z, T_\theta(z))
$$

where $T_\theta$ a set of invariant transforms.


## Reconstruction loss

*This is the most forbidden cooking method*

Each notebook vector $y$ represents the spectrum of a specific pitch class profile. The weighted sum of each element should produce a vector close to the original spectrogram. As for these weights, they can be parametrized by a single feed forward layer (a trainable set of neurons), in order to reconstruct the original data. This layer can be used during inference to determine the activation probability of each note class. 

$$
    L_{reconstruct} = \min \left( \sum_i^k w_iy_i,  x \right)
$$