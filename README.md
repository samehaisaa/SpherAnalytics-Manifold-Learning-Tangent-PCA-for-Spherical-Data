## Screenshot

Data generated on a 2D spherical manifold under the von Mises distribution. It shows the tangent plane at the Fréchet mean and the projection via the log map and PCA, highlighting the principal axis of variation. The data is then projected back using the exp map to preserve the geometry.

![Spherical Data with Tangent Plane and PCA](assets/tangent.png)

### Formulas

- **Log Map**: \( \log_p(q) = \frac{\theta}{\sin(\theta)} (q - \cos(\theta)p) \)
- **Exp Map**: \( \exp_p(v) = \cos(\|v\|)p + \sin(\|v\|)\frac{v}{\|v\|} \)
- **Von Mises Distribution**: \( f(\theta|\mu,\kappa) = \frac{e^{\kappa \cos(\theta - \mu)}}{2\pi I_0(\kappa)} \)

