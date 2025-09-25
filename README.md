# Dynamic Loss Weighting

PyTorch modules for training with **multiple loss terms**. Implements both:

* **Scaling Weights** – learn positive coefficients for each loss term.
* **Uncertainty Weights** – learn noise-based weights as in *[Kendall et al. (2017)](https://arxiv.org/abs/1705.07115)*.

---

## Why?

In multi-task setups, the naive sum

$$\mathcal{L} = \sum_i \mathcal{L}_i$$

assumes all tasks are equally important. However, different tasks can have different scales or difficulties.

* **Scaling Weights**:

  $$\mathcal{L} = \sum_i w_i \mathcal{L}_i, \quad w_i > 0$$

  (weights learned directly).

* **Uncertainty Weights** (Kendall et al.):

  $$\mathcal{L} = \sum_i \frac{1}{\sigma_i^2} \mathcal{L}_i + \log(\sigma_i)$$

  (weights tied to task uncertainty, with regularization).

---

## Usage

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)

# Option 1: Scaling weights
multi_loss = VanillaMultiLoss(n_losses=2).to(device)

# Option 2: Uncertainty-based weighting
multi_loss = MultiNoiseLoss(n_losses=2).to(device)

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': multi_loss.parameters()}], lr=0.001)

loss_1 = cross_entropy(pred1, target1)
loss_2 = cross_entropy(pred2, target2)

loss = multi_loss([loss_1, loss_2])
loss.backward()
optimizer.step()
```

⚠️ Include `multi_loss.parameters()` in your optimizer.

---

## Note

In the paper by Kendall et al, regression tasks are weighted by $\frac{1}{2\sigma_i^2}$ and include a $\log(\sigma_i)$ term. This implementation does not apply the factor of $0.5$ (which is directed for regression tasks) but can be adapted to do so.
