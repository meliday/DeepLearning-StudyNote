# Deep Learning Study Notes

This repository documents my progress in learning deep learning through
first-principles implementation and controlled experiments. The main thread is
the hands-on study track in [`study_free/`](study_free/), which develops each
topic from a NumPy implementation into reproducible PyTorch experiments.

MIT 6.S191 provided the starting point and remains an important lecture
reference, while CS231n and other materials are used where they support the
implementation roadmap. The goal is to understand not only how a model runs,
but why it works, how to verify it, and how experimental choices affect the
result.

> This is an active study repository. Incomplete code and intermediate
> experiments may remain as part of the learning record.

## Current Progress

| Stage | Topic | Status |
|---|---|---|
| Reference | MIT 6.S191 fundamentals | In progress |
| 1 | Two-layer NumPy MLP and manual backpropagation | Completed |
| 2 | Numerical gradient checking | Completed |
| 3 | Learning-rate and initialization experiments | Completed |
| 4 | PyTorch fundamentals and autograd comparison | Completed |
| 5 | MNIST MLP | Next |
| 6 | CIFAR-10 CNN | Planned |
| 7 | Tiny PINN | Planned |

CS231n topics such as optimization, convolution, normalization, and
regularization will be studied alongside the implementation roadmap.

## Study Notes

### Core Implementation Track

[`study_free/`](study_free/) is the central learning track in this repository.
Its notebooks connect theory, implementation, verification, controlled
experiments, and written interpretation in one cumulative sequence.

| Notebook | Focus |
|---|---|
| [`day1_propagation.ipynb`](study_free/notes/day1_propagation.ipynb) | Two-layer NumPy MLP for XOR, manual forward/backward passes, SGD updates, and tensor-shape checks |
| [`day2_gradient.ipynb`](study_free/notes/day2_gradient.ipynb) | Numerical gradient checking, central differences, relative error, and diagnosis of ReLU kinks, reduction mismatches, and poor step sizes |
| [`day3_SgdLr.ipynb`](study_free/notes/day3_SgdLr.ipynb) | Controlled experiments on learning rates, weight initialization, train/validation splits, reproducibility, and multi-seed stability |
| [`day4_introPyTorch.ipynb`](study_free/notes/day4_introPyTorch.ipynb) | PyTorch tensor operations, `nn.Module`, an SGD training loop, evaluation mode, and comparison of manual NumPy gradients with autograd |

Generated tables and plots from the experiments are stored in
[`study_free/notes/results/`](study_free/notes/results/).

### Course References

[`MIT-6.S191/`](MIT-6.S191/) contains lecture-based notes and examples on:

- Perceptrons and multi-output networks
- Activation functions and loss functions
- Gradient descent and backpropagation
- Introductory PyTorch and TensorFlow syntax

## Key Results

### Manual Backpropagation

A two-layer NumPy MLP learned XOR without autograd:

```text
X (4, 2) -> Linear (2, 8) -> ReLU -> Linear (8, 1) -> Sigmoid
```

The BCE loss decreased from `0.695377` to `0.001799`, and the final predicted
classes matched `[0, 1, 1, 0]`.

### Gradient Checking

The analytical gradients were compared with central-difference numerical
gradients. All parameters passed the target relative error of `1e-5`.

| Parameter | Maximum relative error |
|---|---:|
| `W1` | `4.827e-07` |
| `b1` | `7.555e-07` |
| `W2` | `1.575e-08` |
| `b2` | `1.820e-09` |

The experiment also showed that a failed check does not always imply an
incorrect backward pass: non-differentiable ReLU points and poorly chosen
finite-difference step sizes can produce misleading errors.

### Optimization Experiments

Learning-rate and initialization experiments were run under controlled
conditions with fixed data, architecture, training duration, and random seeds.

- A learning rate of `0.1` converged reliably in the tested setup, while
  `1e-5` learned too slowly and `10.0` was unstable.
- Zero initialization failed because hidden units remained symmetric.
- Xavier and He initialization each succeeded on `9/10` tested seeds.
- A fast result from one seed was not sufficient evidence of a robust setup;
  validation performance and multi-seed success rates were also needed.

### PyTorch and Autograd

The NumPy binary classifier was rebuilt with `nn.Linear`, `nn.ReLU`,
`BCEWithLogitsLoss`, and `torch.optim.SGD`.

- After 1,000 epochs, the training loss reached `0.008252` and the validation
  loss reached `0.006746`.
- Training and validation accuracy both reached `100%` on the fixed split.
- Manual NumPy gradients matched PyTorch autograd for `W1`, `b1`, `W2`, and
  `b2` with an absolute tolerance of `1e-6`.
- The largest reported gradient difference was `6.985e-10` for `W1`.

## Repository Structure

```text
DeepLearning-StudyNote/
├── README.md
├── MIT-6.S191/
│   ├── README.md
│   ├── lecture1/
│   └── lecture_slide/
└── study_free/
    ├── README.md
    └── notes/
        ├── day1_propagation.ipynb
        ├── day2_gradient.ipynb
        ├── day3_SgdLr.ipynb
        ├── day4_introPyTorch.ipynb
        └── results/
```

## Study Principles

- Implement core operations and gradients from scratch before relying on
  higher-level abstractions.
- Change one major experimental variable at a time.
- Record seeds, hyperparameters, metrics, failure cases, and interpretations.
- Keep code executable and document enough context to reproduce key results.

## Next Steps

1. Add environment and dependency specifications.
2. Build a reproducible MNIST training and validation pipeline.
3. Compare MLP architectures and regularization choices on MNIST.
4. Move from fully connected networks to a CIFAR-10 CNN.

## References

- [MIT 6.S191: Introduction to Deep Learning](https://introtodeeplearning.com/)
- [Stanford CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/)
- [CS231n Course Notes](https://cs231n.github.io/)

## Notes

- Copyright for lecture materials belongs to their respective authors and
  institutions.
- External code is cited in the relevant file when referenced or modified.
