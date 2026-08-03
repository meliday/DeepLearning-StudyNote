# Deep Learning Study Notes

This repository documents my journey of learning deep learning by implementing
and validating its core principles.

Starting with MIT 6.S191, I plan to progress from NumPy-based neural networks
and backpropagation to PyTorch, convolutional neural networks (CNNs), and a
small Physics-Informed Neural Network (PINN). Rather than simply summarizing
lecture material, I aim to:

- Implement and explain core operations and gradients from scratch
- Compare baselines and improvements under the same experimental conditions
- Document each project so that its results can be reproduced in other
  environments

> This repository is a work in progress. Incomplete code and intermediate
> experiments are intentionally preserved as part of the learning process.
> Validated results will be documented in each project's README and experiment
> logs.

## Learning Roadmap

| Stage | Topic | Main Deliverable | Status |
|---|---|---|---|
| 1 | Deep learning fundamentals | MIT 6.S191 lecture notes and examples | In progress |
| 2 | NumPy MLP | Two-layer neural network with forward and backward passes implemented from scratch | **Completed** |
| 3 | Gradient checking | Comparison of numerical and analytical gradients | **Completed** |
| 4 | PyTorch fundamentals | Tensors, autograd, `nn.Module`, and training loops | Next |
| 5 | MNIST MLP | Training and validation pipeline with result plots | Planned |
| 6 | CIFAR-10 CNN | Comparison between a baseline CNN and an improved model | Planned |
| 7 | Tiny PINN | Learning equation residuals, initial conditions, and boundary conditions | Planned |
| 8 | CS231n core topics | Backpropagation, optimization, CNNs, normalization, and regularization | Planned |

## Repository Structure

The repository currently has the following structure:

```text
MITDeepLearning-6.S191/
├── README.md
├── MIT-6.S191/
│   ├── README.md
│   ├── lecture1/
│   │   ├── mynote_lecture1.ipynb
│   │   ├── pytorch_syntax_guide.ipynb
│   │   ├── lecture_code/
│   │   └── imagefile/
│   └── lecture_slide/
└── study_free/
    ├── README.md
    └── notes/
        ├── day1_728.ipynb
        └── day2_801.ipynb
```

As the study progresses, the core implementations will be organized into the
following planned structure:

```text
00_numpy_mlp/
01_gradient_check/
02_pytorch_basics/
03_mnist_mlp/
04_cifar10_cnn/
05_tiny_pinn/
experiments/
reports/
slides/
```

## Current Contents

### MIT 6.S191

- Perceptrons and multi-output neural networks
- Activation functions
- Mean squared error and binary cross-entropy
- Gradient descent and backpropagation
- Introductory PyTorch and TensorFlow examples
- Personal lecture notes and additional experiments

### Free Study

This section contains independent implementations and explorations outside the
fixed lecture sequence.

[`day1_728.ipynb`](study_free/notes/day1_728.ipynb) covers the NumPy MLP:

- A two-layer NumPy MLP trained on XOR without autograd
- Forward propagation with Linear, ReLU, Sigmoid, and binary cross-entropy
- Manual backpropagation through Sigmoid-BCE, Linear, and ReLU operations
- SGD parameter updates and tensor-shape checks
- Notes connecting neural-network diagrams to `XW + b`, feature dimensions,
  batch dimensions, and bias broadcasting
- A training-loss history and final prediction check

[`day2_801.ipynb`](study_free/notes/day2_801.ipynb) covers gradient checking:

- Why a decreasing loss is not evidence that backpropagation is correct
- Central difference versus forward difference, and the origin of their
  `O(h^2)` and `O(h)` error terms
- Numerical gradients for every parameter, generalized into reusable
  `numerical_gradient` and `max_relative_error` helpers
- Three documented failure causes, each with its own error signature
- A deliberately injected `sum`/`mean` bug and the diagnosis that followed
- A diagnostic checklist for telling a real backpropagation bug apart from a
  measurement artifact

## Latest Milestone: Gradient Checking

The manual backpropagation from the previous milestone was validated against
numerical gradients on 2026-08-03, using the same XOR network and a fixed seed
of `42`. All four parameters pass well below the `1e-5` target.

| Parameter | Maximum relative error | Result |
|---|---:|---|
| `W1` | `4.827e-07` | PASS |
| `b1` | `7.555e-07` | PASS |
| `W2` | `1.575e-08` | PASS |
| `b2` | `1.820e-09` | PASS |

Each parameter is perturbed one element at a time, with the loss recomputed by
a full forward pass on both sides and the original value restored afterwards:

```python
numerical = (loss_plus - loss_minus) / (2 * h)

relative_error = (
    np.abs(analytical - numerical)
    / np.maximum(1e-8, np.abs(analytical) + np.abs(numerical))
)
```

### Documented Failure Causes

Passing the check turned out to be less instructive than the three ways it
failed. Each cause leaves a distinctive signature in the relative error.

| Cause | Error signature | Genuine backprop bug? |
|---|---|---|
| ReLU kink | exactly `1.0` | No — checked at a non-differentiable point |
| Reduction mismatch (`sum` vs `mean`) | a fixed constant, `0.6` when `N = 4` | Yes |
| Poorly chosen `h` | changes when `h` changes | No — a floating-point limitation |

**ReLU kink.** Because `X[0]` is `[0, 0]` and `b1` starts at zeros, `Z1[0]` is
exactly zero, so perturbing `b1` by `±h` straddles the ReLU corner. The
analytical gradient commits to one side while the central difference averages
both, and the relative error is exactly `1.0` wherever one side is zero.
Re-checking away from the corner, where `min |Z1|` is `0.00278` against
`h = 1e-5`, restores agreement at `7.555e-07` and proves the backward pass was
correct all along.

**Reduction mismatch.** Replacing `np.sum` with `np.mean` in `db2` scales the
gradient by `1/N`. The relative error is then exactly `3/5`, since
`|g - g/4| / (|g| + |g/4|)` reduces to `0.6`. No shape check can detect this
class of bug.

**Poorly chosen `h`.** Sweeping `h` produces a U-shaped error curve, because
two error terms move in opposite directions:

```text
total error(h) ≈ C1 * h^2  +  C2 * eps / h
                 truncation   cancellation
```

| `h` | Maximum relative error |
|---|---:|
| `1e-01` | `1.039e-05` |
| `1e-02` | `1.039e-07` |
| `1e-03` | `9.889e-10` |
| `1e-05` | `1.575e-08` |
| `1e-08` | `3.533e-05` |
| `1e-12` | `1.818e-01` |
| `1e-14` | `1.000e+00` |

The left half measures the theory directly: each tenfold reduction in `h`
divides the error by almost exactly `100`, which is what `O(h^2)` means. Past
the minimum the trend reverses, because `loss_plus` and `loss_minus` agree in
too many leading digits for `float64` to resolve their difference. At
`h = 1e-14` that difference is `-2.22e-16`, roughly twice the smallest gap the
format can represent, and the result is meaningless.

The working notebook is
[`study_free/notes/day2_801.ipynb`](study_free/notes/day2_801.ipynb).

## Previous Milestone: NumPy MLP

The first from-scratch neural-network milestone was completed on 2026-08-01.
The model uses two input features, one hidden layer with eight ReLU units, and
one sigmoid output.

```text
X (4, 2) → Linear (2, 8) → ReLU → Linear (8, 1) → Sigmoid → Y_hat (4, 1)
```

Training result on XOR:

| Metric | Result |
|---|---:|
| Initial BCE loss | `0.695377` |
| BCE loss at epoch 9,000 | `0.001799` |
| Final predicted classes | `[0, 1, 1, 0]` |
| Ground-truth classes | `[0, 1, 1, 0]` |

The implementation covers the complete learning cycle:

```text
forward pass → loss → backward pass → parameter update → repeat
```

Key gradients were derived and implemented manually:

```python
dZ2 = (Y_hat - y) / N
dW2 = A1.T @ dZ2
db2 = dZ2.sum(axis=0, keepdims=True)
dA1 = dZ2 @ W2.T

dZ1 = dA1 * (Z1 > 0)
dW1 = X.T @ dZ1
db1 = dZ1.sum(axis=0, keepdims=True)
```

The working implementation and accompanying explanations are recorded in
[`study_free/notes/day1_728.ipynb`](study_free/notes/day1_728.ipynb).

## Next Steps

The immediate follow-up work is to strengthen and validate the implementation:

1. Pass the loss function into `numerical_gradient` explicitly. It currently
   mutates a parameter array and relies on a module-level `compute_loss`
   reading the same object, so calling it with a copy silently returns zeros.
2. Warn when `min |Z1|` is close to `h`, so a ReLU kink is reported as a
   measurement artifact rather than a failure.
3. Extract the gradient-check helpers into a standalone, importable module.
4. Record environment and dependency versions for reproducibility.
5. Move to PyTorch tensors, autograd, `nn.Module`, and training loops, then
   cross-check the manual gradients against `torch.autograd`.

## Definition of Done

A core project is considered complete when it meets the following criteria:

- The code runs successfully from start to finish.
- The execution environment and commands are documented.
- The random seed and major hyperparameters are recorded.
- Learning curves or quantitative metrics are included.
- The implementation principles and major tensor shapes can be explained.
- Failure cases, limitations, and possible next experiments are documented.

Additional project-specific criteria:

- **NumPy MLP:** Implement the forward pass, backward pass, and parameter
  updates without autograd.
- **Gradient checking:** Achieve a relative error of approximately `1e-5` or
  lower for the main parameters.
- **MNIST:** Separate training and evaluation, report validation metrics, and
  provide reproducible commands.
- **CIFAR-10:** Compare a baseline with at least one improvement under the same
  conditions.
- **Tiny PINN:** Visualize the predicted solution and residuals while training
  with residual, initial-condition, and boundary-condition losses.

## Experiment Log

Each comparison experiment should change only one major variable at a time.
The following information will be recorded:

- Objective and hypothesis
- Related commit
- Dataset and data split
- Model and modifications
- Hyperparameters, random seed, and device
- Training, validation, and test results
- Interpretation, possible causes of failure, and next steps

## Reproducibility

The following will be added progressively to make each project reproducible:

- Fixed Python and library versions
- A `requirements.txt` or `environment.yml` file
- Dataset preparation instructions
- Training, evaluation, and visualization commands
- Shared random-seed configuration
- Default configuration files
- The commit and configuration used to produce representative results

Datasets, checkpoints, and large log files will not be stored directly in the
repository. Their download instructions or external locations will be
documented instead.

## CS231n Scope

Before participating in research, my goal is not necessarily to complete every
part of CS231n, but to be able to implement and explain its core topics:

- Computational graphs and backpropagation
- Optimization and learning-rate management
- Convolution, pooling, receptive fields, and output shapes
- Initialization, normalization, dropout, and regularization
- PyTorch-based training and validation loops
- Comparison and interpretation of baselines and improvements

Advanced topics such as detection, segmentation, visualization, and modern
architectures will be explored after the core implementations are complete.

## References

- [MIT 6.S191: Introduction to Deep Learning](https://introtodeeplearning.com/)
- [Stanford CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/)
- [CS231n Course Notes](https://cs231n.github.io/)

## Notes

- Copyright for lecture materials belongs to their respective authors and
  institutions.
- The goal is to focus on personal explanations, implementations, and
  experimental results rather than reproducing lecture materials verbatim.
- Sources will be cited in the relevant files or documents when external code
  is referenced or modified.
