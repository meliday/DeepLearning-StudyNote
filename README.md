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
| 3 | Gradient checking | Comparison of numerical and analytical gradients | Next |
| 4 | PyTorch fundamentals | Tensors, autograd, `nn.Module`, and training loops | Planned |
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
        └── day1_728.ipynb
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
fixed lecture sequence. The current notebook includes:

- A two-layer NumPy MLP trained on XOR without autograd
- Forward propagation with Linear, ReLU, Sigmoid, and binary cross-entropy
- Manual backpropagation through Sigmoid-BCE, Linear, and ReLU operations
- SGD parameter updates and tensor-shape checks
- Notes connecting neural-network diagrams to `XW + b`, feature dimensions,
  batch dimensions, and bias broadcasting
- A training-loss history and final prediction check

## Latest Milestone: NumPy MLP

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

1. Refactor the NumPy MLP into initialization, forward, loss, backward, update,
   and prediction functions.
2. Add numerical gradient checking and target a relative error of `1e-5` or
   lower.
3. Record environment and dependency versions for reproducibility.
4. Move to PyTorch tensors, autograd, `nn.Module`, and training loops.

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
