# Core Deep Learning Study Track

스스로 짠 계획에 맞추어, 학습한 내용을 기록하고 나중에 참고할 수 있게 쓰는용도

## Learning Flow

```text
수식과 개념
    -> NumPy 직접 구현
    -> gradient와 shape 검증
    -> 학습 조건 비교 실험
    -> PyTorch 및 autograd와 대조
    -> 실제 데이터셋과 더 깊은 모델로 확장
```

## Notebook Index

| Day | Notebook | 핵심 내용 | 상태 |
|---|---|---|---|
| 1 | [`day1_propagation.ipynb`](notes/day1_propagation.ipynb) | 2-layer NumPy MLP, forward/backward propagation, XOR 학습 | 완료 |
| 2 | [`day2_gradient.ipynb`](notes/day2_gradient.ipynb) | central difference 기반 gradient checking과 실패 원인 분석 | 완료 |
| 3 | [`day3_SgdLr.ipynb`](notes/day3_SgdLr.ipynb) | learning rate와 weight initialization 비교, multi-seed 검증 | 완료 |
| 4 | [`day4_introPyTorch.ipynb`](notes/day4_introPyTorch.ipynb) | PyTorch tensor와 `nn.Module`, 학습 루프, NumPy gradient와 autograd 비교 | 완료 |

## Working Method

- 핵심 연산은 가능한 한 NumPy로 먼저 구현한다.
- 라이브러리 구현을 사용하기 전에 tensor shape와 gradient 흐름을 확인한다.
- 실험에서는 한 번에 하나의 주요 조건만 바꾸고 seed와 평가 기준을 기록한다.
- 단일 실행 결과보다 validation 결과와 여러 seed에서의 안정성을 우선한다.
- 코드는 실행 결과와 함께 남기고, 성공뿐 아니라 실패 원인도 해석한다.

## Next Direction

다음 단계는 재현 가능한 실행 환경을 정리하고 MNIST MLP 학습 파이프라인을
구축하는 것이다. 이후 regularization과 architecture를 비교한 뒤 CIFAR-10
CNN과 작은 PINN으로 확장한다.

강의별 원본 정리와 참고 자료는 [`../MIT-6.S191/`](../MIT-6.S191/)에서
분리해 관리한다.
