# DIFFERENTIALLY PRIVATE LEARNING NEEDS BETTER FEATURES (OR MUCH MORE DATA)

```console
@misc{tramèr2021differentially,
      title={Differentially Private Learning Needs Better Features (or Much More Data)}, 
      author={Florian Tramèr and Dan Boneh},
      year={2021},
      eprint={2011.11660},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Introduction:

The paper discusses the importance of hand-crafted features with respect to privacy enabled machine learning, specifically differential privacy. 

## Contributions:

The paper proves the importance of hand-crafted features, they are able to support their claims with results, where they are able to:

* On CIFAR-10 they exceed the accuracy reported by Papernot et al. (2020b) while simultaneously improving the provable DP-guarantee by 130 times.
* On MNIST, they match the privacy-utility guarantees obtained with PATE (Papernot et al., 2018) without requiring access to any public data.

They are able to emperically support their statement by showing the test results for handcrafted scatternet predictions and previous work employing the same epsilon values.

![Image 1](./images/img1.PNG)

### Discussion Points:

1. Learning Rate: Low lr leads to convergence with or without privacy.
2. Handcrafted Features: lead to faster convergence during non-private training and hence even while training with privacy, they are able to outperform non-handcrafted feature implementations.

Question is how to overcome performance gap between handcrafted and automated feature learning.

1. More data: They are able to reinforce the claim that more data could be an answer to outperforming handcrafted features. 
2. Differentially Private Transfer Learning: can display strong privacy at only a minor cost in accuracy, making this the more favourable option (CIFAR-10 model to 92.7% accuracy for a DP budget of ε = 2).

## Aim:

Handcrafted features can allow easier learning than end-to-end deep learning.

### Scattering Networks:

The use of data independent filters for feature extraction allows us to skip the step of end-to-end learning and acheive greater performance with just shallow networks.

### Differentially Private ScatterNet Classifiers:

#### DP-SGD algorithm:
1. Batches of size `B` are sampled randomly.
2. Gradients clipped to Norm `C`.
3. Gaussian noise of variance (σ<sup>2</sup>C<sup>2</sup>/B<sup>2</sup>) is added to the mean gradient

#### Normalizing 
Normalizing is important for high performance output. Two approaches are considered:
1. Group Normalization, whereby channels of `S(x)` are split into G groups and normalized.
2. Data Normalization, whereby each channel is normalized which incurs a privacy cost. - Performs better

## Experimental Setup:

1. They are able to show that using ScatterNets is convenient and robust to hyperparmeter searches. They also talk about privacy leakage due to hyperparameter search, as that iself is based on the data provided. 
2. Much more accurate than end-to-end learning.

## Why do these conclusions and results happen?

1. Smaller Models are not easier to train: ScatterNets are smaller models, but that doesn't warrant their high perfomance rather ScatterNets tend to have higher number of parameters than typical/comparable CNNs.
2. Models with handcrafted features converge faster without privacy: DP-SGD requires a lower LR to average out noise. With a large learning rate, a disparity can be seen between when privacy is utilized and when it is not. However, it is the opposite when we talk about a smaller LR, as both of them converge at roughly the sametime. 

## Conclusions:

1. Access to a larger private training set
2. Access to a public image dataset from a different distribution

1. Larger private datasets allow DP-SGD to be run for more steps at a fixed privacy budget and noise level.
2. Transfer Learning: better features from public data.
