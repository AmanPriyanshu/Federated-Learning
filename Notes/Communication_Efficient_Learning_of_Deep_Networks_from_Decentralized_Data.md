Federated learning (FL) is a machine learning setting where many clients (e.g. mobile devices or whole organizations) collaboratively train a model under the orchestration of a central server (e.g. service provider), while keeping the training data decentralized.

Federated Learning is an up and coming field in the world of Privacy Enabled Machine Learning (some others include Homomorphic Encryptions and Differential Privacy). 

Introduced in the paper 

### (`Communication-Efficient Learning of Deep Networks from Decentralized Data`)[https://arxiv.org/pdf/1602.05629.pdf].

#### AIM:

The paper talks about the use of Mobile Devices as a source of personalized data, which can find various uses in Language Models, Speech Models, etc. It brings about the conflict of privacy breach that use of such data would bring. Here in, they advocate an alternative which utilizes Model Aggregation and Localized Training, thereby allowing privacy preservation. Another path of exploration taken by the paper is implementation of aforementioned model in a Non-IID Data Distribution setting.

#### Federated Optimization:

1. Non-IID:

Local data, is user based, that is to say that, it may or may not represent the general population. Each user is their own individual generating their own dataset, this concept of data distribution is Non-IID.

2. Unbalanced:

Some users generated much more data than others. This may be due to more use of the particular device/service, allowing for further collection of dataset.

3. Massively distributed:

Number of Clients used during aggregation is larger than the average number of samples of dataset they generate individually.

4. Limited communication 

Communication issues/Availability of Clients.

#### Experimental Environment and Algorithm:

* *STEP1:* K Clients with a fixed local dataset.
* *STEP2:* Random fraction C is selected for aggregation.
* *STEP3:* Trains on data locally.
* *STEP4:* Returns updates model for aggregation. 
* *STEP5:* Repeats from Step 2.

#### Algorithms:

System Requirements:

* Convexity NN
* Number of clients is much smaller than the number of examples per client
* IID fashion
* Each node has an identical number of data points

```python
aggregated_weights = (1/N) * sum(number_of_samples_client_k * client_weight)
```

where,

```python
client_weight = (1/number_of_samples_client_k) * sum(sample_weight)
```

Therefore,

```python
aggregated_weights = (1/N) * sum(client_weight)
```

##### TRIALS:
1. SGD + FL:

SGD applied to FL by employing batchwise gradient calculation every round of communication.
* +ves: Computationally Efficient
* -ves: Large number of epochs required

2. FederatedSGD:

* *STEP1:* select a C-fraction of clients on each round
* *STEP2:* compute the gradient of the loss over all the data held by these clients.
* *STEP3:* Aggregation: `updated_weight ← weight - lr * sum( number_of_samples_in_client_k * gradient_of_client_k )/N `

3. FederatedAveraging:

Assumptions:

* loss surfaces of sufficiently over-parameterized NNs are less prone to bad local minima.
* same random initialization

##### ALGORITHM - FederatedAveraging:

Assumption: The K clients are indexed by k; B is the local minibatch size, E is the number of local epochs, and η is the learning rate.

Server executes:
* *Step1:* initialize w0
* *Step2:* for each round t = 1, 2, . . . do
* *Step3:* 		m ← max(C · K, 1)
* *Step4:* 		St ← (random set of m clients)
* *Step5:* 		for each client k ∈ St in parallel do
* *Step6:* 			client_weight ← ClientUpdate(k, aggregated_weight)

ClientUpdate(k, w): // Run on client k
* *Step1:* B ← (split Pk into batches of size B)
* *Step2:* for each local epoch i from 1 to E do
* *Step3:* 		for batch b ∈ B do
* *Step4:* 			w ← w − η * gradient

#### Experimental Results:

NON-IID division: 

* Sort Dataset
* Divide it into N shards
* Assign each client k, int(N/C) shards

This will result in not all clients having all the labels, i.e. the initial few will only have label_1, whereas the last few will only have label_L

#### Conclusion:

```console
Our experiments show that federated learning can be made practical, as FedAvg trains high-quality models using relatively few rounds of communication, as demonstrated by results on a variety of model architectures: a multi-layer perceptron, two different convolutional NNs, a two-layer character LSTM, and a large-scale word-level LSTM.
```

#### Personal Conclusion:

Federated Learning poses an executable and easy to implement solution for privacy enabled learning. With the introduction of `FederatedAveraging` and `FederatedSGD`, implementing FL becomes easier. Much correlations can be made between the parameters:

* K - Clients
* B - MiniBatch Size
* E - Number of Local Epochs
* η - Learning Rate

and the performance of the model. 

* FedAvg

Having lower B improves perfomance in general. With a lower B one can also introduce a higher E, however, tuning the two is extremely important.

* FedSGD

Highly sensitive to learning rate (η).

Other than this, another major outcome is the performance of *_FedAvg being better than FedSGD._*

#### Citations:

```console
@ARTICLE{2016arXiv160205629B,
       author = {Brendan McMahan, H. and Moore, Eider and Ramage, Daniel and Hampson, Seth and Ag{\"u}era y Arcas, Blaise},
        title = "{Communication-Efficient Learning of Deep Networks from Decentralized Data}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning},
         year = 2016,
        month = feb,
          eid = {arXiv:1602.05629},
        pages = {arXiv:1602.05629},
archivePrefix = {arXiv},
       eprint = {1602.05629},
 primaryClass = {cs.LG},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2016arXiv160205629B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```