# Federated-Learning
Privacy preserving learning for Federated Learning.

Exploring Federated Learning and developing a better understanding of such technologies become important as digital data continues to grow. Although powerful, Machine Learning/Deep Learning consume too much data, which may enable threats for data leakage. Consumer alignments, choices, details, etc. can come under threat and therefore Federated Learning is something I wish to explore.

Federated Learning allows for smarter models, lower latency, and less power consumption, all while ensuring privacy. And this approach has another immediate benefit: in addition to providing an update to the shared model, the improved model is readily available for deployment on the edge where it is being trained.

* Centralized federated learning
In the centralized federated learning setting, a central server is used to control and manage the different nodes. Everything from selection of Nodes to Aggregation is done by this central server. May lead to latency in server during communication.

* Decentralized federated learning
In the decentralized federated learning setting, the nodes are able to coordinate themselves to obtain the global model. This set-up allows much more flexibility but comes at the cost of Performance.

Another two aspects of Federated Learning are, IID and Non IID data:

* IID:
The dataset used here is Independent and Identically distributed. My understanding: say, the dataset taken is produced by a generator, then a random sample of the dataset, irrespective of its label is taken and given to a client for training. This would be unbiased and solely random, leading to the independent and identical tag.

* Non-IID:
The dataset here is Non-Independent and Identicallt Distributed. My understanding: say, the dataset is again produced by a generator but here, it is for a classification problem for say 10 classes. We first sort the data according to the label and then proceed to divide them into N shards. The N shards are then randomly sampled and provided to X clients.

## ITERATIVE LEARNING:
Assuming a federated round composed by one iteration of the learning process, the learning procedure can be summarized as follows:[6]

1. Initialization: according to the server inputs, a machine learning model (e.g., linear regression, neural network, boosting) is chosen to be trained on local nodes and initialized. Then, nodes are activated and wait for the central server to give the calculation tasks.
2. Client selection: a fraction of local nodes is selected to start training on local data. The selected nodes acquire the current statistical model while the others wait for the next federated round.
3. Configuration: the central server orders selected nodes to undergo training of the model on their local data in a pre-specified fashion (e.g., for some mini-batch updates of gradient descent).
4. Reporting: each selected node sends its local model to the server for aggregation. The central server aggregates the received models and sends back the model updates to the nodes. It also handles failures for disconnected nodes or lost model updates. The next federated round is started returning to the client selection phase.
5. Termination: once a pre-defined termination criterion is met (e.g., a maximum number of iterations is reached or the model accuracy is greater than a threshold) the central server aggregates the updates and finalizes the global model.

## Non-IID:
The main categories for non-iid data can be summarized as follows:

1. Covariate shift: local nodes may store examples that have different statistical distributions compared to other nodes. An example occurs in natural language processing datasets where people typically write the same digits/letters with different stroke widths or slants.
2. Prior probability shift: local nodes may store labels that have different statistical distributions compared to other nodes. This can happen if datasets are regional and/or demographically partitioned. For example, datasets containing images of animals vary significantly from country to country.
3. Concept shift (same label, different features): local nodes may share the same labels but some of them correspond to different features at different local nodes. For example, images that depict a particular object can vary according to the weather condition in which they were captured.
4. Concept shift (same features, different labels): local nodes may share the same features but some of them correspond to different labels at different local nodes. For example, in natural language processing, the sentiment analysis may yield different sentiments even if the same text is observed.
5. Unbalancedness: the data available at the local nodes may vary significantly in size.

## Federated averaging
Federated averaging (FedAvg) is a allows local nodes to perform more than one batch update on local data and exchanges the updated weights rather than the gradients. Further, averaging tuned weights coming from the same initialization does not necessarily hurt the resulting averaged model's performance.
