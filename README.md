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
