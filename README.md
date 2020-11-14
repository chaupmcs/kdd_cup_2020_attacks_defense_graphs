**KDDCup 2020 - Track 2 - Adversarial Attacks and Defense on Academic Graph**

https://www.biendata.xyz/competition/kddcup_2020_formal/

The challenge: Creating an **attacker** (i.e., a modified input consisting of graph structure (adjacency matrix) and node features (embedding vectors)) and a **defender** (i.e., a robust Graph Neural Network model). 
The organizers will match all attackers and defenders from all teams and rank the final leaderboard.

![image](https://raw.githubusercontent.com/chaupmcs/kdd_cup_2020_attacks_defense_graphs/master/intro.png)

**Final result: 7th** (Teamed up with [Vung Pham](https://github.com/phamvanvung))

--------------------


**REFERENCES**

**Some Relevant Papers/Articles**
| Paper (year)  | Category | Link
| ------------- | ------------- |------------- |
|Semi-supervised Classification with Graph Convolutional Networks (GCN) (2017)|graph_neural_network|https://arxiv.org/pdf/1609.02907.pdf|
|Adversarial Examples on Graph Data: Deep Insights into Attack and Defense (2019)|attack,defense|https://arxiv.org/pdf/1903.01610.pdf|
|Type of Attacks in ML|attack|https://towardsdatascience.com/how-to-attack-machine-learning-evasion-poisoning-inference-trojans-backdoors-a7cb5832595c|
|Adversarial Attacks on Neural Networks for Graph Data (Nettack) (2018)|attack|https://arxiv.org/pdf/1805.07984.pdf|
|Attacking Graph-based Classification via Manipulating theGraph Structure (2019)|attack|https://arxiv.org/pdf/1903.00553.pdf|
|Backdoor Attacks to Graph Neural Networks (2020)|attack,defense|https://arxiv.org/pdf/2006.11165.pdf|
|Inductive Representation Learning on Large Graphs (GraphSAGE) (2017)|graph_neural_network|https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf|
|Adversarial Attack on Graph Structured Data (2018)|attack|https://arxiv.org/pdf/1806.02371.pdf|
|Practical Attacks Against Graph-based Clustering (2017)|attack|https://arxiv.org/pdf/1708.09056.pdf|
|Adversarial Attack and Defense on Graph Data: A Survey (2020)|attack,defense|https://arxiv.org/pdf/1812.10528.pdf
|HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (2019)|graph_neural_network|https://arxiv.org/pdf/1810.00826.pdf
|A Comprehensive Survey on Graph Neural Networks (2019)|graph_neural_network|https://arxiv.org/pdf/1901.00596.pdf


**Updates: Some notes on the solution of the winning team - SPEIT (#1): https://www.bilibili.com/s/video/BV1UZ4y1M7uA**
  + **Attacker**: Attack the structure first, keep it unchanged, then attack the features using gradient descent. *Multilayer internal connection* (i.e., Some fake nodes only connect to other fake nodes) is the most efficient strategy for structure attack. As for the loss function, [Carlini-Wagner](https://medium.com/@iambibek/explanation-of-the-carlini-wagner-c-w-attack-algorithm-to-generate-adversarial-examples-6c1db8669fa2) & *Cross Entropy* were used. GCN was employed as the surrogate model to attack.
  + **Defender**: In the first step, the authors looked at the statistics to denoise outlier nodes (assign their features to 0). After that, they used normalization (`Arctan` function) to mitigate the effect of the attack. On top of that, they observed [TAGCN](https://arxiv.org/abs/1710.10370) outperformed both GCN and GraphSAGE in terms of accuracy and robustness. 

**Other Resources**
- https://github.com/yenchenlin/awesome-adversarial-machine-learning
- https://stellargraph.readthedocs.io/en/stable/demos/index.html#find-algorithms-for-a-task
- https://adversarial-ml-tutorial.org/
- CS224W: Machine Learning with Graphs (Stanford - Fall 2019) [Course](http://web.stanford.edu/class/cs224w/) [Video](https://www.youtube.com/playlist?list=PL1OaWjIc3zJ4xhom40qFY5jkZfyO5EDOZ)
