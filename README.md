# BIRCH clustering algorithm
BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) is an unsupervised data mining algorithm used to perform hierarchical clustering, especially suitable for very large databases.<br>
It clusters, incrementally and dynamically, incoming multi-dimensional metric data points to try to produce the best quality clustering with the available resources (i. e., available memory and time constraints).


BIRCH typically finds a good clustering with a single scan of the data and improve the quality further with a few additional scans. 

#### CF Tree
The efficiency of BIRCH is based on a data structure called CF Tree (Clustering Feature Tree), which represents clusters in a compact and hierarchical way.
The CF Tree is a height-balanced tree structure employed by BIRCH to keep a "summary" of the clusters in memory instead of storing all the data points.
**The tree structure and its construction are the core of the algorithm.**

#### Clustering Features
A Clustering Feature is a triple summarizing the information that we maintain about a cluster: (number of points in the cluster, linear sum of the points, square sum of the points).

---
*Implementation based on [Zhang, T.; Ramakrishnan, R.; Livny, M. (1996). "BIRCH: an efficient data clustering method for very large databases"](https://dl.acm.org/doi/10.1145/235968.233324)*