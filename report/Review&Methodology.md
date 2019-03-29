## REVIEW

Hornby:2006:AAP:1143997.1144142 assign a attri $age$ to each gene and, which indicate how long the induvial .use a fixed size $age-layer​$ to store the populations

DBLP:journals/corr/abs-1802-01548提出了另一种age模型，

#### METHOD



结合 Hornby:2006:AAP:1143997.1144142 和 DBLP:journals/corr/abs-1802-01548提出的两种基于age的EA，我们提出了一种新的EA模式。在Hornby:2006:AAP:1143997.1144142中$age​$作为gene的属性，保证了优秀的基因的延续性；而DBLP:journals/corr/abs-1802-01548中则是引入了个体老化的思想，为individual加入$age​$属性，淘汰掉达到一定寿命的个体。

我们结合二者的思想，提出年龄和血统EA算法(Aging Bloodlines Evolution). 我们的主要目的是在一定程度上保留优良基因的同时适时淘汰掉达到预定寿命的个体。我们的做法是记录每个个体的$age​$和$lifetime​$，在每代演化中更新所有个体的$age​$，当$age​$达到$lifetime​$时将其从种群中移除。其中，当一个个体被选为亲代生成子代时，对其生成子代的fitness进行评估. 如果他的子代的fitness达到种群中的前$p​$%,which indicate that that parent may have outstanding gene,为使这个gene可以延续，我们为这个个体增加$t​$的寿命。

System Design



```
\end{algorithm}  


	\begin{algorithm}[htb]  
	\caption{ Aging Bloodlines Evolution:}
	\begin{algorithmic}[1]  

	\State $population\gets \phi$
	\State $entireGen\gets \phi$
	\State $generation\gets 0$
	
	\While{$population$ size$<N$}
	\State $newNetwork\gets networkInit()$
	\State $trainNetwork(newNewwork)$
	\State add $newNetwork$ to $popution$ and $entireGen$
	\EndWhile
	

	
	\While {$genetation < G$}
		\State $parent\gets$ select a parent from population using tournament selection 
		\State $child \gets mutate(parent)$
		\State $trainNetwork(child)$
		\State add $child$ to $popution$ and $entireGen$
		\If {accuracy of $child$ is better than $p\%$ individuals in population}
			\State extend the $lifetime$ of $parent$ by $t$
		\EndIf
		
		\ForAll {$individual$ in $population$}
			\State update the $age$ of $individual$
			\State remove current $individual$ if its age reaches its $lifetime$
		\EndFor
		
		
	\EndWhile
	
	
	\\  
	\Return the network model with highest accuracy  
\end{algorithmic}  
\end{algorithm}  

```





```
@inproceedings{Hornby:2006:AAP:1143997.1144142,
 author = {Hornby, Gregory S.},
 title = {ALPS: The Age-layered Population Structure for Reducing the Problem of Premature Convergence},
 booktitle = {Proceedings of the 8th Annual Conference on Genetic and Evolutionary Computation},
 series = {GECCO '06},
 year = {2006},
 isbn = {1-59593-186-4},
 location = {Seattle, Washington, USA},
 pages = {815--822},
 numpages = {8},
 url = {http://doi.acm.org/10.1145/1143997.1144142},
 doi = {10.1145/1143997.1144142},
 acmid = {1144142},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {age, computer-automated design, evolutionary algorithms, open-ended design, premature convergence},
} 
```

```
@article{DBLP:journals/corr/abs-1802-01548,
  author    = {Esteban Real and
               Alok Aggarwal and
               Yanping Huang and
               Quoc V. Le},
  title     = {Regularized Evolution for Image Classifier Architecture Search},
  journal   = {CoRR},
  volume    = {abs/1802.01548},
  year      = {2018},
  url       = {http://arxiv.org/abs/1802.01548},
  archivePrefix = {arXiv},
  eprint    = {1802.01548},
  timestamp = {Mon, 13 Aug 2018 16:48:15 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1802-01548},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```