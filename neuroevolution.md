You've probably heard of gradient-based methods (vanilla Gradient Descent, SGD, Adagrad, etc.). They are the workhorse 
behind the success of modern deep learning. Let us review: they use local information about how the objective function 
is changing at the current parameters (the gradient) to inform an update, usually in the form of some multiple of the 
gradient.

Notice that, in today's workflow, a model's architecture is specified by the human... and we're bad at it! Model 
specification of neural networks is somewhat of an art, because there is such an immense freedom to choose. Doesn't that 
strike you as odd? The term *machine learning* would seem to imply this kind of work would be handled by the machine.


# NEAT
Neuroevolution of Augmenting Topologies (NEAT) is a genetic algorithm for learning network topologies. The "augmenting 
topologies" means that the architecture of the network is also being handled by the algorithm. 

Genetic algorithms, and evolutionary computation in general, is concerned with using evolution as an algorithm. This 
seems a reasonable choice to me - evolution is the only process through which we know intelligent life can form. We
know that, in principle at least, this thing works!

Genetic algorithms maintain a population of candidate models, encoded in a genotype. At each iteration, the genotypes
are decoded into the phenotype (the actual model), and evaluated on the objective function. Now, the algorithm generally
keeps the top performers, and creates the next generation by different methods of reproduction. Of course, mutation is 
important here for introducing innovation.

![How genetic algorithms reach a solution, via [this post](http://blog.otoro.net/2017/10/29/visual-evolution-strategies/)](genetic_evolution.gif)

If you squint hard enough, this looks a little like gradient descent. At each "point", local information is used to 
inform the direction to go. In reality the point is a potentially wide-reaching cluster of candidate solutions, and the
next "point" is the general location of the next generation of candidates.

A neural network is a directed, weighted graph. This lends a convenient encoding in which we list connections between
nodes. Mutations come in as adding hidden nodes or adding connections. 

NEAT comes with a couple other insights:

- Addressing competing conventions: there are multiple ways to structure a neural network to solve some problem. If two 
  neural networks are used to reproduce with crossover, the result will likely not also be a solution. NEAT uses a 
  clever scheme of historical markers to align genomes during crossover.
- Speciation: I will quote the paper here: "In nature, different structures tend to be in different species that compete 
  in different niches. Thus, innovation is implicitly protected within a niche." Innovations that are still developing
  are protected using a speciation mechanism.
- Start with minimality: models are initialized with minimal structure. This ensures that resulting solutions are close
  to as minimal as possible.

## Main Algorithm
1. initialize minimal population of networks
2. until finished do
    1. evaluate fitness of each individual of each species, using compatibility distance
    2. assign a number of offspring to each species, proportional to sum of adjusted fitness of its members
    3. eliminate lowest performing individuals of each species
    4. reproduce within each species, and replace the previous individuals
    
### Compatibility Distance
Compute $\delta = \frac{c_1 E}{N} + \frac{c_2 D}{N} + c_3 \bar{W}$, where E is number of excess genes, D is number of 
disjoint genes, and $\bar{W}$ is the average weight differences of matching genes, including disabled genes. $c_1$, 
$c_2$, and $c_3$ are hyperparameters controlling performance of respective values, and N normalizes for genome size.

### Adjusted Fitness
Compute $f_i' = \frac{f_i}{\sum_{j=1}^{n}{sh(\delta(i,j))}}$, where $f_i$ is the unadjusted fitness, the sharing function 
sh is 0 when distance $\delta(i,j)$ is above threshold $\delta_t$, and 1 otherwise. 
    
## Reproduction
1. align parent 1 and parent 2 by historical markers
2. decide genes of descendant
    1. inherit matching genes randomly
    2. inherit disjoint and excess genes from more fit parent
3. mutate individual
    1. generate either an add connection or add node innovation
    2. if adding a connection, pick two nodes to put a new connection between
    3. if adding a node
        1. pick two nodes that already have a connection
        2. disable the connection
        3. add a connection gene from the first node to the new node, with weight 1
        4. add a connection gene from the new node to the second node, with the weight of the now disabled connection
    4. assign each innovation the global innovation number, then increment the global innovation number
    
    
# HyperNEAT
When evolving neural networks, there are two types of encoding: direct, in which the network is explicitly represented
by the genotype, and indirect, in which the network is implicitly contained in the genotype, and is generated using 
rules.

HyperNEAT builds on NEAT with a clever indirect encoding of the ANN. Traditional neural networks treat the input
and output as independent sets of parameters to be related. However, this is throwing out valuable information. The 
authors point to evidence that the body is laid out with certain "geometric motifs": symmetry, periodicity, symmetry 
with variation, etc. 

## Compositional Pattern Producing Networks (CPPNs)
A CPPN is a network that takes in two coordinates in $n$-dimensional space and produces an intensity. **This intensity 
represents the weight on a connection between the nodes placed at those two points**, if those nodes were to be included 
in the final ANN. 

The CPPN is composed of nodes with activations that abstract the motifs found in nature. For example, a gaussian 
function abstracts symmetry, and sine abstracts periodicity. Notice that a CPPN looks a lot like a neural network.

## Producing the final ANN
In HyperNEAT, the CPPN is genotype of the individual. This is then translated into the phenotype, an ANN *orders of 
magnitude* larger. In order to construct this, a grid of coordinates is chosen beforehand, called the substrate. These 
coordinates designate locations where nodes can be placed. This selection is fundamental to the types of solutions 
found. 

## Scalability of HyperNEAT
A big advantage of HyperNEAT is its ability to scale to different resolutions. For example, the authors test the 
algorithm on a visual discrimination task, and it performs as well when the dimension of the input field is increased 
from $11\times 11$ to $55\times 55$, without any additional evolution.

# ES-HyperNEAT
HyperNEAT introduces a very powerful encoding of neural networks, but it also introduces an enormous decision for the 
human... how do we select the substrate? 

Evolvable-Substrate-HyperNEAT (ES-HyperNEAT) has its foundation in a clever insight: "a  representation that encodes the 
pattern of connectivity across a network automatically contains implicit information on where the nodes should be 
placed". Specifically, for regions in the connectivity pattern encoded by the CPPN, we can determine the density that
we should place nodes there - there is a density at which increasing further will not improve results further. If we
can determine these densities, then we know where to put the nodes - the substrate does not need to be selected. 

The solution that the authors chose to solve this does not strike me as the fundamental way to realize this idea. It
works, but it seems a bit ad-hoc. I summarize it below for completeness.

## Algorithm
1. Division Phase
    1. Create hypertree  by recursively dividing hypercube until desired resolution r is reached.
    2. Query CPPN for each leaf for weight $w$.
    3. Determine variance for all tree nodes.
    4. Repeat division on leaves with parent variance $\sigma_p^2 > d_t$ (and maximum resolution r_t not reached).
2. Pruning Phase
    1. Remove all nodes with parent variance $\sigma_p^2 < \sigma_t^2$.
    2. For all leaf nodes: If $|w| > w_t$ and band level $b > b_t$ create connection with a weight proportional to $w$ 
    and corresponding hidden neurons if not existent.
3. Integration Phase
    1. Query connections to outputs from hidden nodes without any outgoing connections.
    2. Query connections from inputs to output and hidden nodes without any incoming connections.
    3. For i. and ii.: If $|w| > w_t$ create connection with weight proportional to $w$.


# Application in Reinforcement Learning
Reinforcement Learning is concerned with creating good "agents": models that act in an environment, receiving feedback
from it. This is a *hard* problem. The feedback can be sporadic. Consider a chess player, whose feedback comes in the 
form of a win or a loss after dozens of moves. Which moves were most responsible for the win or loss? In order to use 
gradient methods, we need to figure this out. The gradients can be extremely small, which leads to slow learning that
may never find a good solution.

Now, consider using genetic algorithms to learn. This is the problem of life on earth - we're essentially a really
tough reinforcement learning scenario. It seems to be a considerably more natural approach to learning agents. 
