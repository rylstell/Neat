# Neat
A topology and weight evolving artificial neural network (TWEANN).<br/>
This module is a basic implementation of the genetic algorithm described in [this](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) paper.<br/>
Usage is rather simple:
```python
from neat import Neat
from random import randint

nt = Neat("config.txt")

for _ in range(10):
    for genome in nt.population:
        genome.fitness = randint(0, 100)
    nt.next_generation()

nets = nt.get_ff_nets()

inputs = [0.5, 0.1, 0.8]
guess = nets[0].guess(inputs)
print(guess)
```

\* It works okay. Honestly, I haven't tested it much...
