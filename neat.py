from random import uniform, random, choice, shuffle
from bisect import bisect_left
import math




def sigmoid(num):
    return 1 / (1 + math.e**-num)




class NeatNode(object):
    def __init__(self, num):
        self.num = num




class FFNet(object):
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.nodes = NodeHolder()

    def __str__(self):
        ffnet = "FFNet(\n"
        for node in self.nodes:
            ffnet += "  " + str(node) + "\n"
            for con in node.connections:
                ffnet += "    " + str(con) + "\n"
        return ffnet + ")"

    def guess(self, inputs):
        for i in range(self.num_inputs):
            self.nodes[i].activation = inputs[i]
        outputs = []
        for i in range(self.num_inputs, self.num_inputs + self.num_outputs):
            outputs.append(self.nodes[i].get_activation())
        return outputs


class FFNode(NeatNode):
    def __init__(self, num):
        NeatNode.__init__(self, num)
        self.connections = []
        self.activation = 0

    def __str__(self):
        return f"FFNode(num:{self.num}, activation:{self.activation}, num_cons:{len(self.connections)})"

    def get_activation(self):
        if not self.connections:
            return self.activation
        weighted_sum = 0
        for con in self.connections:
            weighted_sum += con.from_node.get_activation() * con.weight
        return sigmoid(weighted_sum)


class FFCon(object):
    def __init__(self, from_node, weight):
        self.from_node = from_node
        self.weight = weight

    def __str__(self):
        return f"FFCon(from_node:{self.from_node.num}, weight:{self.weight})"









class NodeHolder(object):
    def __init__(self):
        self.nodes = []
        self.nums = []

    def __contains__(self, node):
        pos = bisect_left(self.nums, node.num)
        return pos < len(self.nodes) and self.nodes[pos] == node

    def __iter__(self):
        return iter(self.nodes)

    def __getitem__(self, index):
        return self.nodes[index]

    def __len__(self):
        return len(self.nodes)

    def add(self, node):
        pos = bisect_left(self.nums, node.num)
        self.nodes.insert(pos, node)
        self.nums.insert(pos, node.num)

    def get_random(self):
        return choice(self.nodes)

    def find_by_num(self, num):
        pos = bisect_left(self.nums, num)
        if pos >= len(self.nodes) or self.nodes[pos].num != num:
            return None
        return self.nodes[pos]








class ConHolder(object):
    def __init__(self):
        self.connections = []
        self.inovs = []

    def __contains__(self, con):
        pos = bisect_left(self.inovs, con.inov_num)
        return pos < len(self.connections) and self.connections[pos] == con

    def __iter__(self):
        return iter(self.connections)

    def __getitem__(self, index):
        return self.connections[index]

    def __len__(self):
        return len(self.connections)

    def add(self, con):
        pos = bisect_left(self.inovs, con.inov_num)
        self.connections.insert(pos, con)
        self.inovs.insert(pos, con.inov_num)

    def con_between(self, from_node, to_node):
        for con in self.connections:
            if con.from_node == from_node and con.to_node == to_node:
                return True
        return False

    def find(self, con):
        for i in range(len(self.connections)):
            if self.connections[i] == con:
                return i
        return -1

    def get_random(self):
        return choice(self.connections)







class Node(NeatNode):
    def __init__(self, num, x):
        NeatNode.__init__(self, num)
        self.x = x

    def __str__(self):
        return "Node(num:" + str(self.num) + ", x:" + str(self.x) + ")"

    def __eq__(self, other):
        return self.num == other.num

    def copy(self):
        node = Node(self.num, self.x)
        return node





class Connection(object):
    def __init__(self, from_node, to_node, inov_num=-1):
        self.from_node = from_node
        self.to_node = to_node
        self.inov_num = inov_num
        self.weight = uniform(-2, 2)
        self.enabled = True
        self.split_nodes = []

    def __str__(self):
        return f"Connection(from:{self.from_node.num}, to:{self.to_node.num}, weight:{round(self.weight, 4)}, inov_num:{self.inov_num}, enabled:{self.enabled})"

    def __eq__(self, other):
        return self.from_node == other.from_node and self.to_node == other.to_node

    def copy(self):
        con = Connection(self.from_node, self.to_node, self.inov_num)
        con.weight = self.weight
        con.enabled = self.enabled
        con.split_nodes = [n for n in self.split_nodes]
        return con










class Genome(object):
    def __init__(self, neat, num_inputs, num_outputs):
        self.neat = neat
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.nodes = NodeHolder()
        self.connections = ConHolder()
        self.fitness = 0

    def __str__(self):
        genome = "Genome(\n"
        for node in self.nodes:
            genome += "  " + str(node) + "\n"
        for con in self.connections:
            genome += "  " + str(con) + "\n"
        return genome + ")"

    def copy(self):
        newg = Genome(self.neat, self.num_inputs, self.num_outputs)
        for node in self.nodes:
            newg.nodes.nodes.append(node.copy())
            newg.nodes.nums.append(node.num)
        for con in self.connections:
            newg.connections.connections.append(con.copy())
        return newg

    def distance(self, g2):
        g1 = self
        g1p, g2p = 0, 0
        disjoint, excess = 0, 0
        weight_diff = 0
        num_similar = 0
        while g1p < len(g1.connections) and g2p < len(g2.connections):
            g1_con = g1.connections[g1p]
            g2_con = g2.connections[g2p]
            if g1_con.inov_num == g2_con.inov_num:
                num_similar += 1
                weight_diff += abs(g1_con.weight - g2_con.weight)
                g1p += 1
                g2p += 1
            elif g1_con.inov_num < g2_con.inov_num:
                disjoint += 1
                g1p += 1
            else:
                disjoint += 1
                g2p += 1
        if g1p == len(g1.connections):
            excess = len(g2.connections) - g1p
        else: excess = len(g1.connections) - g2p
        if len(g1.connections) > len(g2.connections):
            num_conns = len(g1.connections)
        else: num_conns = len(g2.connections)
        if num_conns < 20: num_conns = 1
        excess_val = (excess * self.neat.config["excess_cons_coeficient"]) / num_conns
        disjoint_val = (disjoint * self.neat.config["disjoint_cons_coeficient"]) / num_conns
        if num_similar == 0: weight_diff_val = 0
        else: weight_diff_val = (weight_diff / num_similar) * self.neat.config["weight_diff_coeficient"]
        return excess_val + disjoint_val + weight_diff_val

    @staticmethod
    def crossover(g1, g2):
        if g2.fitness > g1.fitness:
            tmp = g2
            g2 = g1
            g1 = tmp
        newg = Genome(g1.neat, g1.num_inputs, g1.num_outputs)
        g1p, g2p = 0, 0
        while g1p < len(g1.connections) and g2p < len(g2.connections):
            g1_con = g1.connections[g1p]
            g2_con = g2.connections[g2p]
            if g1_con.inov_num == g2_con.inov_num:
                newcon = g1_con.copy() if random() > 0.5 else g2_con.copy()
                newg.connections.add(newcon)
                g1p += 1
                g2p += 1
            elif g1_con.inov_num < g2_con.inov_num:
                newcon = g1_con.copy()
                newg.connections.add(newcon)
                g1p += 1
            else:
                g2p += 1
        if g2p == len(g2.connections):
            while g1p < len(g1.connections):
                newg.connections.add(g1.connections[g1p].copy())
                g1p += 1
        for i in range(g1.num_inputs + g1.num_outputs):
            newg.nodes.add(g1.nodes[i].copy())
        for con in newg.connections:
            if con.from_node not in newg.nodes:
                newg.nodes.add(con.from_node.copy())
            if con.to_node not in newg.nodes:
                newg.nodes.add(con.to_node.copy())
        return newg

    def mutate(self):
        if random() < self.neat.config["add_con_rate"]:
            self.add_connection()
        if random() < self.neat.config["add_node_rate"]:
            self.add_node()
        if random() < self.neat.config["mutate_weight_rate"]:
            self.mutate_weight()
        if random() < self.neat.config["replace_weight_rate"]:
            self.replace_weight()
        if random() < self.neat.config["toggle_enabled_rate"]:
            self.toggle_connection()

    def get_node_pair(self):
        n1 = self.nodes.get_random()
        n2 = self.nodes.get_random()
        while n1.x == n2.x:
            n2 = self.nodes.get_random()
        if n1.x > n2.x:
            tmp = n1
            n1 = n2
            n2 = tmp
        return (n1, n2)

    def add_node(self):
        if len(self.connections) > 0:
            old_con = self.connections.get_random()
            old_con.enabled = False
            con1, new_node, con2 = self.neat.get_new_node(self, old_con)
            con1.weight = 1
            con2.weight = old_con.weight
            self.nodes.add(new_node)
            self.connections.add(con1)
            self.connections.add(con2)

    def add_connection(self):
        for _ in range(50):
            n1, n2 = self.get_node_pair()
            if not self.connections.con_between(n1, n2):
                # print("ADDING CONNECTION")
                con = self.neat.get_new_connection(n1, n2)
                # print("new con:", con)
                self.connections.add(con)
                break

    def mutate_weight(self):
        if len(self.connections) > 0:
            con = self.connections.get_random()
            con.weight += uniform(-0.3, 0.3)

    def replace_weight(self):
        if len(self.connections) > 0:
            con = self.connections.get_random()
            con.weight = uniform(-2, 2)

    def toggle_connection(self):
        if len(self.connections) > 0:
            con = self.connections.get_random()
            con.enabled = not con.enabled

    def ff_net(self):
        ffnet = FFNet(self.num_inputs, self.num_outputs)
        for node in self.nodes:
            ffnet.nodes.add(FFNode(node.num))
        for ffnode in ffnet.nodes:
            for con in self.connections:
                if con.to_node.num == ffnode.num and con.enabled:
                    from_node = ffnet.nodes.find_by_num(con.from_node.num)
                    ffcon = FFCon(from_node, con.weight)
                    ffnode.connections.append(ffcon)
        return ffnet













class Neat(object):
    def __init__(self, config):
        self.process_config(config)
        self.population = []
        self.nodes = NodeHolder()
        self.connections = ConHolder()
        self.node_count = 0
        self.inov_num_count = 0
        self.species = []
        self.init_population()
        self.speciate()
        self.reset_species()

    def process_config(self, config):
        self.config = {
            "num_inputs": None,
            "num_outputs": None,
            "population_size": 100,
            "distance_threshold": 10,
            "excess_cons_coeficient": 1,
            "disjoint_cons_coeficient": 1,
            "weight_diff_coeficient": 0.2,
            "add_con_rate": 0.3,
            "add_node_rate": 0.08,
            "mutate_weight_rate": 0.8,
            "replace_weight_rate": 0.1,
            "toggle_enabled_rate": 0.05
        }
        def isnumber(s):
            try:
                float(s)
                return True
            except: return False
        with open(config, "r") as file:
            for line in file.readlines():
                split = line.split("=")
                if len(split) > 2:
                    continue
                split = list(map(lambda s: s.strip(), split))
                if split[0] not in self.config:
                    continue
                if split[1].isnumeric():
                    self.config[split[0]] = int(split[1])
                elif isnumber(split[1]):
                    self.config[split[0]] = float(split[1])
        if not isinstance(self.config["num_inputs"], int):
            raise TypeError("Number of inputs must be an integer")
        if not isinstance(self.config["num_outputs"], int):
            raise TypeError("Number of outputs must be an integer")
        if not isinstance(self.config["population_size"], int):
            raise TypeError("Population size must be an integer")

    def next_node_count(self):
        self.node_count += 1
        return self.node_count

    def next_inov_num_count(self):
        self.inov_num_count += 1
        return self.inov_num_count

    def init_population(self):
        for i in range(self.config["num_inputs"]):
            self.nodes.add(Node(self.next_node_count(), 0))
        for i in range(self.config["num_outputs"]):
            self.nodes.add(Node(self.next_node_count(), 1))
        for i in range(self.config["population_size"]):
            genome = Genome(self, self.config["num_inputs"], self.config["num_outputs"])
            for node in self.nodes:
                genome.nodes.add(node.copy())
            self.population.append(genome)
        for genome in self.population:
            genome.mutate()

    def get_new_connection(self, n1, n2):
        con = Connection(n1, n2)
        pos = self.connections.find(con)
        if pos == -1:
            con.inov_num = self.next_inov_num_count()
            self.connections.add(con)
        else:
            con.inov_num = self.connections[pos].inov_num
        return con.copy()

    def get_new_node(self, genome, old_con):
        pos = self.connections.find(old_con)
        shuffled_nums = self.connections[pos].split_nodes.copy()
        shuffle(shuffled_nums)
        for num in shuffled_nums:
            if not genome.nodes.find_by_num(num):
                new_node = self.nodes.find_by_num(num)
                con1, con2 = None, None
                for con in self.connections:
                    if con.from_node == old_con.from_node and con.to_node == new_node:
                        con1 = con
                    if con.from_node == new_node and con.to_node == old_con.to_node:
                        con2 = con
                return (con1.copy(), new_node.copy(), con2.copy())
        new_x = (old_con.from_node.x + old_con.to_node.x) / 2
        new_node = Node(self.next_node_count(), new_x)
        con1 = Connection(old_con.from_node, new_node, inov_num=self.next_inov_num_count())
        con2 = Connection(new_node, old_con.to_node, inov_num=self.next_inov_num_count())
        self.nodes.add(new_node)
        self.connections.add(con1)
        self.connections.add(con2)
        self.connections[pos].split_nodes.append(new_node.num)
        return (con1.copy(), new_node.copy(), con2.copy())

    def speciate(self):
        for genome1 in self.population:
            placed = False
            for species in self.species:
                dist = genome1.distance(species.representative)
                if dist < self.config["distance_threshold"]:
                    species.add(genome1)
                    placed = True
                    break
            if not placed:
                self.species.append(Species(genome1))
        for i in range(len(self.species)-1, -1, -1):
            if len(self.species[i]) == 0:
                self.species.pop(i)

    def next_generation(self):
        self.speciate()
        tot_fitness_sum = 0
        for species in self.species:
            for genome in species:
                genome.fitness /= len(species)
                species.fitness_sum += genome.fitness
            tot_fitness_sum += species.fitness_sum
            species.sort()
        self.population.clear()

        repro_qnts = []
        for species in self.species:
            fit = int(species.fitness_sum // tot_fitness_sum * self.config["population_size"])
            repro_qnts.append(fit)
        ii = 0
        while sum(repro_qnts) < self.config["population_size"]:
            repro_qnts[ii] += 1
            ii += 1
            ii = ii % len(repro_qnts)

        for j in range(len(self.species)):
            species = self.species[j]
            species.repro_qnt = repro_qnts[j]
            if len(species) < 3:
                for i in range(species.repro_qnt):
                    newg = species[0].copy()
                    newg.mutate()
                    self.population.append(newg)
            else:
                for i in range(species.repro_qnt):
                    top_half = species.genomes[:math.ceil(len(species)/2)]
                    g1 = choice(top_half)
                    g2 = choice(top_half)
                    while g2 == g1: g2 = choice(top_half)
                    newg = Genome.crossover(g1, g2)
                    newg.mutate()
                    self.population.append(newg)

        for species in self.species:
            species.set_random_rep()
            species.reset()

    def reset_species(self):
        for species in self.species:
            species.reset()

    def get_ff_nets(self):
        return [genome.ff_net() for genome in self.population]








class Species(object):
    def __init__(self, rep=None):
        self.representative = rep
        self.genomes = [self.representative] if rep else []
        self.fitness_sum = 0
        self.repro_qnt = 0

    def __getitem__(self, index):
        return self.genomes[index]

    def __iter__(self):
        return iter(self.genomes)

    def __len__(self):
        return len(self.genomes)

    def set_random_rep(self):
        self.representative = choice(self.genomes)

    def sort(self):
        self.genomes = sorted(self.genomes, key=lambda g: g.fitness, reverse=True)

    def add(self, genome):
        self.genomes.append(genome)

    def reset(self):
        self.genomes.clear()
        self.fitness_sum = 0
        self.repro_qnt = 0








def main():

    neat = Neat("config.txt")

    genome1 = neat.population[0]
    genome2 = neat.population[1]

    for i in range(10):
        genome1.mutate()
        genome2.mutate()

    child = Genome.crossover(genome1, genome2)

    ffnet = child.ff_net()

    print("Parent 1:", genome1)
    print("Parent 2:", genome2)
    print("Child:", child)
    print("Child:", ffnet)





if __name__ == "__main__":
    main()
