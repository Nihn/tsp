import networkx as nx
import numpy as np
from argh import dispatching
from matplotlib import pyplot
from progressbar import Bar, Percentage, ProgressBar
from time import sleep, time


def generate_random_matrix(n, max_cost):
    """
    Generate costs matrix
    :param n: array will be n x n shape
    :param max_cost: max cost value
    :return: array of arrays (n x n)
    """
    random = np.random.randint(1, max_cost, (n, n))
    np.fill_diagonal(random, 0)
    return random


def inverse_mutate(generation):
    """
    Inverse mutation
    :param generation: generation to mutate
    :return: mutated generation
    """
    generation = np.copy(generation)
    n = len(generation)

    for breeder in generation:
        l_rand = np.random.randint(n)
        r_rand = np.random.randint(l_rand, n)
        breeder[l_rand:r_rand] = breeder[l_rand:r_rand][::-1]

    return generation


def make_rating_function(n_breeders, callback1, callback2, callback3,
                         cost_matrix, distance_matrix, time_matrix):
    """
    Make function to rate breeders
    :param n_breeders: number of breeders in one generation
    :param callback1: cost function of cost_matrix
    :param callback2: cost function of distance_matrix
    :param callback3: cost function of time_matrix
    :param cost_matrix:
    :param distance_matrix:
    :param time_matrix:
    :return: n_breeders best breeders from one generation,
             average costs per breeder,
             lowest cost in generation
    """
    def rating(generation):

        for breed_index, breed in enumerate(generation):
            previous = 0
            x = y = z = 0
            for index, node in enumerate(breed):
                x += cost_matrix[previous, node + 1]
                y += distance_matrix[previous, node + 1]
                z += time_matrix[previous, node + 1]
                previous = node

            total_costs = callback1(x) + callback2(y) + callback3(z)
            generation[breed_index] = total_costs, breed

        generation = sorted(generation, key=lambda x: x[0])[:n_breeders]

        total_costs, generation = zip(*generation)

        return generation, np.sum(total_costs)/n_breeders, total_costs[0]

    return rating


def pmx_crossbreed(generation):
    """
    PMX crossbreed
    :param generation:
    :return: crossbreeeded generation
    """

    n = len(generation)

    new_generation = []

    for i in range(0, n, 2):
        l_rand = np.random.randint(n)
        r_rand = np.random.randint(l_rand, n)
        common1 = generation[i][l_rand:r_rand]
        common2 = generation[i+1][l_rand:r_rand]

        roll1 = np.roll(generation[i], -r_rand)
        roll2 = np.roll(generation[i+1], -r_rand)

        diff1 = roll2[np.array(map(lambda x: x not in common1, roll2))]
        diff2 = roll1[np.array(map(lambda x: x not in common2, roll1))]

        descendant1 = np.roll(np.concatenate((common1, diff1)), -r_rand-1)
        descendant2 = np.roll(np.concatenate((common2, diff2)), -r_rand-1)

        new_generation.append(descendant1)
        new_generation.append(descendant2)

    return new_generation


def generate_graph(matrix, way):
    """
    Generate and plot graph
    :param matrix: nodes of graph
    :param way: finded way
    :return: None
    """

    graph = nx.from_numpy_matrix(matrix)
    edge_list = []
    previous = 0

    for i, node in enumerate(way):
        edge_list.append((previous, node,
                          {'weight': matrix[previous, node]}))
        previous = node

    edge_list.append((previous, len(way) + 1,
                      {'weight': matrix[previous, len(way) + 1]}))

    nx.draw(graph, pos=nx.spring_layout(graph, k=0.9),
            edgelist=edge_list, with_labels=True)


def evolution(dimension=100, max_cost=10, n_breeders=100, n_generations=100, save=None, load=None):
    """
    Main function, in here generations evolve
    :param dimension: number of cities, costs matrix will be dim x dim
    :param max_cost: max cost in costs matrix will be <1, max_cost>
    :param n_breeders: number of breeders in one generation
    :param n_generations: number of generation before which algorithm halts
    :param save if specified generated arrays will be saved as {cost, distance, time}_matrix.npy
    :return: None
    """

    if load is None:
        cost_matrix = generate_random_matrix(dimension, max_cost)
        distance_matrix = generate_random_matrix(dimension, max_cost)
        time_matrix = generate_random_matrix(dimension, max_cost)
    else:
        print 'Loading saved files...'
        cost_matrix = np.load(load + '_cost.npy')
        distance_matrix = np.load(load + '_distance.npy')
        time_matrix = np.load(load + '_time.npy')

    if save is not None:
        print 'Saving files...'
        np.save(save + '_cost', cost_matrix)
        np.save(save + '_distance', distance_matrix)
        np.save(save + '_time', time_matrix)

    print 'Cost matrix:\n', cost_matrix
    print 'Distance matrix:\n', distance_matrix
    print 'Time matrix:\n', time_matrix

    start = time()

    generation = [(np.random.permutation(dimension - 2) + 1)
                  for _ in range(n_breeders)]

    rating = make_rating_function(n_breeders,
                                  lambda x: x**2,
                                  lambda x: np.sqrt(x),
                                  lambda x: 2*(x+3),
                                  cost_matrix,
                                  distance_matrix,
                                  time_matrix)

    generation, avg_cost, best_cost = rating(generation)
    all = [(generation, avg_cost, best_cost)]

    with ProgressBar(maxval=n_generations, widgets=['Evolution... ', Bar(), Percentage()]) as p:
        for i in range(n_generations):
            number_of_children = np.random.randint(n_breeders)
            new_generation = np.array(generation)[np.random.random_integers(
                0, n_breeders - 1, number_of_children - number_of_children % 2)]
            new_generation = pmx_crossbreed(new_generation)
            new_generation = inverse_mutate(new_generation)

            generation, avg_cost, best_cost = rating(list(generation) +
                                                     list(new_generation))
            all.append((generation, avg_cost, best_cost))
            p.update(i)

    sleep(0.01)

    print 'Finded way:\n 0 -> {0} -> {1}'.format(' -> '.join(map(str, generation[0])), dimension - 1)
    print 'Finall cost: ', best_cost
    print 'Computing time: {0:.2f}s'.format(time() - start)

    pyplot.figure(1)
    pyplot.subplot(212)

    avg_costs, = pyplot.plot(range(len(all)), zip(*all)[1])
    best_costs, = pyplot.plot(range(len(all)), zip(*all)[2])
    pyplot.legend([avg_costs, best_costs], ['Avg costs', 'Lowest costs'])
    pyplot.ylabel('Costs')
    pyplot.xlabel('Generation')

    pyplot.subplot(211)
    generate_graph(cost_matrix, generation[0])
    pyplot.title('Finall cost: %d (%d breeders)' % (best_cost, n_breeders))

    pyplot.show()


def main():
    dispatching.dispatch_command(evolution)


if __name__ == '__main__':
    main()
