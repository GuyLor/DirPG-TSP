import numpy as np
import torch
from concorde.tsp import TSPSolver
import collections as C


def solve_batch_graphs(batch, first_cities, tmp_name='tmpname'):
    def create_concorde_file_xy_coord(arr, name="route", template=''):

        n_cities = arr.shape[0]

        assert len(arr.shape) == 2

        # space delimited string
        matrix_s = ""
        for i, row in enumerate(arr.tolist()):
            # print(row)
            matrix_s += "".join("{} {} {}".format(i, row[0], row[1]))
            # print(matrix_s)
            # print('-------------')
            matrix_s += "\n"
        # print(template.format(**{'name': name,
        #                              'n_cities': n_cities,
        #                              'matrix_s': matrix_s}))
        return template.format(**{'name': name,
                                  'n_cities': n_cities,
                                  'matrix_s': matrix_s})

    def write_file(outf, x, template):
        with open(outf, 'w') as dest:
            dest.write(create_concorde_file_xy_coord(x, name="My Route", template=template))  # 100*x.squeeze(0)
        return

    Trajectory = C.namedtuple('Trajectory', 'costs actions')

    actions = torch.zeros((batch.size(0), batch.size(1)))
    costs = []
    M = 100000000
    for i, x in enumerate(batch.cpu().numpy()):
        template = "NAME: {name}\nTYPE: TSP\nCOMMENT: {name}\nDIMENSION: {n_cities}\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n{matrix_s}EOF"
        x = M * x
        x = x.astype(int)
        outf = "/tmp/{}.tsp".format(tmp_name)

        write_file(outf, x, template)

        solver = TSPSolver.from_tspfile(outf)
        solution = solver.solve()
        #print(list(solution.tour))
        if first_cities[i] not in list(solution.tour):
            print('----')
            print(first_cities[i])
            print(list(solution.tour))
        actions[i] = torch.from_numpy(
            np.roll(solution.tour, solution.tour.shape[0] - list(solution.tour).index(first_cities[i])))
        costs.append(solution.optimal_value / M)

    return (Trajectory(costs=costs, actions=actions))
