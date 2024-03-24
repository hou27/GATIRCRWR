import time
from datetime import datetime
from torch_geometric.datasets import Planetoid

from run_more_param import train_gat, get_training_args

#
# Citeseer specific constants
#

CITESEER_TRAIN_RANGE = [0, 120]
CITESEER_VAL_RANGE = [120, 120+500]
CITESEER_TEST_RANGE = [2308, 2308+1000]
CITESEER_NUM_INPUT_FEATURES = 3703
CITESEER_NUM_CLASSES = 6

def save_to_file(filename, content):
    with open(filename, 'a') as f:
        f.write(content)

if __name__ == '__main__':
    filename = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_citeseer_test.txt'
    save_to_file(filename, 'Citeseer\n')
    dataset = Planetoid(root='../data/Citeseer', name='Citeseer')
    init_time_start = time.time()
    for i in range(4):
        cases = [[False, False, "GAT"], [True, False, "GAT with Random walk with restart"], [False, True, "GAT with Initial residual connection"], [True, True, "GAT with Random walk with restart and Initial residual connection"]]
        for case in cases:
            content = f"{i+2} layers {case[-1]}\n"
            save_to_file(filename, content)
            time_start = time.time()
            content = train_gat(get_training_args(time_start, dataset, CITESEER_TRAIN_RANGE, CITESEER_VAL_RANGE, CITESEER_TEST_RANGE, CITESEER_NUM_INPUT_FEATURES, CITESEER_NUM_CLASSES, random_walk_with_restart=case[0], add_residual_connection=case[1], num_of_additional_layer=i), save_to_file, filename)
            save_to_file(filename, f'Total training time: {(time.time() - time_start):.2f} [s]\n')

    save_to_file(filename, f'\n\nTotal training time for Full Process: {(time.time() - init_time_start):.2f} [s]\n')