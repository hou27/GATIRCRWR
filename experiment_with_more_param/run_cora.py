import time
from datetime import datetime
from torch_geometric.datasets import Planetoid

from run_more_param import train_gat, get_training_args

#
# Cora specific constants
#

# Thomas Kipf et al. first used this split in GCN paper and later Petar Veličković et al. in GAT paper
CORA_TRAIN_RANGE = [0, 140]  # we're using the first 140 nodes as the training nodes
CORA_VAL_RANGE = [140, 140+500]
CORA_TEST_RANGE = [1708, 1708+1000]
CORA_NUM_INPUT_FEATURES = 1433
CORA_NUM_CLASSES = 7

def save_to_file(filename, content):
    with open(filename, 'a') as f:
        f.write(content)

if __name__ == '__main__':
    filename = f'./experiment_result/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_cora_test.txt'
    save_to_file(filename, 'Cora\n')
    dataset = Planetoid(root='../data/Cora', name='Cora')
    init_time_start = time.time()
    for i in range(4):
        cases = [[False, False, "GAT"], [True, False, "GAT with Random walk with restart"], [False, True, "GAT with Initial residual connection"], [True, True, "GAT with Random walk with restart and Initial residual connection"]]
        for case in cases:
            content = f"{i+2} layers {case[-1]}\n"
            save_to_file(filename, content)
            time_start = time.time()
            content = train_gat(get_training_args(time_start, dataset, CORA_TRAIN_RANGE, CORA_VAL_RANGE, CORA_TEST_RANGE, CORA_NUM_INPUT_FEATURES, CORA_NUM_CLASSES, random_walk_with_restart=case[0], add_residual_connection=case[1], num_of_additional_layer=i), save_to_file, filename)
            save_to_file(filename, f'Total training time: {(time.time() - time_start):.2f} [s]\n')

    save_to_file(filename, f'\n\nTotal training time for Full Process: {(time.time() - init_time_start):.2f} [s]\n')