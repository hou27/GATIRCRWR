import time
from datetime import datetime
from torch_geometric.datasets import Planetoid

from run import train_gat, get_training_args

#
# Pubmed specific constants
#

PUBMED_TRAIN_RANGE = [0, 60]
PUBMED_VAL_RANGE = [60, 60+500]
PUBMED_TEST_RANGE = [18717, 18717+1000]
PUBMED_NUM_INPUT_FEATURES = 500
PUBMED_NUM_CLASSES = 3

def save_to_file(filename, content):
    with open(filename, 'a') as f:
        f.write(content)

if __name__ == '__main__':
    filename = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_pubmed_test.txt'
    save_to_file(filename, 'Pubmed\n')
    dataset = Planetoid(root='../data/Pubmed', name='Pubmed')
    init_time_start = time.time()
    for i in range(4):
        cases = [[False, False, "GAT"], [True, False, "GAT with Random walk with restart"], [False, True, "GAT with Initial residual connection"], [True, True, "GAT with Random walk with restart and Initial residual connection"]]
        for case in cases:
            content = f"{i+2} layers {case[-1]}\n"
            save_to_file(filename, content)
            time_start = time.time()
            content = train_gat(get_training_args(time_start, dataset, PUBMED_TRAIN_RANGE, PUBMED_VAL_RANGE, PUBMED_TEST_RANGE, PUBMED_NUM_INPUT_FEATURES, PUBMED_NUM_CLASSES, random_walk_with_restart=case[0], add_residual_connection=case[1], num_of_additional_layer=i), save_to_file, filename)
            save_to_file(filename, f'Total training time: {(time.time() - time_start):.2f} [s]\n')

    save_to_file(filename, f'\n\nTotal training time for Full Process: {(time.time() - init_time_start):.2f} [s]\n')