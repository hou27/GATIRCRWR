from run import run

#
# Citeseer specific constants
#

CITESEER_TRAIN_RANGE = [0, 120]
CITESEER_VAL_RANGE = [120, 120+500]
CITESEER_TEST_RANGE = [2308, 2308+1000]
CITESEER_NUM_INPUT_FEATURES = 3703
CITESEER_NUM_CLASSES = 6

#
# Hyperparameters
#

GAMMA = 0.7
BETA = 0.5

if __name__ == '__main__':
    run('Citeseer', CITESEER_TRAIN_RANGE, CITESEER_VAL_RANGE, CITESEER_TEST_RANGE, CITESEER_NUM_INPUT_FEATURES, CITESEER_NUM_CLASSES, GAMMA, BETA)