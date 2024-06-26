from run import run

#
# Pubmed specific constants
#

PUBMED_TRAIN_RANGE = [0, 60]
PUBMED_VAL_RANGE = [60, 60+500]
PUBMED_TEST_RANGE = [18717, 18717+1000]
PUBMED_NUM_INPUT_FEATURES = 500
PUBMED_NUM_CLASSES = 3

#
# Hyperparameters
#

GAMMA = 0.7
BETA = 0.5

if __name__ == '__main__':
    run('Pubmed', PUBMED_TRAIN_RANGE, PUBMED_VAL_RANGE, PUBMED_TEST_RANGE, PUBMED_NUM_INPUT_FEATURES, PUBMED_NUM_CLASSES, GAMMA, BETA)