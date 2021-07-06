from src.processing.load_dataframe import load_dataframe
from src.search.mcts.mcts import MCTSTree
import time
import wandb

if __name__ == '__main__':

    hyperparameters_default = {
        'exploration': 5.
    }

    wandb.init(project='searchalgosportsteam', entity='searchalgosports',
        config=hyperparameters_default)
    config = wandb.config

    empty_positions = ['PG','SG','SF','PF','C','F','G','UTIL']
    dataframe = load_dataframe('2021_03_12_slate.csv', empty_positions)
    budget = 50000
    mcts_tree = MCTSTree(dataframe, empty_positions, budget, exploration=config.exploration)

    start = time.time()
    best_node = mcts_tree.run(5000)
    print(f'Best config found in {time.time()-start:.1f}s: {best_node} with value {best_node.value}')

