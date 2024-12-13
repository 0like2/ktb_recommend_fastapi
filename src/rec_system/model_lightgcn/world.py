import os
from os.path import join
import torch
import multiprocessing
import sys

config = None
device = None
CORES = None

def initialize_world():

    global config, device, CORES
    if config is not None:
        return

    from src.rec_system.model_lightgcn.parse import parse_args

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # Filter arguments for parse_args
    filtered_args = [arg for arg in sys.argv if arg.startswith("--bpr_batch") or arg.startswith("--latent_dim")]
    sys.argv = [sys.argv[0]] + filtered_args

    args = parse_args()

    # Define paths
    ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
    INPUT_PATH = join(ROOT_PATH, 'input')
    OUTPUT_PATH = join(ROOT_PATH, 'output')
    BOARD_PATH = join(OUTPUT_PATH, 'runs')
    FILE_PATH = join(OUTPUT_PATH, 'checkpoints')

    # Ensure necessary directories exist
    if not os.path.exists(FILE_PATH):
        os.makedirs(FILE_PATH, exist_ok=True)

    # Initialize config
    config = {
        'bpr_batch_size': args.bpr_batch,
        'latent_dim': args.latent_dim,
        'n_layers': args.n_layers,
        'dropout': args.dropout,
        'keep_prob': args.keepprob,
        'A_n_fold': args.a_fold,
        'test_u_batch_size': args.testbatch,
        'multicore': args.multicore,
        'lr': args.lr,
        'decay': args.decay,
        'pretrain': args.pretrain,
        'A_split': False,
        'bigdata': False,
        'creator_file': os.path.join(INPUT_PATH, 'Creator_random25.csv'),
        'item_file': os.path.join(INPUT_PATH, 'Item_random25.csv'),
        'similarity_matrix_file': os.path.join(INPUT_PATH, 'similarity_matrix.csv'),
        'dataset': args.dataset,
        'model': args.model,
        'train_epochs': args.epochs,
        'load': args.load,
        'path': args.path,
        'topks': eval(args.topks),
        'tensorboard': args.tensorboard,
        'comment': args.comment,
        'seed': args.seed,
    }

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CORES = multiprocessing.cpu_count() // 2  # Use half the available cores

    # Dataset and model validation
    all_datasets = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book', 'custom-similarity']
    all_models = ['mf', 'lgn']

    dataset = config['dataset']
    model_name = config['model']

    if dataset not in all_datasets:
        raise ValueError(f"Dataset '{dataset}' not supported! Available datasets: {all_datasets}")
    if model_name not in all_models:
        raise ValueError(f"Model '{model_name}' not supported! Available models: {all_models}")


# Utility function for colored print
def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")
