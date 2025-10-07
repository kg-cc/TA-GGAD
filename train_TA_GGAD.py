import argparse
import time
from pathlib import Path

from utils import *
import warnings
from train_test import TA_GGAD_Detector
import numpy as np

import logging
from datetime import datetime


def setup_logging():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logging.info("Logging is set up.")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train(args):
    setup_logging()

    datasets_train = ['pubmed', 'Flickr', 'questions', 'YelpChi']

    datasets_test = ['ACM', 'Facebook', 'Amazon', 'cora', 'citeseer', 'BlogCatalog', 'Reddit', 'weibo', 'cs', 'photo',
                     'elliptic', 'tfinance']

    # datasets_test = ['DGgraph-fin']
    model = args.model
    print('Training on {} datasets:'.format(len(datasets_train)), datasets_train)
    print('Test on {} datasets:'.format(len(datasets_test)), datasets_test)

    train_config = {
        'device': 'cuda:0',
        'epochs': 40,
        'testdsets': datasets_test,
    }

    model_config = read_json(model, args.shot, args.json_dir)
    dataset_config = read_data_config(args.json_dir)
    args.dataset_config = dataset_config

    if model_config is None:
        model_config = {
            "model": "TA-GGAD",
            "lr": 1e-5,
            "drop_rate": 0.3,
            "h_feats": 1024,
            "num_prompt": 10,
            "num_hops": 2,
            "weight_decay": 5e-5,
            "in_feats": 64,
            "num_layers": 4,
            "activation": "ELU"
        }
        print('use default model config')
    else:
        print('use saved best model config')
        print(model_config)

    dims = 64
    data_train = [Dataset(args, dims, "train", name, './dataset/') for name in datasets_train]
    data_test = [Dataset(args, dims, "test", name, './dataset/') for name in datasets_test]  # CPU

    for tr_data in data_train:
        tr_data.propagated(model_config['num_hops'])
    for te_data in data_test:
        te_data.propagated(model_config['num_hops'])

    model_config['model'] = model
    model_config['in_feats'] = dims

    # Initialize dictionaries to store scores for each test dataset

    auc_dict = {}
    pre_dict = {}

    auc_dict_MOE = {}
    pre_dict_MOE = {}
    seeds = [0, 1, 2, 4, 5]
    for t in seeds:

        seed = t
        set_seed(seed)
        local_time = time.localtime()
        args.lct = time.strftime("%Y_%m_%d_%H_%M_%S", local_time)
        output_dir = Path.cwd().joinpath(
            "output",
            f"seed_{seed}"
        )

        print("Model {}, Trial {}".format(model, seed))
        train_config['seed'] = seed
        for te_data in data_test:
            te_data.random_sample(args.shot)
        for tr_data in data_train:
            tr_data.train_random_sample(args.shot)

        data = {'train': data_train, 'test': data_test}
        detector = TA_GGAD_Detector(train_config, model_config, data, args)
        try:
            if detector.model_exists(seed):
                logging.info(f"📌 Model for seed {seed} exists. Attempting to load...")
                detector.load_model(seed)
                logging.info("✅ Model loaded successfully")
            else:
                logging.info(f"🛠️ No model found for seed {seed}. Starting training...")
                detector.train(args)
                detector.save_model(seed)
        except Exception as e:
            logging.error(f"❌ Error loading model: {str(e)}")
            logging.info("🔄 Reinitializing and training new model...")
            detector = TA_GGAD_Detector(train_config, model_config, data, args)
            detector.train(args)
            detector.save_model(seed)

        test_score_list, test_score_list_MOE = detector.test(args)

        # Aggregate scores for each test dataset
        for test_data_name, test_score in test_score_list.items():
            if test_data_name not in auc_dict:
                auc_dict[test_data_name] = []
                pre_dict[test_data_name] = []
            auc_dict[test_data_name].append(test_score['AUROC'])
            pre_dict[test_data_name].append(test_score['AUPRC'])
            print(f'Test on {test_data_name}, AUC is {auc_dict[test_data_name]}')

        print("-" * 100 + '\n\n\n')
        # Aggregate scores for each test dataset
        for test_data_name, test_score in test_score_list_MOE.items():
            if test_data_name not in auc_dict_MOE:
                auc_dict_MOE[test_data_name] = []
                pre_dict_MOE[test_data_name] = []
            auc_dict_MOE[test_data_name].append(test_score['AUROC'])
            pre_dict_MOE[test_data_name].append(test_score['AUPRC'])
            print(f'Test on {test_data_name}, AUC is {auc_dict_MOE[test_data_name]}')

    auc_mean_dict_moe, auc_std_dict_moe, pre_mean_dict_moe, pre_std_dict_moe = {}, {}, {}, {}

    for test_data_name in auc_dict_MOE:
        auc_mean_dict_moe[test_data_name] = np.mean(auc_dict_MOE[test_data_name])
        auc_std_dict_moe[test_data_name] = np.std(auc_dict_MOE[test_data_name])
        pre_mean_dict_moe[test_data_name] = np.mean(pre_dict_MOE[test_data_name])
        pre_std_dict_moe[test_data_name] = np.std(pre_dict_MOE[test_data_name])

    print('-' * 50 + '-' * 50 + '\n\n\n')
    for test_data_name in auc_mean_dict_moe:
        str_result = 'AUROC:{:.4f}+-{:.4f}, AUPRC:{:.4f}+-{:.4f}'.format(
            auc_mean_dict_moe[test_data_name],
            auc_std_dict_moe[test_data_name],
            pre_mean_dict_moe[test_data_name],
            pre_std_dict_moe[test_data_name])
        print('-' * 50 + test_data_name + '-' * 50)
        print('str_result___MOE', str_result)

        with open(output_dir.parent.joinpath(f"{test_data_name}_exp_results_MOE"), "a+", encoding='utf-8') as f:
            f.write(f'time:{args.lct}\n')
            f.write(f'args:{args}\n')
            f.write('-' * 50 + test_data_name + '-' * 50)
            f.write(f'str_result____MOE:{str_result} \n\n\n')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    # parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--model', type=str, default='ARC')
    parser.add_argument('--shot', type=int, default=10)
    parser.add_argument('--lamda', type=float, default=0.1)
    parser.add_argument('--cut_rate', type=float, default=0.00)
    parser.add_argument('--json_dir', type=str, default='./params')
    parser.add_argument('--delete_edge_rate', type=float, default=0.0)
    parser.add_argument('--code_size', type=int, default=2048)
    parser.add_argument('--topk', type=int, default=15)
    parser.add_argument('--count_node', type=int, default=1)
    parser.add_argument('--normal_ratio', type=int, default=10)
    parser.add_argument('--count_num', type=int, default=2)
    args = parser.parse_known_args()[0]
    train(args)
