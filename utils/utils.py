import logging
import time
from ruamel.yaml import YAML

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Prepare_logger(args, eval=False):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(0)
    logger.addHandler(handler)

    date = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    # logfile = args.snapshot_pref+date+'.log' if not eval else args.snapshot_pref + f'/{date}-Eval.log'
    logfile = args.logs_dir+date+'.log' if not eval else args.logs_dir + f'/{date}-Eval.log'

    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# def get_configs(dataset):
#     yaml = YAML(typ='safe')
#     data = yaml.load(open('./configs/dataset_cfg.yaml'))
#     return data[dataset]

# def get_and_save_args(parser):
#     args = parser.parse_args()
#     # dataset = args.dataset
#     yaml = YAML(typ='safe')
#     default_config = yaml.load(open('./configs/default_config.yaml', 'r'), Loader=yaml.RoundTripLoader)
#     current_config = vars(args)
#     for k, v in current_config.items():
#         if k in default_config:
#             if (v != default_config[k]) and (v is not None):
#                 print(f"Updating:  {k}: {default_config[k]} (default) ----> {v}")
#                 default_config[k] = v
#     yaml.dump(default_config, open('./current_configs.yaml', 'w'), indent=4, Dumper=yaml.RoundTripDumper)
#     return default_config
from ruamel.yaml import YAML, RoundTripLoader, RoundTripDumper
def get_configs(dataset: str):
    yaml = YAML(typ='rt')
    try:
        with open('./configs/dataset_cfg.yaml', 'r', encoding='utf-8') as f:
            data = yaml.load(f)
            return data.get(dataset, {})
    except FileNotFoundError:
        print(f"Error: File './configs/dataset_cfg.yaml' not found.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return {}

def get_and_save_args(parser):
    args = parser.parse_args()
    yaml = YAML(typ='rt')
    
    try:
        # Load default config with preserved comments/formatting
        with open('/data1/cy/MTN/cmbs/configs/default_config.yaml', 'r', encoding='utf-8') as f:
            default_config = yaml.load(f)
        
        # Update config with command-line arguments
        current_config = vars(args)
        for k, v in current_config.items():
            if k in default_config and (v is not None and v != default_config[k]):
                print(f"Updating: {k}: {default_config[k]} (default) → {v}")
                default_config[k] = v
        
        # Save updated config while preserving formatting
        with open('/data1/cy/MTN/cmbs/configs/current_configs.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f)
        
        return default_config
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error saving YAML: {e}")
        return {}