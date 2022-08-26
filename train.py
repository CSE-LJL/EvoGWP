from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
torch.backends.cudnn.benchmark=True
import argparse
import yaml
from model.pytorch.supervisor import GTSSupervisor
from lib.utils import load_graph_data
from pathos.helpers import mp


def main(args):

    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        save_adj_name = args.config_filename[11:-5]
        # device = torch.device("cuda:0" if torch.cuda.is_available() and not args.use_cpu_only else "cpu")
        # global device
        supervisor = GTSSupervisor(save_adj_name, load_model=args.load_model, device=args.device, **supervisor_config)
        supervisor.train()
        # {0: 198, 1: 197, 2: 159}
        # {500: 196, 1000: 190}
        # supervisor.evaluate(dataset='test')
        # supervisor.obtain_pred_details('prediction_result_dict_ali_1000', save=True)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/model/para_la.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    # parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--load_shapelets', default=False, type=bool, help='Set to load shapelets.')
    parser.add_argument('--load_model', default=True, type=bool, help='Set to load pre-trained model.')
    parser.add_argument('--device', type=str, default='cuda:1', help='')
    args = parser.parse_args()
    main(args)

# python train.py --config_filename=data/model/tencent_trace.yaml --temperature=0.5
# python train.py --config_filename=data/model/tencent_trace.yaml --use_cpu_only True
# python train.py --config_filename=data/model/tencent_trace.yaml --device cuda:0
# python train.py --config_filename=data/model/tencent_trace.yaml --device cuda:0 --load_model True
# python train.py --config_filename=data/model/alibaba_trace.yaml --device cpu
