import json
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from lib import utils
from model.pytorch.model_v2 import GTSModel_v3 as GTSModel
from model.pytorch.loss import masked_mae_loss, masked_mape_loss, masked_rmse_loss, masked_mse_loss
import pandas as pd
import os
import time
from model.time2graph.core import time_aware_shapelets, shapelet_utils
from ..config import *

from scipy.special import softmax
from sklearn.preprocessing import minmax_scale

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class GTSSupervisor:
    def __init__(self, save_adj_name, load_model, device, **kwargs):
        # self.use_cpu_only = use_cpu_only
        self.device = torch.device(device)
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self.graph_mode = kwargs.get('graph_mode')
        self.opt = self._train_kwargs.get('optimizer')
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)
        self.ANNEAL_RATE = 0.00003
        self.temp_min = 0.1
        self.save_adj_name = save_adj_name
        self.num_sample = self._train_kwargs.get('num_sample')
        # self.topk = 10

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        self._writer = SummaryWriter('runs/' + self._log_dir)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        self.graph_mode = 'shapelet'
        self.forward_index = 0

        ### Feas
        if self._data_kwargs['dataset_dir'] == 'data/METR-LA':
            df = pd.read_hdf('./data/metr-la.h5')
        elif self._data_kwargs['dataset_dir'] == 'data/PEMS-BAY':
            df = pd.read_hdf('./data/pems-bay.h5')
        elif self._data_kwargs['dataset_dir'] == 'data/Alibaba-Trace_500':
            df = pd.read_csv(self._data_kwargs['dataset_dir'] + '/alibaba_trace_t2g_cpu_usage_500.csv', header=None)
            print('alibaba df loaded', df.shape)
        elif self._data_kwargs['dataset_dir'] == 'data/Alibaba-Trace_1000':
            df = pd.read_csv(self._data_kwargs['dataset_dir'] + '/alibaba_trace_t2g_cpu_usage_1000.csv', header=None)
            print('alibaba df loaded', df.shape)
        elif self._data_kwargs['dataset_dir'] == 'data/Alibaba-Trace_2000':
            df = pd.read_csv(self._data_kwargs['dataset_dir'] + '/alibaba_trace_t2g_cpu_usage_2000.csv', header=None)
            print('alibaba df loaded', df.shape)
        elif self._data_kwargs['dataset_dir'] == 'data/Tencent-Trace_500':
            # 需要知道time2graph需要什么样的df
            df = pd.read_csv(self._data_kwargs['dataset_dir'] + '/tencent_trace_t2g_cpu_usage_500.csv', header=None)
            # df.drop([len(df) - 1], inplace=True)
            df_GTS = pd.read_csv(self._data_kwargs['dataset_dir'] + '/tencent_trace_cpu_usage_500.csv', index_col=['timestamp'])
            # df_GTS.drop(columns=df_GTS.columns.values[-1], inplace=True)
        elif self._data_kwargs['dataset_dir'] == 'data/Tencent-Trace_3960':
            # 需要知道time2graph需要什么样的df
            df = pd.read_csv(self._data_kwargs['dataset_dir'] + '/tencent_trace_t2g_cpu_usage_3960.csv', header=None)
            # df.drop([len(df) - 1], inplace=True)
            # df_GTS = pd.read_csv(self._data_kwargs['dataset_dir'] + '/tencent_trace_cpu_usage_3960.csv', index_col=['timestamp'])
            # df_GTS.drop(columns=df_GTS.columns.values[-1], inplace=True)
        elif self._data_kwargs['dataset_dir'][:23] == 'data/Tencent-Trace_3960':
            # 需要知道time2graph需要什么样的df
            df = pd.read_csv(self._data_kwargs['dataset_dir'] + '/tencent_trace_t2g_cpu_usage_3960_period%d.csv' % (self._data_kwargs['period_index'] + 1), header=None)

        self.num_nodes = int(self._model_kwargs['num_nodes'])  # num_shapelets?
        self.num_shapelets = int(self._model_kwargs['num_shapelets'])
        self.input_dim = int(self._model_kwargs['input_dim'])
        self.seq_len = int(self._model_kwargs['seq_len'])  # for the encoder
        self.output_dim = int(self._model_kwargs['output_dim'])
        self.use_curriculum_learning = bool(self._model_kwargs['use_curriculum_learning'])
        self.horizon = int(self._model_kwargs['horizon'])  # for the decoder

        if self.graph_mode == 'shapelet':
            shapelets_feat, g, sdist, data_feat_d = self.shapelet_graph(df, load=self._train_kwargs['load'])
            # scaler = utils.StandardScaler(mean=train_feas.mean(), std=train_feas.std())
            # train_feas = scaler.transform(train_feas)  # 这个要么就直接大胆一点，就是shapelet
            self.shapelets_feat = torch.Tensor(shapelets_feat).to(self.device)
            self.tmat = g
            # self.adj_mx = torch.Tensor(g).to(self.device)
            self.sdist = sdist
            self.data_feat_d = data_feat_d
        else:
            data_feat_d = None
            sdist = None
            g = None
            shapelets_feat = pd.read_csv(self._data_kwargs['dataset_dir'] + '/train_feas_%d_%d_%d.csv' % (self.num_nodes, self.num_shapelets * 10, self.num_shapelets), header=None)
            self.shapelets_feat = torch.Tensor(shapelets_feat.values).to(self.device)
            self.adj_mx = None

        # data set
        self._data = utils.load_dataset(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']

        # setup model
        para = {'n_shapelets': self.num_shapelets, 'num_nodes': self.num_nodes, 'Evo_layers': 1, 'node_dim': 128, 'graph_dim': 256, 'embedding_dim': 100, 'dim_fc': 128}
        # GTS_model = GTSModel(data_feat_d, sdist, g, self._logger, decoder_evo=True, device=device, **self._model_kwargs)
        GTS_model = GTSModel(para, sdist, g, self._logger, device=device, **self._model_kwargs)
        self.GTS_model = GTS_model.to(self.device)
        self._logger.info("Model created")
        self._logger.info('used dataset %s' % self._data_kwargs['dataset_dir'])
        self._logger.info(json.dumps(self.GTS_model.parameter_dict))
        self._logger.info('encoder num_rnn_layers %d' % GTS_model.encoder_model.num_rnn_layers)
        self._logger.info('encoder evo_lstm_layers %d' % GTS_model.encoder_model.num_Evo_layers)
        self._logger.info('decoder num_rnn_layers %d' % GTS_model.decoder_model.num_rnn_layers)

        # self.evolving_segment_cnt = 0
        self.evolving_unit = self._train_kwargs['evolving_unit']

        self._epoch_num = self._train_kwargs['epoch']

        if self._epoch_num > 0 and load_model:
            self.load_model()
        self._epoch_num = 0

    def save_shapelets(self, fpath):
        torch.save(self.shapelets, fpath)

    def shapelet_graph(self, df, load, batch_size=100, data_size=1, gpu_enable=False):
        gpu_enable = gpu_enable and torch.cuda.is_available()
        # para_dict = {0: (61, 118), 1: (60, 120), 2: (60, 112)}
        # if self._data_kwargs['period_index'] == 0:
        #     seg_length, num_segment = para_dict[self._data_kwargs['period_index']]
        # elif 0 < self._data_kwargs['period_index'] < 5:
        #     seg_length, num_segment = para_dict[1]
        # elif self._data_kwargs['period_index'] == 5:
        #     seg_length, num_segment = para_dict[2]
        # df预处理，注意去掉训练集和验证集部分
        num_samples = df.shape[1]
        if 'Tencent' in self._data_kwargs['dataset_dir']:
            seg_length = 60  # tencent 60, alibaba 120
        elif 'Alibaba' in self._data_kwargs['dataset_dir']:
            seg_length = 120
        num_segment = (num_samples - 1) // seg_length
        print(df.shape)
        print('num_segment', num_segment, 'seg_length', seg_length)
        # segment_length_dict = {0: 14398, 1: 14400, 2: num_samples - 28798}
        # 14398 = 2 * 23 * 313
        # 14400 = 2 * 2 * 2 * 2 * 2 * 2 * 3 * 3 * 5 * 5
        # 13921 = 13921
        # 13920 = 2 * 2 * 2 * 2 * 2 * 3 * 5 * 29
        # 7198 = 2 * 59 * 61
        # 7200 = 2 * 2 * 2 * 2 * 2 * 3 * 3 * 5 * 5
        # 7200 = 2 * 2 * 2 * 2 * 2 * 3 * 3 * 5 * 5
        # 7200 = 2 * 2 * 2 * 2 * 2 * 3 * 3 * 5 * 5
        # 7200 = 2 * 2 * 2 * 2 * 2 * 3 * 3 * 5 * 5
        # 6721 = 11 * 13 * 47
        # 14386
        # 14388
        # 13909

        num_train = num_samples  # 这玩意好像还要保证等于num_segment*seg_length
        if self._data_kwargs['period_index'] == 5:
            x, y = df.values[:, 1:num_segment * seg_length + 1].astype(np.float).reshape(-1, seg_length * num_segment, 1), \
                   df[0].values.astype(np.int)
        else:
            x, y = df.values[:, 1:num_segment * seg_length + 1].astype(np.float).reshape(-1, seg_length * num_segment, 1), \
                   df[0].values.astype(np.int)
        print('line 156', x.shape, y.shape)
        print(np.min(x), np.min(y))
        assert not np.isnan(np.min(x))
        lbs = np.unique(y)
        y_return = np.copy(y)
        for idx, val in enumerate(lbs):
            y_return[y == val] = idx
        y = y_return
        # assert x.shape[1] == num_segment * seg_length
        from os import cpu_count
        njobs = cpu_count()
        if njobs >= 40:
            njobs = int(njobs / 2)
        kwargs = {
            'opt_metric': 'f1',
            'init': 0,
            'warp': 2,
            'tflag': True,
            'mode': 'embedding',
            'candidate_method': 'greedy',
            'lr': 1e-3,
            'p': 2,
            'alpha': 0.1,
            'beta': 0.05,
            'debug': False,
            'batch_size': 200,
            'measurement': 'gdtw',
            'max_iters': 1,
            'optimizer': 'Adam',
            'njobs': njobs,
            'percentile': 80,
            'K': self.num_shapelets,
            'C': self.num_shapelets * 10
        }
        if load:
            self.shapelets = torch.load(self._data_kwargs['dataset_dir'] + '/' + '_shapelets_%d_%d_%d.pth' % (self.num_nodes, kwargs['C'], kwargs['K']), map_location=self.device)
            try:
                shapelets_feas = pd.read_csv(self._data_kwargs['dataset_dir'] + '/shapelets_feas_%d_%d_%d.csv' % (self.num_nodes, kwargs['C'], kwargs['K']), header=None)
            except FileNotFoundError:
                shapelets_feas = pd.read_csv(self._data_kwargs['dataset_dir'] + '/train_feas_%d_%d_%d.csv' % (self.num_nodes, kwargs['C'], kwargs['K']), header=None)
            t_matrix = np.load(self._data_kwargs['dataset_dir'] + '/' + 'evolved_tmat_%d_%d_%d.npy' % (self.num_nodes, kwargs['C'], kwargs['K']))
            sdist = np.load(self._data_kwargs['dataset_dir'] + '/' + 'shapelets_sdist_%d_%d_%d.npy' % (self.num_nodes, kwargs['C'], kwargs['K']))
            dist_threshold = np.load(self._data_kwargs['dataset_dir'] + '/' + 'threshold_%d_%d_%d.npy' % (self.num_nodes, kwargs['C'], kwargs['K']))
        else:
            self.shapelets = time_aware_shapelets.learn_time_aware_shapelets(
                time_series_set=x, label=y, K=kwargs.pop('K'), C=kwargs.pop('C'), p=kwargs.pop('p'),
                num_segment=num_segment, seg_length=seg_length, data_size=data_size,
                lr=kwargs.pop('lr'), alpha=kwargs.pop('alpha'), beta=kwargs.pop('beta'), num_batch=int(x.shape[0] / batch_size),
                measurement=kwargs.pop('measurement'), gpu_enable=gpu_enable, **kwargs)  # 建议把参数都加进来

            kwargs = {
                'opt_metric': 'f1',
                'init': 0,
                'warp': 2,
                'tflag': True,
                'mode': 'embedding',
                'candidate_method': 'greedy',
                'lr': 1e-3,
                'p': 2,
                'alpha': 0.1,
                'beta': 0.05,
                'debug': True,
                'batch_size': 200,
                'measurement': 'gdtw',
                'max_iters': 1,
                'optimizer': 'Adam',
                'njobs': njobs,
                'percentile': 80,
                'K': self.num_shapelets,
                'C': self.num_shapelets * 10
            }
            self.save_shapelets(self._data_kwargs['dataset_dir'] + '/' + '_shapelets_%d_%d_%d.pth' % (self.num_nodes, kwargs['C'], kwargs['K']))
            raw_shapelets_arr = np.zeros((seg_length, kwargs['K']))
            for i in range(len(self.shapelets)):
                raw_shapelets_arr[:, i] = self.shapelets[i][0].reshape(-1)
            shapelets_feas = pd.DataFrame(raw_shapelets_arr)
            shapelets_feas.to_csv(self._data_kwargs['dataset_dir'] + '/shapelets_feas_%d_%d_%d.csv' % (self.num_nodes, kwargs['C'], kwargs['K']), index=False, header=False)

            threshold = None

            t_matrix, sdist, dist_threshold = shapelet_utils.transition_matrix(time_series_set=x, shapelets=self.shapelets, seg_length=seg_length,
                                                                               tflag=kwargs.pop('tflag'), multi_graph=True, percentile=kwargs['percentile'], threshold=threshold, tanh=kwargs.get('tanh', False), debug=kwargs['debug'],
                                                                               init=kwargs['init'], warp=kwargs['warp'], measurement=kwargs['measurement'])

            np.save(self._data_kwargs['dataset_dir'] + '/' + 'evolved_tmat_%d_%d_%d.npy' % (self.num_nodes, kwargs['C'], kwargs['K']), t_matrix)
            np.save(self._data_kwargs['dataset_dir'] + '/' + 'shapelets_sdist_%d_%d_%d.npy' % (self.num_nodes, kwargs['C'], kwargs['K']), sdist)
            np.save(self._data_kwargs['dataset_dir'] + '/' + 'threshold_%d_%d_%d.npy' % (self.num_nodes, kwargs['C'], kwargs['K']), np.array([dist_threshold]))
        print('dist_threshold', dist_threshold)

        data_feat = df.values[:, 1:]
        data_feat_dict = {}

        for segment_index in range(num_segment):
            data_feat_dict[segment_index] = data_feat[:, segment_index * seg_length: (segment_index + 1) * seg_length]

        return shapelets_feas.values, t_matrix, sdist, data_feat_dict

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'GTS_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        if not os.path.exists('models/' + self._data_kwargs['dataset_dir'][5:]):
            os.makedirs('models/' + self._data_kwargs['dataset_dir'][5:], exist_ok=True)

        config = dict(self._kwargs)
        config['model_state_dict'] = self.GTS_model.state_dict()
        config['epoch'] = epoch
        torch.save(config, self._log_dir + '/epo%d.tar' % epoch)
        # torch.save(self.GTS_model, self._log_dir + '/model_epo%d.pth' % epoch)
        self._logger.info("Saved model at {}".format(epoch))
        return self._log_dir + 'epo%d.tar' % epoch

    def load_model(self, epoch_num=None):
        self._setup_graph()
        model_path = 'models/' + self._data_kwargs['dataset_dir'][5:]
        assert os.path.exists(model_path + '/epo%d.tar' % self._epoch_num), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load(model_path + '/epo%d.tar' % self._epoch_num, map_location='cpu')
        # try:
        self.GTS_model.load_state_dict(checkpoint['model_state_dict'])
        # except RuntimeError:
        #     self.GTS_model = torch.load(self._log_dir + '/model_epo%d.pth' % self._epoch_num)
        self._logger.info("Loaded model at %s_%d" % (model_path, self._epoch_num))
        print('model loaded')

    def _setup_graph(self):
        with torch.no_grad():
            self.GTS_model = self.GTS_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.GTS_model(x, self.forward_index, y)
                break

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)
    
    def obtain_pred_details(self, file_name, save=True):
       # self.load_model()

        with torch.no_grad():
            pred_results = {'y_pred': [], 'y': []}
            pred_results_v2 = {'y_pred': [], 'y': []}
            self.GTS_model = self.GTS_model.eval()

            train_iterator = self._data['train_loader'].get_iterator()
            val_iterator = self._data['val_loader'].get_iterator()
            test_iterator = self._data['test_loader'].get_iterator()
            iterators = [train_iterator, val_iterator, test_iterator]
            losses = []
            mapes = []
            rmses = []
            mses = []

            l_3 = []
            m_3 = []
            r_3 = []
            l_6 = []
            m_6 = []
            r_6 = []
            l_12 = []
            m_12 = []
            r_12 = []
            print('starting to obtain pred results...')
            for iterator in iterators:
                for batch_idx, (x, y) in enumerate(iterator):
                    if batch_idx % 50 == 0:
                        print(batch_idx, 'starting...')
                    x, y = self._prepare_data(x, y)
                    output, adj_v2 = self.GTS_model(x, current_seg_index=self.forward_index)
                    y_true = self.standard_scaler.inverse_transform(y)
                    y_pred = self.standard_scaler.inverse_transform(output)
                    loss = self._compute_loss(y, output)
                    mapes.append(masked_mape_loss(y_pred, y_true).item())
                    mses.append(masked_mse_loss(y_pred, y_true).item())
                    rmses.append(masked_rmse_loss(y_pred, y_true).item())
                    losses.append(loss.item())
                    l_3.append(masked_mae_loss(y_pred[2:3], y_true[2:3]).item())
                    m_3.append(masked_mape_loss(y_pred[2:3], y_true[2:3]).item())
                    r_3.append(masked_mse_loss(y_pred[2:3], y_true[2:3]).item())
                    l_6.append(masked_mae_loss(y_pred[5:6], y_true[5:6]).item())
                    m_6.append(masked_mape_loss(y_pred[5:6], y_true[5:6]).item())
                    r_6.append(masked_mse_loss(y_pred[5:6], y_true[5:6]).item())
                    l_12.append(masked_mae_loss(y_pred[11:12], y_true[11:12]).item())
                    m_12.append(masked_mape_loss(y_pred[11:12], y_true[11:12]).item())
                    r_12.append(masked_mse_loss(y_pred[11:12], y_true[11:12]).item())

                    pred_results['y_pred'].append(output)
                    pred_results['y'].append(y)
                    pred_results_v2['y_pred'].append(y_pred)
                    pred_results_v2['y'].append(y_true)

            mean_loss = np.mean(losses)
            mean_mape = np.mean(mapes)
            mean_rmse = np.sqrt(np.mean(mses))
            message = 'Test: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mean_loss, mean_mape, mean_rmse)
            self._logger.info(message)

            # Followed the DCRNN TensorFlow Implementation
            message = 'Horizon 15mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_3), np.mean(m_3),
                                                                                       np.sqrt(np.mean(r_3)))
            self._logger.info(message)
            message = 'Horizon 30mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_6), np.mean(m_6),
                                                                                       np.sqrt(np.mean(r_6)))
            self._logger.info(message)
            message = 'Horizon 60mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_12), np.mean(m_12),
                                                                                       np.sqrt(np.mean(r_12)))
            self._logger.info(message)

            self._writer.add_scalar('{} loss'.format(self._data_kwargs['dataset_dir'][5:]), mean_loss)

            if save:
                print('starting to save files at', file_name + '.pth')
                print(len(pred_results_v2['y_pred']))
                torch.save(pred_results, file_name + '.pth')
                torch.save(pred_results_v2, file_name + '_v2.pth')

    def evaluate(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.GTS_model = self.GTS_model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []
            mapes = []
            rmses = []
            mses = []

            l_1 = []
            l_2 = []
            l_3 = []
            l_4 = []
            l_5 = []
            l_6 = []
            l_7 = []
            l_8 = []
            l_9 = []
            l_10 = []
            l_11 = []
            l_12 = []

            m_1 = []
            m_2 = []
            m_3 = []
            m_4 = []
            m_5 = []
            m_6 = []
            m_7 = []
            m_8 = []
            m_9 = []
            m_10 = []
            m_11 = []
            m_12 = []

            r_1 = []
            r_2 = []
            r_3 = []
            r_4 = []
            r_5 = []
            r_6 = []
            r_7 = []
            r_8 = []
            r_9 = []
            r_10 = []
            r_11 = []
            r_12 = []

            # pred_results = {'y': [], 'y_pred': []}

            for batch_idx, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)

                output, adj_v2 = self.GTS_model(x, current_seg_index=self.forward_index)


                loss = self._compute_loss(y, output)
                y_true = self.standard_scaler.inverse_transform(y)
                y_pred = self.standard_scaler.inverse_transform(output)
                # pred_results['y'].append(y_true)
                # pred_results['y_pred'].append(y_pred)
                mapes.append(masked_mape_loss(y_pred, y_true).item())
                mses.append(masked_mse_loss(y_pred, y_true).item())
                rmses.append(masked_rmse_loss(y_pred, y_true).item())
                losses.append(loss.item())


                # Followed the DCRNN TensorFlow Implementation
                l_1.append(masked_mae_loss(y_pred[0:1], y_true[0:1]).item())
                l_2.append(masked_mae_loss(y_pred[1:2], y_true[1:2]).item())
                l_3.append(masked_mae_loss(y_pred[2:3], y_true[2:3]).item())
                l_4.append(masked_mae_loss(y_pred[3:4], y_true[3:4]).item())
                l_5.append(masked_mae_loss(y_pred[4:5], y_true[4:5]).item())
                l_6.append(masked_mae_loss(y_pred[5:6], y_true[5:6]).item())
                l_7.append(masked_mae_loss(y_pred[6:7], y_true[6:7]).item())
                l_8.append(masked_mae_loss(y_pred[7:8], y_true[7:8]).item())
                l_9.append(masked_mae_loss(y_pred[8:9], y_true[8:9]).item())
                l_10.append(masked_mae_loss(y_pred[9:10], y_true[9:10]).item())
                l_11.append(masked_mae_loss(y_pred[10:11], y_true[10:11]).item())
                l_12.append(masked_mae_loss(y_pred[11:12], y_true[11:12]).item())

                m_1.append(masked_mape_loss(y_pred[0:1], y_true[0:1]).item())
                m_2.append(masked_mape_loss(y_pred[1:2], y_true[1:2]).item())
                m_3.append(masked_mape_loss(y_pred[2:3], y_true[2:3]).item())
                m_4.append(masked_mape_loss(y_pred[3:4], y_true[3:4]).item())
                m_5.append(masked_mape_loss(y_pred[4:5], y_true[4:5]).item())
                m_6.append(masked_mape_loss(y_pred[5:6], y_true[5:6]).item())
                m_7.append(masked_mape_loss(y_pred[6:7], y_true[6:7]).item())
                m_8.append(masked_mape_loss(y_pred[7:8], y_true[7:8]).item())
                m_9.append(masked_mape_loss(y_pred[8:9], y_true[8:9]).item())
                m_10.append(masked_mape_loss(y_pred[9:10], y_true[9:10]).item())
                m_11.append(masked_mape_loss(y_pred[10:11], y_true[10:11]).item())
                m_12.append(masked_mape_loss(y_pred[11:12], y_true[11:12]).item())

                r_1.append(masked_mse_loss(y_pred[0:1], y_true[0:1]).item())
                r_2.append(masked_mse_loss(y_pred[1:2], y_true[1:2]).item())
                r_3.append(masked_mse_loss(y_pred[2:3], y_true[2:3]).item())
                r_4.append(masked_mse_loss(y_pred[3:4], y_true[3:4]).item())
                r_5.append(masked_mse_loss(y_pred[4:5], y_true[4:5]).item())
                r_6.append(masked_mse_loss(y_pred[5:6], y_true[5:6]).item())
                r_7.append(masked_mse_loss(y_pred[6:7], y_true[6:7]).item())
                r_8.append(masked_mse_loss(y_pred[7:8], y_true[7:8]).item())
                r_9.append(masked_mse_loss(y_pred[8:9], y_true[8:9]).item())
                r_10.append(masked_mse_loss(y_pred[9:10], y_true[9:10]).item())
                r_11.append(masked_mse_loss(y_pred[10:11], y_true[10:11]).item())
                r_12.append(masked_mse_loss(y_pred[11:12], y_true[11:12]).item())


            mean_loss = np.mean(losses)
            mean_mape = np.mean(mapes)
            mean_rmse = np.sqrt(np.mean(mses))
            # mean_rmse = np.mean(rmses) #another option

            if dataset == 'test':

                # Followed the DCRNN PyTorch Implementation
                message = 'Test: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mean_loss, mean_mape, mean_rmse)
                self._logger.info(message)

                # Followed the DCRNN TensorFlow Implementation
                message = 'Horizon 1: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_1), np.mean(m_1),
                                                                                           np.sqrt(np.mean(r_1)))
                self._logger.info(message)
                message = 'Horizon 2: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_2), np.mean(m_2),
                                                                                      np.sqrt(np.mean(r_2)))
                self._logger.info(message)
                message = 'Horizon 3: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_3), np.mean(m_3),
                                                                                      np.sqrt(np.mean(r_3)))
                self._logger.info(message)
                message = 'Horizon 4: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_4), np.mean(m_4),
                                                                                      np.sqrt(np.mean(r_4)))
                self._logger.info(message)
                message = 'Horizon 5: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_5), np.mean(m_5),
                                                                                      np.sqrt(np.mean(r_5)))
                self._logger.info(message)

                message = 'Horizon 6: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_6), np.mean(m_6),
                                                                                           np.sqrt(np.mean(r_6)))
                self._logger.info(message)
                message = 'Horizon 7: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_7), np.mean(m_7),
                                                                                      np.sqrt(np.mean(r_7)))
                self._logger.info(message)
                message = 'Horizon 8: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_8), np.mean(m_8),
                                                                                      np.sqrt(np.mean(r_8)))
                self._logger.info(message)
                message = 'Horizon 9: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_9), np.mean(m_9),
                                                                                      np.sqrt(np.mean(r_3)))
                self._logger.info(message)
                message = 'Horizon 10: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_10), np.mean(m_10),
                                                                                      np.sqrt(np.mean(r_10)))
                self._logger.info(message)
                message = 'Horizon 11: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_11), np.mean(m_11),
                                                                                      np.sqrt(np.mean(r_11)))
                self._logger.info(message)

                message = 'Horizon 12: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_12), np.mean(m_12),
                                                                                           np.sqrt(np.mean(r_12)))
                self._logger.info(message)

                # torch.save(pred_results, dataset + '_' + self._data_kwargs['dataset_dir'].split('/')[1] + '_' + str(self._data_kwargs['period_index']) + '_results_dict.pth')

            self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)

            return mean_loss, mean_mape, mean_rmse

    def source_evolution(self, topk=10):
        start_time = time.time()
        adj_mx = np.zeros((self.num_nodes, self.num_nodes))
        for current_seg_index in range(self.tmat.shape[0]):
            shapelets_adj = self.tmat[current_seg_index, :, :]
            sdist_mat = self.sdist[:, current_seg_index, :]
            nodes2shapelets = {}
            shapelets2nodes = {}
            for shapelet in range(self.tmat.shape[1]):
                s_dist_list = sdist_mat[:, shapelet]
                shapelets2nodes[shapelet] = []
                shapelets2nodes[shapelet].extend(np.argpartition(s_dist_list, topk)[:topk].tolist())

            for node in range(self.num_nodes):
                dist_list = sdist_mat[node, :]
                nodes2shapelets[node] = []
                nodes2shapelets[node].extend(np.argpartition(dist_list, topk)[:topk].tolist())

            # print('source evolution', current_seg_index, shapelets2nodes, nodes2shapelets)

            # 首先应该是确定哪些shapelets当前对应哪些nodes，再弄adj
            for i in range(self.num_nodes):
                for shapelet in nodes2shapelets[i]:
                    for map_node in shapelets2nodes[shapelet]:
                        adj_mx[i, map_node] += shapelets_adj[shapelet, :].max()

            for i in range(shapelets_adj.shape[0]):
                for j in range(shapelets_adj.shape[0]):
                    for map_node in shapelets2nodes[i]:
                        adj_mx[map_node, shapelets2nodes[j]] += shapelets_adj[i, j]
            # print(current_seg_index, np.nonzero(adj_mx))
            assert not np.all(adj_mx[current_seg_index, :, :] == 0)
            # adj_mx[current_seg_index, :, :] += np.eye(self.num_nodes, self.num_nodes)

        end_time = time.time()

        if (end_time - start_time) / 60 > 1:
            print('graph post-process cost time %.4f' % ((end_time - start_time) / 60), 'min.')

        if 'Tencent' in self._data_kwargs['dataset_dir']:
            np.save(self._data_kwargs['dataset_dir'] +'/%s_shapelet2node_trans_adj_period%d_top%d.npy' % (self._data_kwargs['dataset_dir'][5:23], self._data_kwargs['period_index'], topk), adj_mx)
        else:
            np.save(self._data_kwargs['dataset_dir'] +'/%s_shapelet2node_trans_adj_top%d.npy' % (self._data_kwargs['dataset_dir'][5:], topk), adj_mx)
        print('source graph saved!')
        return adj_mx

    def _train(self, base_lr,
               steps, patience=200, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=True,
               test_every_n_epochs=10, epsilon=1e-8, topk=10, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        self.forward_index = 0
        if self.opt == 'adam':
            optimizer = torch.optim.Adam(self.GTS_model.parameters(), lr=base_lr, eps=epsilon)
        elif self.opt == 'sgd':
            optimizer = torch.optim.SGD(self.GTS_model.parameters(), lr=base_lr)
        else:
            optimizer = torch.optim.Adam(self.GTS_model.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=float(lr_decay_ratio))

        self._logger.info('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num
        # if batches_seen == 0:
        # train_iterator = self._data['train_loader'].get_iterator()
        # self.GTS_model.init_evolution(train_iterator)
        loss_dict = []
        if 'Tencent' in self._data_kwargs['dataset_dir']:
            if os.path.exists(self._data_kwargs['dataset_dir'] +'/%s_shapelet2node_trans_adj_period%d_top%d.npy' % (self._data_kwargs['dataset_dir'][5:23], self._data_kwargs['period_index'], topk)):
                adj_mx = np.load(self._data_kwargs['dataset_dir'] +'/%s_shapelet2node_trans_adj_period%d_top%d.npy' % (self._data_kwargs['dataset_dir'][5:23], self._data_kwargs['period_index'], topk))
                print(self._data_kwargs['dataset_dir'] +'/%s_shapelet2node_trans_adj_period%d_top%d.npy' % (self._data_kwargs['dataset_dir'][5:23], self._data_kwargs['period_index'], topk) + ' loaded.')
            else:
                adj_mx = self.source_evolution()
        else:
            if os.path.exists(self._data_kwargs['dataset_dir'] + '/%s_shapelet2node_trans_adj_top%d.npy' % (self._data_kwargs['dataset_dir'][5:], topk)):
                adj_mx = np.load(self._data_kwargs['dataset_dir'] + '/%s_shapelet2node_trans_adj_top%d.npy' % (self._data_kwargs['dataset_dir'][5:], topk))
                print(self._data_kwargs['dataset_dir'] + '/%s_shapelet2node_trans_adj_top%d.npy' % (self._data_kwargs['dataset_dir'][5:], topk) + ' loaded.')
            else:
                adj_mx = self.source_evolution()

        for epoch_num in range(self._epoch_num, epochs):
            print("Num of epoch:", epoch_num)
            self.GTS_model = self.GTS_model.train()
            # if epoch_num % 20 == 0:
            #     self.GTS_model.init_evolution(train_iterator)
            train_iterator = self._data['train_loader'].get_iterator()
            # val_iterator = self._data['val_loader'].get_iterator()
            losses = []
            start_time = time.time()
            time_recoder = []
            time_recoder.append(start_time)
            optimizer.zero_grad(set_to_none=True)
            self.forward_index = 0
            # self.GTS_model.adj_mx = adj_mx
            for batch_idx, (x, y) in enumerate(train_iterator):
                time0 = time.time()
                # optimizer.zero_grad()
                # print(batch_idx, 'check x, y', x.shape, y.shape)
                x, y = self._prepare_data(x, y)
                # print(batch_idx, 'check x, y', x.shape, y.shape)
                # 如何计算演化时间阶段？
                # self.evolving_segment_cnt += 1
                # if batch_idx + 1 % 5 == 0:
                #     mid_time = time.time()
                #     print(batch_idx, 'batch time cost', (mid_time - time_recoder[-1]) / 60, 'min.')
                #     time_recoder.append(mid_time)
                if len(time_recoder) > 1:
                    del time_recoder[:-1]
                # if 0 < batches_seen <= 48:
                #     self.forward_index = 0
                time1 = time.time()
                # if batches_seen == 0:
                #     self.forward_index = 0
                #     adj_mx = self.source_evolution(self.forward_index)
                #     self.GTS_model.adj_mx = adj_mx
                # if batches_seen > 0 and batches_seen / 60 * x.shape[1] == 0:
                self.forward_index = batch_idx
                # cur_adj_mx = self.source_evolution(self.forward_index)
                self.GTS_model.adj_mx = adj_mx[self.forward_index, :, :]
                # print('next period', self.forward_index)
                output, adj_v2 = self.GTS_model(x, self.forward_index, y, batches_seen)
                # if epoch_num % 100 == 0:
                #     self.GTS_model.init_evolution(train_iterator)
                # consolidation_loss = self.GTS_model.compute_consolidation_loss(train_iterator)
                # print('test consolidation_loss', consolidation_loss)
                if (epoch_num % epochs) == epochs - 1:
                    output, adj_v2 = self.GTS_model(x, self.forward_index, y, batches_seen)
                time2 = time.time()
                # print('time2 - time1', (time2 - time1) / 60, 'min.')

                if batches_seen == 0:
                    if self.opt == 'adam':
                        optimizer = torch.optim.Adam(self.GTS_model.parameters(), lr=base_lr, eps=epsilon)
                    elif self.opt == 'sgd':
                        optimizer = torch.optim.SGD(self.GTS_model.parameters(), lr=base_lr)
                    else:
                        optimizer = torch.optim.Adam(self.GTS_model.parameters(), lr=base_lr, eps=epsilon)

                self.GTS_model.to(self.device)

                loss = self._compute_loss(y, output)
                losses.append(loss.item())

                self._logger.debug(loss.item())
                batches_seen += 1
                # print('loss', loss, losses)
                loss.backward()

                # gradient clipping - this does it in place
                # if batch_idx % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.GTS_model.parameters(), self.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                time3 = time.time()
                # print('time3 - time2', (time3 - time2) / 60, 'min.')
                # print(batch_idx, 'batch total training time cost', (time3 - time0) / 60, 'min.')

                # torch.save(adj_v1, self._log_dir + '/batchidx_%d_source_evo_graph.pth' % batch_idx)
                # torch.save(adj_v2, self._log_dir + '/batchidx_%d_target_evo_graph.pth' % batch_idx)
            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")
            end_time = time.time()

            # if epoch_num % evaluation_steps == 0:
            val_loss, val_mape, val_rmse = self.evaluate(dataset='val', batches_seen=batches_seen)
            end_time2 = time.time()
            self._writer.add_scalar('training loss',
                                    np.mean(losses),
                                    batches_seen)
            loss_dict.append(np.mean(losses))
            print('loss_dict', len(loss_dict))
            print('losses', len(losses))
            if (epoch_num % log_every) == log_every - 1:
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, val_mape: {:.4f}, val_rmse: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s, {:.1f}s'.format(epoch_num, epochs, batches_seen,
                                                    np.mean(losses), val_loss, val_mape, val_rmse,
                                                    lr_scheduler.get_lr()[0],
                                                    (end_time - start_time), (end_time2 - start_time))
                self._logger.info(message)

            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                test_loss, test_mape, test_rmse = self.evaluate(dataset='test', batches_seen=batches_seen)
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f}, test_mape: {:.4f}, test_rmse: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s, {:.1f}s'.format(epoch_num, epochs, batches_seen,
                                                    np.mean(losses), test_loss, test_mape, test_rmse,
                                                    lr_scheduler.get_lr()[0],
                                                    (end_time - start_time), (end_time2 - start_time))
                self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                if save_model:
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss

            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break
        if 'Alibaba' in self._data_kwargs['dataset_dir'][5:]:
            torch.save(loss_dict, '{}_loss_record.pth'.format(self._data_kwargs['dataset_dir'][5:]))
        elif 'Tencent' in self._data_kwargs['dataset_dir'][5:]:
            torch.save(loss_dict, '{}_{}_loss_record.pth'.format(self._data_kwargs['dataset_dir'][5:23], self._data_kwargs['period_index']))

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(self.device), y.to(self.device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        # print('line 509 at supervisor.py', x.shape)
        # print(x.size())
        # print(self.seq_len, batch_size, self.num_nodes, self.input_dim)
        # print(self.output_dim, self.horizon)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        # 加上consolidation loss
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)
