import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import autograd
from model.pytorch.cell import DCGRUCell, EvoLSTMCell
import numpy as np
from .causal_cnn import CausalCNN
from torch.utils.data import DataLoader
import torch.nn.functional as func
import time



# from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
# from dgl.nn.pytorch import GATv2Conv
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, **model_kwargs):
        # self.adj_mx = adj_mx
        # print(model_kwargs['max_diffusion_step'])
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 2000))
        self.filter_type = model_kwargs.get('filter_type', 'dual_random_walk')
        self.num_nodes = int(model_kwargs.get('num_nodes', 3960))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 2))
        self.rnn_units = int(model_kwargs.get('rnn_units', 128))
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.batch_size = int(model_kwargs.get('batch_size', 5))
        self.parameter_dict = {'max_diffusion_step': self.max_diffusion_step, 'cl_decay_steps': self.cl_decay_steps,
                          'filter_type': self.filter_type, 'num_nodes': self.num_nodes, 'num_rnn_layers': self.num_rnn_layers,
                          'rnn_units': self.rnn_units, 'hidden_state_size': self.hidden_state_size, 'batch_size': self.batch_size}


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, data_dict, device, target_evolution=False, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.target_evolution = target_evolution
        self.adj = None
        self.device = device
        self.data_dict = data_dict
        self.forward_index = 0
        self.cached_features = []
        self.dcgru_layers = nn.ModuleList([DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes, device=self.device,
                                                     filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def target_evolution_func(self, data_iterator, criterion=func.mse_loss, device=None):
        self._history_consolidation(data_iterator, criterion, device=self.device)

    def _history_consolidation(self, data_iterator, criterion, device):
        for param_name, param in self.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.register_buffer(_buff_param_name + '_estimated_mean', param.data.clone())
        _buff_param_names = [param[0].replace('.', '__') for param in self.named_parameters()]
        est_fisher_info = {name: 0.0 for name in _buff_param_names}
        hidden_states = [None]
        for i, data in enumerate(data_iterator):
            x = torch.from_numpy(data[0]).float()
            y = torch.from_numpy(data[1]).float()
            x = x.permute(1, 0, 2, 3)
            y = y.permute(1, 0, 2, 3)
            batch_size = x.size(1)
            x = x.view(12, batch_size, self.num_nodes * 2)
            y = y[..., :1].view(12, batch_size, self.num_nodes * 1)
            for t in range(x.size(0)):
                pred, hidden_state = self.forward(x[t], self.adj, hidden_states[-1])
                log_likelihood = criterion(y, pred, reduction='mean')
                grad_log_liklihood = autograd.grad(log_likelihood, self.parameters())
                hidden_states.append(hidden_state)
            for name, grad in zip(_buff_param_names, grad_log_liklihood):
                est_fisher_info[name] += grad.data.clone() ** 2
        for name in _buff_param_names:
            self.register_buffer(name + '_estimated_fisher', est_fisher_info[name])

    def compute_consolidation_loss(self):
        losses = []
        ewc_lambda = 0.0001
        for param_name, param in self.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            estimated_mean = getattr(self, '{}_estimated_mean'.format(_buff_param_name))
            estimated_fisher = getattr(self, '{}_estimated_fisher'.format(_buff_param_name))
            losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
        return 1 * (ewc_lambda / 2) * sum(losses)

    def forward(self, inputs, adj_mx, forward_index, hidden_state=None):
        """
        Encoder forward pass.
        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        # print(inputs.size())
        batch_size, _ = inputs.size()
        self.adj = adj_mx
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj_mx)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        if len(self.cached_features) == 0:
            self.cached_features.append(output.reshape(-1, self.num_nodes, self.rnn_units))
        else:
            self.cached_features = self.cached_features[-1:]
            self.cached_features.append(output.reshape(-1, self.num_nodes, self.rnn_units))

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class EncoderModel_v2(nn.Module, Seq2SeqAttrs):
    def __init__(self, para, sdist, tmat, device, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.adj = None
        self.device = device
        self.num_Evo_layers = para['Evo_layers']
        self.node_dim = para['node_dim']
        self.num_shapelets = para['n_shapelets']
        self.embedding_dim = para['embedding_dim']
        self.dim_fc = para['dim_fc']
        # self.num_shapelets = para['n_shapelets']
        self.fc = torch.nn.Linear(self.dim_fc, self.embedding_dim)
        self.bn = torch.nn.BatchNorm1d(self.embedding_dim)
        self.fc_out = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)
        self.sdist = sdist
        self.tmat = tmat
        self.num_rnn_layers -= 1
        print('encoder num_rnn_layers', self.num_rnn_layers)
        print('encoder evo_lstm_layers', self.num_Evo_layers)
        self.Evo_layers = nn.ModuleList([EvoLSTMCell(para, is_training=True, device=self.device) for _ in range(self.num_Evo_layers)])
        self.dcgru_layers = nn.ModuleList([DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes, device=self.device,
                                                     filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def source_evolution(self, current_seg_index, adj):
        start_time = time.time()
        adj_mx = np.zeros((self.num_nodes, self.num_nodes))  #  这里需要初始化零矩阵吗？？？
        shapelets_adj = self.tmat[current_seg_index, :, :]
        sdist_mat = adj.cpu().detach().numpy()
        nodes2shapelets = {}
        shapelets2nodes = {}
        for shapelet in range(self.tmat.shape[1]):
            shapelets2nodes[shapelet] = []

        for node in range(self.num_nodes):
            dist_list = sdist_mat[node, :]
            nodes2shapelets[node] = []
            nodes2shapelets[node].extend(np.argpartition(dist_list, 5)[:5].tolist())

        # 首先应该是确定哪些shapelets当前对应哪些nodes，再弄adj
        for i in range(self.num_nodes):
            for shapelet in nodes2shapelets[i]:
                for map_node in shapelets2nodes[shapelet]:
                    adj_mx[i, map_node] += 1

        for i in range(shapelets_adj.shape[0]):
            for j in range(shapelets_adj.shape[0]):
                adj_mx[shapelets2nodes[i], shapelets2nodes[j]] += shapelets_adj[i, j]
        end_time = time.time()
        if (end_time - start_time) / 60 > 1:
            print('graph post-process cost time', (end_time - start_time) / 60, 'min.')
        return adj_mx

    def forward(self, inputs, adj_mx, forward_index, hidden_state=None):
        """
        Encoder forward pass.
        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        self.adj = adj_mx
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=self.device)
        hidden_states = []
        output = inputs
        # print('line 151', hidden_state.shape)
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj_mx)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        node_embeddings_list = []
        for layer_num, evolstm_layer in enumerate(self.Evo_layers):
            if layer_num == 0 and forward_index >= 0:
                node_embeddings, attention_weights = evolstm_layer(self.sdist[:, forward_index - 1, :], self.sdist[:, forward_index, :], self.num_nodes)
                node_embeddings_list.append(node_embeddings)
            elif layer_num > 0 and forward_index > 0:
                node_embeddings, attention_weights = evolstm_layer(self.sdist[:, forward_index - 1, :], self.sdist[:, forward_index, :], self.num_nodes, node_embeddings)
                node_embeddings_list.append(node_embeddings)
            # elif forward_index == 0:
            #     node_embeddings, attention_weights = evolstm_layer(self.sdist[:, forward_index, :], self.sdist[:, forward_index, :], self.num_nodes)
            #     node_embeddings_list.append(node_embeddings)
        # self.pre_node_embedding = node_embeddings

        x = self.fc(node_embeddings)
        x = F.relu(x)
        x = self.bn(x)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)
        # print(x.shape)

        # 第一种方案，source_evolution, 极其耗时，使用需要一轮1200s左右
        # 第二种方案，sdist*adj，暂时没数据
        # 第三种方案，adj0*adj1，好像比较高效，可以一轮350s以内
        adj0 = x[:, 0].clone().reshape(self.num_nodes, -1).detach()
        adj1 = x[:, 1].clone().reshape(-1, self.num_nodes).detach()
        # print('adj shape at line 177', adj.shape)
        # adj = self.source_evolution(forward_index, adj)
        # pre_adj = self.sdist[:, forward_index, :]
        # adj = np.matmul(pre_adj, adj)
        adj = torch.matmul(adj0, adj1)

        return output, torch.stack(hidden_states), adj, node_embeddings, attention_weights  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, device, target_evolution=False, **model_kwargs):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.horizon = int(model_kwargs.get('horizon', 1))  # for the decoder
        self.target_evolution = target_evolution
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.adj = None
        self.device = device
        self.feat_list = []
        self.forward_index = 0
        # self.data_dict = data_dict
        print('decoder num_rnn_layers', self.num_rnn_layers)
        self.dcgru_layers = nn.ModuleList([DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes, device=self.device,
                                                     filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def target_evolution_func(self, data_iterator, criterion=func.mse_loss, device=None):
        self._history_consolidation(data_iterator, criterion, device=self.device)

    def _history_consolidation(self, data_iterator, criterion, device):
        for param_name, param in self.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.register_buffer(_buff_param_name + '_estimated_mean', param.data.clone())
            # print('registered para', _buff_param_name + '_estimated_mean')
        _buff_param_names = [param[0].replace('.', '__') for param in self.named_parameters()]
        est_fisher_info = {name: 0.0 for name in _buff_param_names}
        hidden_states = [None]
        for i, data in enumerate(data_iterator):
            x = torch.from_numpy(data[0]).float().to(device)
            y = torch.from_numpy(data[1]).float().to(device)
            x = x.permute(1, 0, 2, 3)
            y = y.permute(1, 0, 2, 3)
            batch_size = x.size(1)
            x = x.view(12, batch_size, self.num_nodes * 2)
            y = y[..., :1].view(12, batch_size, self.num_nodes * 1)
            # for t in range(x.size(0)):
            pred, hidden_state = self.forward(x[0], self.adj, self.forward_index, hidden_states[-1])
            log_likelihood = criterion(y[0], pred, reduction='mean')
            grad_log_likelihood = autograd.grad(log_likelihood, self.parameters(), allow_unused=True, retain_graph=True)
            hidden_states.append(hidden_state)
            # print('log_likelihood', log_likelihood)
            # print('grad_log_likelihood', grad_log_likelihood)
            for name, grad in zip(_buff_param_names, grad_log_likelihood):
                # print(name, type(grad), grad)
                try:
                    est_fisher_info[name] += grad.data.clone() ** 2
                except AttributeError:
                    continue
            # if i > 5:
            #     break
        # print('est_fisher_info', est_fisher_info)
        for name in _buff_param_names:
            if not isinstance(est_fisher_info[name], torch.Tensor):
                est_fisher_info[name] = torch.Tensor([est_fisher_info[name]])
            self.register_buffer(name + '_estimated_fisher', est_fisher_info[name])
            # print('registered para', name + '_estimated_fisher')

    def compute_consolidation_loss(self):
        # for param_name, param in self.named_parameters():
        #     _buff_param_name = param_name.replace('.', '__')
        #     self.register_buffer(_buff_param_name + '_estimated_mean', param.data.clone())
        losses = []
        ewc_lambda = 0.0001
        for param_name, param in self.named_parameters():
            # print('-------check consolidation loss-------')
            # print('param_name', param_name, param)
            # print('estimated_fisher', losses)
            _buff_param_name = param_name.replace('.', '__')
            try:
                estimated_mean = getattr(self, '{}_estimated_mean'.format(_buff_param_name))
                estimated_fisher = getattr(self, '{}_estimated_fisher'.format(_buff_param_name))
                # print('estimated_mean', estimated_mean, estimated_fisher)
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
            except AttributeError:
                continue
            # print(losses[-1])

        # print('estimated_fisher', losses)
        if len(losses) > 0:
            return 1 * (ewc_lambda / 2) * sum(losses)
        else:
            return 0

    def forward(self, inputs, adj_mx, forward_index, hidden_state=None):
        """
        Decoder forward pass.
        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        if not isinstance(adj_mx, torch.Tensor):
            self.adj = torch.from_numpy(adj_mx).float().to(self.device)
        else:
            self.adj = adj_mx
            self.adj.to(self.device)
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=self.device)
        hidden_states = []
        output = inputs.to(self.device)
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            if self.target_evolution and layer_num == 0 and self.forward_index < forward_index:
                # pre_data = torch.from_numpy(self.data_dict[self.forward_index])
                # cur_data = torch.from_numpy(self.data_dict[self.forward_index + 1])
                pre_feat = self.feat_list[0]
                cur_feat = self.feat_list[1]
                next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj_mx, pre_feat, cur_feat, target_evolution=True)
            else:
                next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj_mx)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        self.forward_index = forward_index
        projected = self.projection_layer(output.view(-1, self.rnn_units)).to(self.device)
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class GTSModel_v2(nn.Module, Seq2SeqAttrs):
    def __init__(self, data_dict, sdist, logger, device, encoder_evo=False, decoder_evo=False, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.num_shapelets = int(model_kwargs.get('num_shapelets', 1))
        self.encoder_model = EncoderModel(sdist, device=device, target_evolution=encoder_evo, **model_kwargs)
        self.decoder_model = DecoderModel(data_dict, device=device, target_evolution=decoder_evo, **model_kwargs)
        self.data_dict = data_dict
        self.sdist = sdist
        # self.tmat = tmat
        self.device = torch.device(device)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger
        self.adj_mx = np.zeros((self.num_nodes, self.num_nodes))
        self.forward_index = 0

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def compute_consolidation_loss(self, data_iterator):
        start_time = time.time()
        consolidation_losses = 0
        if self.encoder_model.target_evolution:
            # self.encoder_model.target_evolution_func(data_iterator)
            encoder_consolidation_loss = self.encoder_model.compute_consolidation_loss()
            consolidation_losses += encoder_consolidation_loss
        if self.decoder_model.target_evolution:
            # self.decoder_model.target_evolution_func(data_iterator)
            decoder_consolidation_loss = self.decoder_model.compute_consolidation_loss()
            consolidation_losses += decoder_consolidation_loss
        end_time = time.time()
        print('consolidation loss compute time cost', (end_time - start_time) / 60, 'min.')
        return consolidation_losses

    def init_evolution(self, train_iterator):
        if self.encoder_model.target_evolution:
            self.encoder_model.target_evolution_func(train_iterator)

        if self.decoder_model.target_evolution:
            self.decoder_model.target_evolution_func(train_iterator)

    def encoder(self, inputs, adj):
        """
        Encoder forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], adj, self.forward_index, encoder_hidden_state)

        return encoder_hidden_state, self.encoder_model.cached_features

    def decoder(self, encoder_hidden_state, adj, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, adj, self.forward_index,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    # def source_evolution(self, current_seg_index):
    #     start_time = time.time()
    #     self.adj_mx = np.zeros((self.num_nodes, self.num_nodes))  #  这里需要初始化零矩阵吗？？？
    #     shapelets_adj = self.tmat[current_seg_index, :, :]
    #     sdist_mat = self.sdist[:, current_seg_index, :]
    #     nodes2shapelets = {}
    #     shapelets2nodes = {}
    #     for shapelet in range(self.tmat.shape[1]):
    #         shapelets2nodes[shapelet] = []
    #
    #     for node in range(self.num_nodes):
    #         dist_list = sdist_mat[node, :]
    #         nodes2shapelets[node] = []
    #         nodes2shapelets[node].extend(np.argpartition(dist_list, 5)[:5].tolist())
    #
    #     # 首先应该是确定哪些shapelets当前对应哪些nodes，再弄adj
    #     for i in range(self.num_nodes):
    #         for shapelet in nodes2shapelets[i]:
    #             for map_node in shapelets2nodes[shapelet]:
    #                 self.adj_mx[i, map_node] += 1
    #
    #     for i in range(shapelets_adj.shape[0]):
    #         for j in range(shapelets_adj.shape[0]):
    #             self.adj_mx[shapelets2nodes[i], shapelets2nodes[j]] += shapelets_adj[i, j]
    #     end_time = time.time()
    #     print('graph post-process cost time', (end_time - start_time) / 60, 'min.')

    def forward(self, inputs, current_seg_index, labels=None, batches_seen=None):
        """
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        # if self.forward_index <= current_seg_index:
        #     self.source_evolution(current_seg_index)
        adj = self.adj_mx
        encoder_hidden_state, features_list = self.encoder(inputs, adj)
        self._logger.debug("Encoder complete, starting decoder")
        self.decoder_model.feat_list = features_list
        outputs = self.decoder(encoder_hidden_state, adj, labels, batches_seen=batches_seen)
        self._logger.debug("Decoder complete")
        self.forward_index = current_seg_index
        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )

        return outputs, adj


class GTSModel_v3(nn.Module, Seq2SeqAttrs):
    def __init__(self, para, sdist, tmat, logger, device, decoder_evo=False, **model_kwargs):
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        super().__init__()
        self.num_shapelets = int(model_kwargs.get('num_shapelets', 1))
        self.encoder_model = EncoderModel_v2(para, sdist, tmat, device=device)
        self.decoder_model = DecoderModel(device=device, target_evolution=decoder_evo, **model_kwargs)
        # self.data_dict = data_dict
        self.sdist = sdist
        # self.tmat = tmat
        self.device = torch.device(device)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger
        self.adj_mx = np.zeros((self.num_nodes, self.num_nodes))
        self.forward_index = 0

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def compute_consolidation_loss(self, data_iterator):
        start_time = time.time()
        consolidation_losses = 0
        if self.encoder_model.target_evolution:
            # self.encoder_model.target_evolution_func(data_iterator)
            encoder_consolidation_loss = self.encoder_model.compute_consolidation_loss()
            consolidation_losses += encoder_consolidation_loss
        if self.decoder_model.target_evolution:
            # self.decoder_model.target_evolution_func(data_iterator)
            decoder_consolidation_loss = self.decoder_model.compute_consolidation_loss()
            consolidation_losses += decoder_consolidation_loss
        end_time = time.time()
        print('consolidation loss compute time cost', (end_time - start_time) / 60, 'min.')
        return consolidation_losses

    def init_evolution(self, train_iterator):
        if self.encoder_model.target_evolution:
            self.encoder_model.target_evolution_func(train_iterator)

        if self.decoder_model.target_evolution:
            self.decoder_model.target_evolution_func(train_iterator)

    def encoder(self, inputs, adj):
        """
        Encoder forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        seq_len = inputs.size(0)
        for t in range(seq_len):
            _, encoder_hidden_state, adj, node_embeddings, attention_weights = self.encoder_model(inputs[t], adj, self.forward_index, encoder_hidden_state)
        # print(node_embeddings.unsqueeze(0).reshape((-1, 5, self.hidden_state_size)))
        encoder_hidden_state = torch.cat([encoder_hidden_state, node_embeddings.unsqueeze(0).reshape((-1, self.batch_size, self.hidden_state_size))])
        parametered_adj = adj
        return encoder_hidden_state.detach(), parametered_adj

    def decoder(self, encoder_hidden_state, adj, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, adj, self.forward_index,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, current_seg_index, labels=None, batches_seen=None):
        """
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        # if self.forward_index <= current_seg_index:
        #     self.source_evolution(current_seg_index)
        adj = self.adj_mx
        encoder_hidden_state, adj_embeddings = self.encoder(inputs, adj)
        self._logger.debug("Encoder complete, starting decoder")
        # self.decoder_model.feat_list = features_list
        adj = adj_embeddings
        outputs = self.decoder(encoder_hidden_state, adj, labels, batches_seen=batches_seen)
        self._logger.debug("Decoder complete")
        self.forward_index = current_seg_index
        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )

        return outputs, adj

