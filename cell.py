import numpy as np
import torch
from torch.nn import functional as F
from torch import autograd
from scipy.spatial import distance
from lib import utils
import networkx as nx
import time

# dev = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# device = torch.device('cpu')


class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str, device):

        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type
        self.device = device

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=self.device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=self.device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class DCGRUCell(torch.nn.Module):
    def __init__(self, num_units, max_diffusion_step, num_nodes, device, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True):
        """
        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        # self._supports = []
        self.device = torch.device(device)
        # global dev
        self._use_gc_for_ru = use_gc_for_ru
        # self._target_evolution = target_evolution
        self.filter_type = filter_type
        self.cache_pool = []
        self.forward_index = 0
        # if self._target_evolution:
        #     pass
        # supports = []
        # supports.append(adj_mx)

        self._fc_params = LayerParams(self, 'fc', device=device)
        self._gconv_params = LayerParams(self, 'gconv', device=device)

    def _detect_evolution(self, adj, pre_input, cur_input, hx, topk):
        start_time = time.time()
        output_size = 2 * self._num_units
        # pre_feat = torch.sigmoid(self._gconv(pre_input, adj, hx, output_size, bias_start=1.0))
        # cur_feat = torch.sigmoid(self._gconv(cur_input, adj, hx, output_size, bias_start=1.0))
        # pre_feat = torch.reshape(pre_input, (-1, self._num_nodes, 1))
        # cur_feat = torch.reshape(cur_input, (-1, self._num_nodes, 1))
        pre_feat = pre_input.permute(1, 0, 2).cpu().detach()
        cur_feat = cur_input.permute(1, 0, 2).cpu().detach()
        score = []
        for i in range(pre_feat.shape[0]):
            _score = 0.0
            for j in range(pre_feat.shape[2]):
                pre_feat[i, :, j] = (pre_feat[i, :, j] - min(pre_feat[i, :, j])) / (max(pre_feat[i, :, j]) - min(pre_feat[i, :, j]))
                cur_feat[i, :, j] = (cur_feat[i, :, j] - min(cur_feat[i, :, j])) / (max(cur_feat[i, :, j]) - min(cur_feat[i, :, j]))
                pre_prob, _ = np.histogram(pre_feat[i, :, j], bins=10, range=(0, 1))
                pre_prob = pre_prob * 1.0 / sum(pre_prob)
                cur_prob, _ = np.histogram(cur_feat[i, :, j], bins=10, range=(0, 1))
                cur_prob = cur_prob * 1.0 / sum(cur_prob)
                _score += distance.jensenshannon(pre_prob, cur_prob)
            score.append(_score)
        end_time = time.time()
        if (end_time - start_time) / 60 > 1:
            print('js detection calculation cost time', (end_time - start_time) / 60, 'min.')
        return np.argpartition(np.asarray(score), -topk)[-topk:]

    def target_evolution(self, adj_mx, pre_input, cur_input, hx, topk, filter_type):
        # adj_mx = adj_mx
        start_time = time.time()
        selected_node_list = self._detect_evolution(adj_mx, pre_input, cur_input, hx, topk)
        mask_matrix = np.zeros((adj_mx.shape[0], adj_mx.shape[0]))
        for influence_node in selected_node_list:
            mask_matrix[influence_node, :] = 1
            mask_matrix[:, influence_node] = 1
        new_adj_mx = mask_matrix * adj_mx
        if filter_type == "laplacian":
            adj_mx = utils.calculate_scaled_laplacian(new_adj_mx, lambda_max=None)
        elif filter_type == "random_walk":
            adj_mx = utils.calculate_random_walk_matrix(new_adj_mx).T
        elif filter_type == "dual_random_walk":
            adj_mx = self._calculate_random_walk_matrix(new_adj_mx).t()
        else:
            adj_mx = utils.calculate_scaled_laplacian(new_adj_mx)
        end_time = time.time()
        if (end_time - start_time) / 60 > 1:
            print('node detection cost time', (end_time - start_time) / 60, 'min.')
        return adj_mx

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=self.device)
        return L

    def _calculate_random_walk_matrix(self, adj_mx):

        # tf.Print(adj_mx, [adj_mx], message="This is adj: ")
        if not isinstance(adj_mx, torch.Tensor):
            adj_mx = torch.from_numpy(adj_mx).float().to(self.device)
        # print('line 141', adj_mx.shape)
        adj_mx = adj_mx.to(self.device) + torch.eye(int(adj_mx.shape[0])).to(self.device)
        d = torch.sum(adj_mx, 1)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(self.device), d_inv)
        d_mat_inv = torch.diag(d_inv)
        random_walk_mx = torch.mm(d_mat_inv, adj_mx)
        return random_walk_mx.to(self.device)

    def forward(self, inputs, hx, adj, pre_data=None, cur_data=None, target_evolution=False):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)
        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        # self.cache_pool.append(inputs)
        # if len(self.cache_pool) > 2:
        #     self.cache_pool = self.cache_pool[-2:]

        if target_evolution and not pre_data is None and not cur_data is None:
            adj_mx = self.target_evolution(adj, pre_data, cur_data, hx, 5, self.filter_type)
        else:
            if self.filter_type == "laplacian":
                adj_mx = utils.calculate_scaled_laplacian(adj, lambda_max=None)
            elif self.filter_type == "random_walk":
                adj_mx = utils.calculate_random_walk_matrix(adj).T
            elif self.filter_type == "dual_random_walk":
                adj_mx = self._calculate_random_walk_matrix(adj).t()
            else:
                raise NotImplementedError('the filter type is not defined!')

        # adj_mx = self._build_sparse_matrix(adj_mx)

        output_size = 2 * self._num_units

        # target evolution
        if self._use_gc_for_ru:
            fn = self._gconv
        else:
            fn = self._fc
        value = torch.sigmoid(fn(inputs, adj_mx, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs, adj_mx, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value

    def _gconv(self, inputs, adj_mx, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1)).to(self.device)
        state = torch.reshape(state, (batch_size, self._num_nodes, -1)).to(self.device)
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if not isinstance(adj_mx, torch.Tensor):
            adj_mx = torch.from_numpy(adj_mx).float().to(self.device)

        if self._max_diffusion_step == 0:
            pass
        else:

            x1 = torch.sparse.mm(adj_mx, x0)
            x = self._concat(x, x1)

            for k in range(2, self._max_diffusion_step + 1):
                x2 = 2 * torch.sparse.mm(adj_mx, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        num_matrices = self._max_diffusion_step + 1   # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])


class EvoLSTMCell(torch.nn.Module):
    def __init__(self, param, is_training, device):
        super().__init__()
        self.forward_index = 0
        self.n_nodes = param['n_shapelets']
        # self.n_nodes = param['num_nodes']
        self.n_features = param['node_dim']
        self.graph_dim = param['graph_dim']
        self.n_event = 0
        self.device = device
        self.__GetLearnableParams__(is_training)

    def __weights__(self, input_dim, output_dim, name, init=True, std=0.1, reg=None):
        if init:
            return torch.randn(input_dim, output_dim, device=self.device, requires_grad=True) * 0 + std
        else:
            return torch.rand(input_dim, output_dim, device=self.device, requires_grad=True)

    def __bias__(self, output_dim, name, init=True):
        if init:
            return torch.ones(output_dim, device=self.device, requires_grad=True)
        else:
            return torch.rand(output_dim, device=self.device, requires_grad=True)

    def __GetLearnableParams__(self, is_training):
        self.Win = self.__weights__(self.n_features, self.n_features, name='In_node_weight', init=is_training)
        self.bin = self.__bias__(self.n_features, name='In_node_bias', init=is_training)

        self.Wout = self.__weights__(self.n_features, self.n_features, name='Out_node_weight', init=is_training)
        self.bout = self.__bias__(self.n_features, name='Out_node_bias', init=is_training)

        self.Wa = self.__weights__(self.n_features, 1, name='Attention_weight', init=is_training)

        self.Wi_n = self.__weights__(self.n_features + self.graph_dim, self.n_features, name='Input_node_weight_1',
                                     init=is_training)
        self.Ui_n = self.__weights__(self.n_features, self.n_features, name='Input_node_weight_2', init=is_training)
        self.bi_n = self.__bias__(self.n_features, name='Input_node_bias', init=is_training)

        self.Wf_n = self.__weights__(self.n_features + self.graph_dim, self.n_features, name='Forget_node_weight_1',
                                     init=is_training)
        self.Uf_n = self.__weights__(self.n_features, self.n_features, name='Forget_node_weight_2', init=is_training)
        self.bf_n = self.__bias__(self.n_features, name='Forget_node_bias', init=is_training)

        self.Wo_n = self.__weights__(self.n_features + self.graph_dim, self.n_features, name='Output_node_weight_1',
                                     init=is_training)
        self.Uo_n = self.__weights__(self.n_features, self.n_features, name='Output_node_weight_2', init=is_training)
        self.bo_n = self.__bias__(self.n_features, name='Output_node_bias', init=is_training)

        self.Wc_n = self.__weights__(self.n_features + self.graph_dim, self.n_features, name='Global_node_weight_1',
                                     init=is_training)
        self.Uc_n = self.__weights__(self.n_features, self.n_features, name='Global_node_weight_2', init=is_training)
        self.bc_n = self.__bias__(self.n_features, name='Global_node_bias', init=is_training)

        self.Wi_g = self.__weights__(self.n_features + self.n_event, self.graph_dim, name='Input_graph_weight_1',
                                     init=is_training)
        self.Ui_g = self.__weights__(self.graph_dim, self.graph_dim, name='Input_graph_weight_2', init=is_training)
        self.bi_g = self.__bias__(self.graph_dim, name='Input_graph_bias', init=is_training)

        self.Wf_g = self.__weights__(self.n_features + self.n_event, self.graph_dim, name='Forget_graph_weight_1',
                                     init=is_training)
        self.Uf_g = self.__weights__(self.graph_dim, self.graph_dim, name='Forget_graph_weight_2', init=is_training)
        self.bf_g = self.__bias__(self.graph_dim, name='Forget_graph_bias', init=is_training)

        self.Wo_g = self.__weights__(self.n_features + self.n_event, self.graph_dim, name='Output_graph_weight_1',
                                     init=is_training)
        self.Uo_g = self.__weights__(self.graph_dim, self.graph_dim, name='Output_graph_weight_2', init=is_training)
        self.bo_g = self.__bias__(self.graph_dim, name='Output_graph_bias', init=is_training)

        self.Wc_g = self.__weights__(self.n_features + self.n_event, self.graph_dim, name='Global_graph_weight_1',
                                     init=is_training)
        self.Uc_g = self.__weights__(self.graph_dim, self.graph_dim, name='Global_graph_weight_2', init=is_training)
        self.bc_g = self.__bias__(self.graph_dim, name='Global_graph_bias', init=is_training)

    def MessagePassing(self, send_nodes, receive_nodes, prev_node_embedding):
        # print(send_nodes.shape)
        # print(receive_nodes.shape)
        Min = torch.matmul(send_nodes.reshape(-1, self.n_nodes, 1), receive_nodes.reshape(-1, 1, self.n_nodes))
        Mout = torch.matmul(receive_nodes.reshape(-1, self.n_nodes, 1), send_nodes.reshape(-1, 1, self.n_nodes))
        # print('line 330', Min.shape, prev_node_embedding.shape)
        Hin_ = torch.matmul(Min, prev_node_embedding).view(-1, self.n_features)
        Hin = torch.tanh(torch.matmul(Hin_, self.Win) + self.bin)
        Hout_ = torch.matmul(Mout, prev_node_embedding).view(-1, self.n_features)
        Hout = torch.tanh(torch.matmul(Hout_, self.Wout) + self.bout)

        H = torch.max(torch.cat((Hin, Hout), -1).view(-1, self.n_nodes, self.n_features, 2), dim=-1).values
        # print('MessagePassing completed.')

        return H

    def TemporalModeling(self, middle_node_emb, prev_node_emb, prev_node_mem):
        # print(middle_node_emb.shape, prev_graph_emb.shape, self.Wa.shape)
        # print(torch.cat((torch.mean(middle_node_emb, dim=1), prev_graph_emb), dim=-1).shape)
        cur_a = torch.matmul(torch.mean(middle_node_emb, dim=1), self.Wa)

        x_ = cur_a
        g_ = torch.ones(self.n_nodes, self.graph_dim, device=self.device) * x_.unsqueeze(dim=1)
        h_input = torch.cat((g_, middle_node_emb), dim=-1)
        cur_node_emb, cur_node_mem = self.__LSTMUnit__(h_input, prev_node_emb, prev_node_mem, name='node_lstm')
        # print('node emb compute completed.')

        # h_ = cur_a * torch.mean(cur_node_emb, dim=1)
        # g_input = h_
        # print('graph_input', g_input.shape, prev_graph_emb.shape, prev_graph_mem.shape)
        # print('start to compute graph emb')
        # cur_graph_emb, cur_graph_mem = self.__LSTMUnit__(g_input, prev_graph_emb, prev_graph_mem, name='graph_lstm')

        return cur_node_emb, cur_node_mem, cur_a

    def __LSTMUnit__(self, input_x, prev_hidden, prev_memory, name):
        if name == 'node_lstm':
            input_x = input_x.view(-1, self.n_features + self.graph_dim)
            prev_hidden = prev_hidden.view(-1, self.n_features)
            prev_memory = prev_memory.view(-1, self.n_features)

            I_gate = torch.sigmoid(torch.matmul(input_x, self.Wi_n) + torch.matmul(prev_hidden, self.Ui_n) + self.bf_n)

            F_gate = torch.sigmoid(torch.matmul(input_x, self.Wf_n) + torch.matmul(prev_hidden, self.Uf_n) + self.bf_n)

            O_gate = torch.sigmoid(torch.matmul(input_x, self.Wo_n) + torch.matmul(prev_hidden, self.Uo_n) + self.bo_n)

            C_ = torch.tanh(torch.matmul(input_x, self.Wc_n) + torch.matmul(F_gate * prev_hidden, self.Uc_n) + self.bc_n)

            Ct = F_gate * prev_memory + I_gate * C_

            current_memory = Ct.view(-1, self.n_nodes, self.n_features)
            current_hidden = O_gate * torch.tanh(Ct)
            current_hidden.view(-1, self.n_nodes, self.n_features)

        elif name == 'graph_lstm':
            input_x = input_x.view(-1, self.n_features)
            prev_hidden = prev_hidden.view(-1, self.n_features)
            prev_memory = prev_memory.view(-1, self.n_features)
            # print('computing I_gate...', input_x.shape, prev_hidden.shape, prev_memory.shape)
            I_gate = torch.sigmoid(torch.matmul(input_x, self.Wi_g) + torch.matmul(prev_hidden, self.Ui_g) + self.bf_g)
            # print('I_gate compute completed.')
            F_gate = torch.sigmoid(torch.matmul(input_x, self.Wf_g) + torch.matmul(prev_hidden, self.Uf_g) + self.bf_g)
            # print('F_gate compute completed.')
            O_gate = torch.sigmoid(torch.matmul(input_x, self.Wo_g) + torch.matmul(prev_hidden, self.Uo_g) + self.bo_g)

            C_ = torch.tanh(torch.matmul(input_x, self.Wc_g) + torch.matmul(F_gate * prev_hidden, self.Uc_g) + self.bc_g)

            Ct = F_gate * prev_memory + I_gate * C_

            current_memory = Ct
            current_hidden = O_gate * torch.tanh(Ct)

        else:
            raise Exception('No valid lstm unit.')

        return current_hidden, current_memory

    def Cell(self, send_nodes, receive_nodes, prev_node_emb, prev_node_mem):

        H_nodes = self.MessagePassing(send_nodes, receive_nodes, prev_node_emb)

        cur_node_emb, cur_node_mem, cur_a = self.TemporalModeling(H_nodes, prev_node_emb, prev_node_mem)

        return cur_node_emb, cur_node_mem, cur_a

    def forward(self, send_nodes, receive_nodes, variable_size, initial_node_embedding=None):

        self.batch_size = variable_size
        # print(input.shape)
        # print(adj_mx.shape)
        # 将input和adj融合
        # print('line 408', send_nodes.shape, receive_nodes.shape)
        send_nodes = torch.from_numpy(send_nodes).float().to(self.device)
        receive_nodes = torch.from_numpy(receive_nodes).float().to(self.device)
        if not initial_node_embedding is None:
            # print('line 423', initial_node_embedding.shape)
            initial_node_embedding = initial_node_embedding.reshape(-1, self.n_nodes, self.n_features)
            cur_node_emb = torch.ones(self.batch_size, self.n_nodes, self.n_features, dtype=torch.float32, device=self.device) * initial_node_embedding
            cur_node_mem = torch.ones(self.batch_size, self.n_nodes, self.n_features, dtype=torch.float32, device=self.device) * initial_node_embedding
        else:
            cur_node_emb = torch.ones(self.batch_size, self.n_nodes, self.n_features, dtype=torch.float32, device=self.device)
            cur_node_mem = torch.ones(self.batch_size, self.n_nodes, self.n_features, dtype=torch.float32, device=self.device)

        # cur_graph_emb = torch.zeros(self.batch_size, self.graph_dim, dtype=torch.float32, device=self.device)
        # cur_graph_mem = torch.zeros(self.batch_size, self.graph_dim, dtype=torch.float32, device=self.device)

        # graph_embs, node_embs, attention_logits = [], [], []

        # for i in range(variable_size):
        node_emb, node_mem, cur_a = self.Cell(send_nodes, receive_nodes, cur_node_emb, cur_node_mem)
            # graph_embs.append(cur_graph_emb)
            # node_embs.append(cur_node_emb)
            # attention_logits.append(cur_a)
        # print(node_emb.shape)
        # print(cur_a.shape)
        # torch.Size([792000, 128])
        # torch.Size([3960, 1])
        # graph_embs = torch.Tensor(graph_embs).permute((1, 0, 2))
        # node_embs = torch.Tensor(graph_embs).permute((1, 0, 2, 3))
        # attention_logits = torch.reshape(torch.Tensor(attention_logits).permute((1, 0, 2)), [-1, variable_size - 1])

        return node_emb.detach(), cur_a.detach()



