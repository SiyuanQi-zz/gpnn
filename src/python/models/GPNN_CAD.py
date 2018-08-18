"""
Created on Oct 07, 2017

@author: Siyuan Qi

Description of the file.

"""

import os

import torch
import torch.nn
import torch.autograd

import units


class GPNN_CAD(torch.nn.Module):
    def __init__(self, model_args):
        super(GPNN_CAD, self).__init__()

        self.link_fun = units.LinkFunction('GraphConvLSTM', model_args)
        self.message_fun = units.MessageFunction('linear_concat', model_args)

        self.update_funs = torch.nn.ModuleList([])
        self.update_funs.append(units.UpdateFunction('gru', model_args))
        self.update_funs.append(units.UpdateFunction('gru', model_args))

        self.subactivity_classes = model_args['subactivity_classes']
        self.affordance_classes = model_args['affordance_classes']
        self.readout_funs = torch.nn.ModuleList([])
        self.readout_funs.append(units.ReadoutFunction('fc_soft_max', {'readout_input_size': model_args['node_feature_size'], 'output_classes': self.subactivity_classes}))
        self.readout_funs.append(units.ReadoutFunction('fc_soft_max', {'readout_input_size': model_args['node_feature_size'], 'output_classes': self.affordance_classes}))

        self.propagate_layers = model_args['propagate_layers']

        self._load_link_fun(model_args)

    def forward(self, edge_features, node_features, adj_mat, node_labels, args):
        # pred_adj_mat = self.link_fun(edge_features)
        # pred_adj_mat = torch.autograd.Variable(torch.ones(adj_mat.size())).cuda()  # Test constant graph
        pred_node_labels = torch.autograd.Variable(torch.zeros(node_labels.size()))
        if args.cuda:
            pred_node_labels = pred_node_labels.cuda()
        hidden_node_states = [node_features.clone() for passing_round in range(self.propagate_layers+1)]
        hidden_edge_states = [edge_features.clone() for passing_round in range(self.propagate_layers+1)]

        # Belief propagation
        for passing_round in range(self.propagate_layers):
            pred_adj_mat = self.link_fun(hidden_edge_states[passing_round])
            # if passing_round == 0:
            #     pred_adj_mat = torch.autograd.Variable(torch.ones(adj_mat.size())).cuda()  # Test constant graph
            #     pred_adj_mat = self.link_fun(hidden_edge_states[passing_round]) # Without iterative parsing

            # Loop through nodes
            for i_node in range(node_features.size()[2]):
                h_v = hidden_node_states[passing_round][:, :, i_node]
                h_w = hidden_node_states[passing_round]
                e_vw = edge_features[:, :, i_node, :]
                m_v = self.message_fun(h_v, h_w, e_vw, args)

                # Sum up messages from different nodes according to weights
                m_v = pred_adj_mat[:, i_node, :].unsqueeze(1).expand_as(m_v) * m_v
                hidden_edge_states[passing_round+1][:, :, :, i_node] = m_v
                m_v = torch.sum(m_v, 2)
                if i_node == 0:
                    h_v = self.update_funs[0](h_v[None].contiguous(), m_v[None])
                else:
                    h_v = self.update_funs[1](h_v[None].contiguous(), m_v[None])

                # Readout at the final round of message passing
                if passing_round == self.propagate_layers-1:
                    if i_node == 0:
                        pred_node_labels[:, i_node, :self.subactivity_classes] = self.readout_funs[0](h_v.squeeze(0))
                    else:
                        pred_node_labels[:, i_node, :] = self.readout_funs[1](h_v.squeeze(0))

        return pred_adj_mat, pred_node_labels

    def _load_link_fun(self, model_args):
        if not os.path.exists(model_args['model_path']):
            os.makedirs(model_args['model_path'])
        best_model_file = os.path.join(model_args['model_path'], '..', 'graph', 'model_best.pth')
        if os.path.isfile(best_model_file):
            checkpoint = torch.load(best_model_file)
            self.link_fun.load_state_dict(checkpoint['state_dict'])

    def _dump_link_fun(self, model_args):
        if not os.path.exists(model_args['model_path']):
            os.makedirs(model_args['model_path'])
        if not os.path.exists(os.path.join(model_args['model_path'], '..', 'graph')):
            os.makedirs(os.path.join(model_args['model_path'], '..', 'graph'))
        best_model_file = os.path.join(model_args['model_path'], '..', 'graph', 'model_best.pth')
        torch.save({'state_dict': self.link_fun.state_dict()}, best_model_file)


def main():
    pass


if __name__ == '__main__':
    main()
