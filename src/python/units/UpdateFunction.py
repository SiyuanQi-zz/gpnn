"""
Created on Oct 05, 2017

@author: Siyuan Qi

Description of the file.

"""

import torch


class UpdateFunction(torch.nn.Module):
    def __init__(self, update_def, args):
        super(UpdateFunction, self).__init__()
        self.u_definition = ''
        self.u_function = None
        self.args = {}
        self.learn_args = torch.nn.ParameterList([])
        self.learn_modules = torch.nn.ModuleList([])
        self.__set_update(update_def, args)

    def forward(self, h_v, m_v, args=None):
        return self.u_function(h_v, m_v, args)

    # Set an update function
    def __set_update(self, update_def, args):
        self.u_definition = update_def.lower()
        self.args = args

        self.u_function = {
            'gru':     self.u_gru,
        }.get(self.u_definition, None)

        if self.u_function is None:
            print('WARNING!: Update Function has not been set correctly\n\tIncorrect definition ' + update_def)

        init_parameters = {
            'gru':     self.init_gru,
        }.get(self.u_definition, lambda x: (torch.nn.ParameterList([]), torch.nn.ModuleList([]), {}))

        init_parameters()

    # Get the name of the used update function
    def get_definition(self):
        return self.u_definition

    def get_args(self):
        return self.args

    # Definition of update functions
    # GRU: node state as hidden state, message as input
    def u_gru(self, h_v, m_v, args):
        output, h = self.learn_modules[0](m_v, h_v)
        return h

    def init_gru(self):
        node_feature_size = self.args['node_feature_size']
        message_size = self.args['message_size']
        num_layers = self.args.get('update_hidden_layers', 1)
        bias = self.args.get('update_bias', False)
        dropout = self.args.get('update_dropout', False)
        self.learn_modules.append(torch.nn.GRU(message_size, node_feature_size, num_layers=num_layers, bias=bias, dropout=dropout))


def main():
    pass


if __name__ == '__main__':
    main()
