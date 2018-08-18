"""
Created on Oct 05, 2017

@author: Siyuan Qi

Description of the file.

"""

import torch


class ReadoutFunction(torch.nn.Module):
    def __init__(self, readout_def, args):
        super(ReadoutFunction, self).__init__()
        self.r_definition = ''
        self.r_function = None
        self.args = {}
        self.learn_args = torch.nn.ParameterList([])
        self.learn_modules = torch.nn.ModuleList([])
        self.__set_readout(readout_def, args)

    def forward(self, h_v):
        return self.r_function(h_v)

    # Set a readout function
    def __set_readout(self, readout_def, args):
        self.r_definition = readout_def.lower()
        self.args = args

        self.r_function = {
            'fc':           self.r_fc,
            'fc_soft_max':      self.r_fc_soft_max,
            'fc_sig':           self.r_fc_sigmoid,
        }.get(self.r_definition, None)

        if self.r_function is None:
            print('WARNING!: Readout Function has not been set correctly\n\tIncorrect definition ' + readout_def)
            quit()

        init_parameters = {
            'fc':           self.init_fc,
            'fc_soft_max':      self.init_fc_soft_max,
            'fc_sig':           self.init_fc_sigmoid,
        }.get(self.r_definition, lambda x: (torch.nn.ParameterList([]), torch.nn.ModuleList([]), {}))

        init_parameters()

    # Get the name of the used readout function
    def get_definition(self):
        return self.r_definition

    def get_args(self):
        return self.args

    # Definition of readout functions
    # Fully connected layers with softmax output
    def r_fc_soft_max(self, hidden_state):
        last_layer_output = hidden_state
        for layer in self.learn_modules:
            last_layer_output = layer(last_layer_output)
        return last_layer_output

    def init_fc_soft_max(self):
        input_size = self.args['readout_input_size']
        output_classes = self.args['output_classes']

        self.learn_modules.append(torch.nn.Linear(input_size, output_classes))
        self.learn_modules.append(torch.nn.Softmax())

    # Fully connected layers with softmax output
    def r_fc_sigmoid(self, hidden_state):
        last_layer_output = hidden_state
        for layer in self.learn_modules:
            last_layer_output = layer(last_layer_output)
        return last_layer_output

    def init_fc_sigmoid(self):
        input_size = self.args['readout_input_size']
        output_classes = self.args['output_classes']

        self.learn_modules.append(torch.nn.Linear(input_size, input_size))
        self.learn_modules.append(torch.nn.Linear(input_size, output_classes))
        # self.learn_modules.append(torch.nn.Sigmoid())

    # Fully connected layers
    def r_fc(self, hidden_state):
        last_layer_output = hidden_state
        for layer in self.learn_modules:
            last_layer_output = layer(last_layer_output)
        return last_layer_output

    def init_fc(self):
        input_size = self.args['readout_input_size']
        output_classes = self.args['output_classes']

        self.learn_modules.append(torch.nn.Linear(input_size, input_size))
        self.learn_modules.append(torch.nn.ReLU())
        # self.learn_modules.append(torch.nn.Dropout())
        # self.learn_modules.append(torch.nn.BatchNorm1d(input_size))
        self.learn_modules.append(torch.nn.Linear(input_size, output_classes))


def main():
    pass


if __name__ == '__main__':
    main()
