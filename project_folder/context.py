import numpy as np

import constants as const
import helpers as h

class Context():
    def __init__(self):
        np.random.seed(const.NUMPY_SEED)

        self.device,_ = h.check_device()

        self.model_name = 'Modified'

        self.gpgs_parameters = {    'grid': None,
                                    'param_names': None,
                                    'param_labels': None,
                                    'param_edges': None,
                                    'precision': const.GPGS_PRECISION,
                                    'n_search': const.N_SEARCH,
                                    'alpha': const.GPGS_ALPHA,
                                    'scale': const.GPGS_SCALE,
                                    'sigma2': const.GPGS_SIGMA2,
                                    'prior': const.GPGS_PRIOR,
                                    'acq_rate': const.GPGS_ACQUASITION_RATE,
                                    'normalized': const.GPGS_NORMALIZED
        }

        self.criterion = const.CRITERION

        self.free_parameters = {    'p_max':const.MAX_SIZE_P,
                                    'd_max':const.MAX_SIZE_D,
                                    'h_size_p':const.HIDDEN_SIZE_PROTEIN,
                                    'h_size_d':const.HIDDEN_SIZE_DRUG,
                                    'h_layers':const.HIDDEN_LAYERS,
                                    'n_channel':const.N_CONV_CHANNELS,
                                    'output_size':const.OUTPUT_SIZE,
                                    'fp_size':const.FP_SIZE,
                                    'kernel_size':const.KERNEL_SIZE,
                                    'embed_size_p':const.EMBED_SIZE_PROTEIN,
                                    'embed_size_d':const.EMBED_SIZE_PROTEIN,
                                    'learning_rate':const.LEARNING_RATE,
                                    'momentum_factor':const.MOMENTUM_FACTOR,
                                    'batch_size': const.BATCH_SIZE,
                                    'epochs': const.EPOCHS}
        
        self.data_parameters = {    'p_alpha':None,
                                    'd_alpha':None,
                                    'n_family':None
        }

        self.prints = True
        self.save = True
        self.load = True
        
    def update_data_parameters(self,DATA):
        d_max,d_alpha,p_max,p_alpha,n_family = DATA.get_dimensions()
        self.data_parameters['p_alpha']=p_alpha
        self.data_parameters['d_alpha']=d_alpha
        self.data_parameters['n_family']=n_family
    
    def update_model_parameters(self,values):
        for i,name in enumerate(self.gpgs_parameters['param_names']):
            self.free_parameters[name] = values[i]

    def init_grid(self,search_dict):
        for k,v in search_dict.items():
            self.gpgs_parameters[k]=v
        self.gpgs_parameters['grid'] = [np.linspace(e[0],e[1],self.gpgs_parameters['precision']) for e in search_dict['param_edges']]


        

        
