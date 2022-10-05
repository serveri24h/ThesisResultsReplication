###########################################################################
#          _ _                                                            #
#     /\  ( | )   Aalto University, Master Thesis, Severi Vapalahti       #
#    /  \  / /                                                            #
#   / /\ \      Deep Learning Methodologies in Drug-Kinase Preadiction    #
#  / ____ \               severi.vapalahti@aalto.fi                       #
# /_/    \_\                                                              #
#                                                                         #
###########################################################################

import numpy as np

import constants as const
from context import Context
from models.models import load_model, save_model
from test_routine import test_routine

## LOCAL LIBRARIES
from data_tools import DTI_DATA_TOOL
from training_tools import GP_GRID_SEARCH, train_model
from models import ModifiedNet


def init_data_and_context():
    CTX = Context()
    DATA = DTI_DATA_TOOL(src_intermediate=const.INTERMEDIATE_DIR, src_data=const.DATA_DIR, device = CTX.device)
    DATA.load_and_prepare()
    CTX.update_data_parameters(DATA)
    return DATA,CTX

def init_grid_search(CTX,search_dict):
    CTX.init_grid(search_dict)
    return GP_GRID_SEARCH(CTX)

def train_best_model(DATA,GPGS,CTX):
    GPGS.run(DATA,CTX)
    _1,values,_2,epochs=GPGS.predict_minimum()
    CTX.update_model_parameters(values)
    NET = ModifiedNet(CTX).to(CTX.device)
    train_model(NET,DATA,CTX)
    return NET

def main_gpgs_routine(param_dict):
    DATA,CTX = init_data_and_context()
    GPGS = init_grid_search(CTX,search_dict=param_dict)
    NET = train_best_model(DATA,GPGS,CTX)
    test_routine(NET,DATA)

def test():
    DATA,CTX = init_data_and_context()
    NET, CTX = load_model('model.pickle')
    DATA.update_batch_size(25)
    CTX.free_parameters['learning_rate']=3.2
    train_model(NET,DATA,CTX)
    test_routine(NET,DATA)
    save_model(NET,CTX)
    train_model(NET,DATA,CTX)
    test_routine(NET,DATA)
    save_model(NET,CTX)

def test_saving_and_loading():
    DATA,CTX = init_data_and_context()
    CTX.free_parameters['epochs']=1
    NET = ModifiedNet(CTX).to(CTX.device)
    train_model(NET,DATA,CTX)
    save_model(NET,CTX)
    NET2,CTX2 = load_model()
    train_model(NET2,DATA,CTX2)

def check_variance():
    checks = 5
    errors = []
    DATA,CTX = init_data_and_context()
    for _ in range(checks):
        NET = ModifiedNet(CTX).to(CTX.device)
        _,e = train_model(NET,DATA,CTX)
        errors.append(np.min(e))
    S = np.var(np.array(errors))
    for er in errors:
        print(er)
    print(f"\nThe variance within {checks} samples was: VAR={S}\n")


def plot_gpgs(param_dict,plot_style='2d'):
    CTX = Context()
    GPGS = init_grid_search(CTX, search_dict=param_dict) 
    GPGS.add_data_from_file(const.RESULTS_DIR+"results.csv")
    _1,values,next_pred,epochs=GPGS.predict_minimum()
    print(_1,values,next_pred)
    if plot_style == '2d':
        GPGS.plot_2d_grid(CTX=CTX,next=next_pred)
    else:
        GPGS.plot_3d_grid(CTX,next_pred)


if __name__=='__main__':
    #plot_gpgs(const.SEARCH_HS)
    #plot_gpgs(const.SEARCH_EMBED)
    #check_variance()
    #main_gpgs_routine(const.SEARCH_EMBED)

    #main_gpgs_routine(const.SEARCH_TRAINING_PARAMS)
    #plot_gpgs(const.SEARCH_TRAINING_PARAMS,plot_style='3d')
    test()
