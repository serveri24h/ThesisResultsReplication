import torch
import sys
import os

# IF NOT ROOT: SET WD TO ROOT
if __name__=='__main__':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import constants as const
from models.models import GregoryNet,ModifiedNet,CNN64,RCNN,RCN
from data_tools.data_tools import DTI_DATA_TOOL


def run_tests():
    batch_size = 4
    max_seq_length = 100
    alphabet_size = 20
    hidden_layers = 2
    hidden_size = 200
    N_conv_channels = 64
    output_size = 256
    fp_size = 1024
    n_family = 10
    
    X = torch.Tensor(batch_size, max_seq_length, alphabet_size)
    print("\nTESTING PYTHON CODE FOR POSSIBLE BUGS:")
    #try:
    #    CNN64_dummy = CNN64(max_seq_length,alphabet_size,N_conv_channels,output_size)
    #    assert CNN64_dummy(X).shape == (batch_size,output_size), "Wrong output shape with CNN64!!"
    #except Exception as e:
    #    print(e)
    #    return 0
    try:
        RCNN_dummy = RCNN(alphabeth_size=alphabet_size,N_conv_channels=N_conv_channels,hidden_layers=hidden_layers,hidden_size=hidden_size,output_size=output_size,device=torch.device("cpu"))
        assert RCNN_dummy(X).shape == (batch_size,output_size), "Wrong output shape with RCNN!!"
    except Exception as e:
        print(e)
        return 0
    return 1

def pad_sequences(X,max_seq_length,pad_value):
    N = len(X)
    X_0 = torch.ones(N,max_seq_length,dtype=torch.long)*pad_value
    for i in range(N):
        X_0[i,0:list(X[i].size())[0]] = X[i] 
    print(f"RUNNING WITH:\n\n {X_0}\n")
    return X_0

def test_embedding():
    # Set padding value
    PADDING_VALUE = 0

    # Globals
    batch_size = 10
    embed_size = 30
    max_seq_length = 7
    alphabet_size = 20
    hidden_layers = 2
    hidden_size = 40
    N_conv_channels = 64
    output_size = 256
    fp_size = 1024
    n_family = 10

    #X = torch.randint(1,alphabet_size+1,(max_seq_length,batch_size))

    X = [torch.randint(1,alphabet_size+1,(l,)) for l in torch.randint(1,max_seq_length+1,(batch_size,1)) ]
    #X = pad_sequences(X,max_seq_length,PADDING_VALUE)
    NET = RCN(alphabeth_size=alphabet_size, N_conv_channels=N_conv_channels,hidden_size=hidden_size,hidden_layers=hidden_layers,embed_size=embed_size,output_size=output_size,device=torch.device("cpu"))
    print(NET(X))

def test_embedding2():
    # Set padding value
    device=torch.device("cpu")

    DATA = DTI_DATA_TOOL(src_intermediate=const.INTERMEDIATE_DIR, src_data=const.DATA_DIR, device = device)
    DATA.load_and_prepare()
    X = DATA.random_batch()[0]


    # Globals
    batch_size = 10
    max_seq_length = max([len(x) for x in X])
    alphabet_size = len(DATA.drug_token_keys)
    hidden_layers = 2
    hidden_size = 200
    N_conv_channels = 64
    output_size = 256
    fp_size = 1024
    n_family = 10


    NET = RCN(alphabeth_size=alphabet_size, N_conv_channels=N_conv_channels,hidden_size=hidden_size,hidden_layers=hidden_layers,output_size=output_size,device=device)
    print(NET(X))

def test_embedding3():
    device=torch.device("cpu")
    DATA = DTI_DATA_TOOL(src_intermediate=const.INTERMEDIATE_DIR, src_data=const.DATA_DIR, device = device)
    DATA.load_and_prepare()
    # Set padding value
    d_max,d_alpha,p_max,p_alpha,n_family = DATA.get_dimensions()
    params = {'model':"Modified",
        'learning_rate':0.1,
        'momentum_factor':0.5,
        'batch_size': 10, 
        'p_alpha':p_alpha,
        'd_alpha':d_alpha,
        'n_family':n_family,
        'h_size':100,
        'h_layers':2,
        'n_channel':64,
        'output_size':256,
        'fp_size':1024,
        'kernel_size':4 }
    
    X = DATA.random_batch(batch_size=params['batch_size'])
    NET = ModifiedNet(params,device=device)
    print(NET(X))

def test_RCN(ctx):
    x = torch.randint( 1,10,(5,8) )
    DATA = DTI_DATA_TOOL(src_intermediate=const.INTERMEDIATE_DIR, src_data=const.DATA_DIR, device = device)
    DATA.load_and_prepare()

def run_test_pipeline(ctx):
    test_embedding()


class Context():
    def __init__(self):
        self.device = torch.device("cpu")


if __name__ == "__main__":
    ctx = Context()
    run_test_pipeline(ctx)
    #test_embedding3()
    #if run_tests():
    #    print("All tests succeeded.\n")
    #else:
    #    print("Something Failed...\n")
