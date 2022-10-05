import torch ##########################################################
#######################################################################
#####                                                             #####
####    OOOOO     OOOOOO    OO    OO    OOOOO  OOOOOOOOO  OOOOO    ####
###    OO   OO   OO    OO   OOO   OO   OO   OO    OO     OO   OO    ###
##    OO        OO      OO  OOOO  OO   OO         OO     OO          ##
##    OO        OO      OO  OO OO OO    OOOOO     OO      OOOOO      ##
##    OO        OO      OO  OO  OOOO        OO    OO          OO     ##
###    OO   OO   OO    OO   OO   OOO   OO   OO    OO     OO   OO    ###
####    OOOOO     OOOOOO    OO    OO    OOOOO     OO      OOOOO    ####
#####                                                             #####
#######################################################################
#######################################################################
###                                                                 ###
##           ALL THE CONSTANT VARIABLES (AND FUNCTIONS)              ##
##                     ARE DEFINED IN THIS FILE                      ##
###                                                                 ###
#######################################################################
#######################################################################

# PADDING VALUE
PADDING_VALUE = 0

# NUMPY SEED
NUMPY_SEED = 2525

#### DIRECTORIES AND FILES

# Directories HOME
DATA_DIR = "project_folder/DATA/RAW/"
INTERMEDIATE_DIR = "project_folder/DATA/SAVED/"
RESULTS_DIR = "project_folder/results/"
DUMMY_RESULTS_DIR = ""
SAVED_MODELS_DIR = "project_folder/SAVED_MODELS/"

# Main data files
TRAINING_FILE = 'DtcDrugTargetInteractions.csv'
TEST_FILE = '41467_2021_23165_MOESM4_ESM.xlsx'
KINASE_FILE = 'Kinases_KinMap.xlsx' 
FASTA_FILE = 'uniprot-filtered-sequence.fasta'

# Intermediate files
ID_SMILES_FILE = 'id_to_smiles.csv'
ID_FPS_FILE = 'id_to_fps.csv'
ID_AMINO_FILE = 'id_to_amino.csv'
INTERMEDIATE_TRAIN_FILE = 'intermediate_train_file'
INTERMEDIATE_TEST_FILE1 = 'intermediate_test_file_1'
INTERMEDIATE_TEST_FILE2 = 'intermediate_test_file_2'

# DEFAULT MODEL VARIABLES
MAX_SIZE_D = 250
MAX_SIZE_P = 2600
EMBED_SIZE_DRUG = 262
EMBED_SIZE_PROTEIN = 400
N_CONV_CHANNELS = 64
OUTPUT_SIZE = 256
FP_SIZE = 1024
HIDDEN_LAYERS = 1
KERNEL_SIZE = 4
HIDDEN_SIZE_DRUG = 50
HIDDEN_SIZE_PROTEIN = 180

# DEFAULT TRAINING VARIABLES
LEARNING_RATE = 2.857142857142857 
MOMENTUM_FACTOR = 0.5
BATCH_SIZE = 20
EPOCHS = 12
MAX_EPOCHS = 10
CRITERION = torch.nn.MSELoss()
N_SEARCH = 6

# DEFAULT GRID SEARCH VARIABLES
GPGS_PRECISION = 20
GPGS_ALPHA = 0.5
GPGS_SCALE = 0.4
GPGS_SIGMA2 = 0.025
GPGS_PRIOR = 0.9
GPGS_ACQUASITION_RATE = 2
GPGS_NORMALIZED = True
MAX_LOSS = 3

# GRID SEARCH FOR HIDDEN SIZE
SEARCH_HS = {   'alpha': 0.3,
                'scale': 0.3,
                'sigma2': 0.025,
                'prior': 0.9,
                'acq_rate': 2,
                'param_names': ['h_size_p','h_size_d'],
                'param_edges': [[50,300],[50,300]],
                'param_labels': ['Hidden Size - Kinase', 'Hidden Size - Drug']
}

# GRID SEARCH FOR EMBEDDING SIZE
SEARCH_EMBED = {'alpha': 0.3,
                'scale': 0.3,
                'sigma2': 0.025,
                'prior': 0.9,
                'acq_rate': 2,
                'param_names': ['embed_size_p','embed_size_d'],
                'param_edges': [[25,400],[25,400]],
                'param_labels': ['Embedding Size - Kinase', 'Embedding Size - Drug']
}

# GRID SEARCH FOR TRAINING HYPERPARAMETERS
SEARCH_TRAININGPARAMS_3D = {   'alpha': 1.5,
                'scale': 0.4,
                'sigma2': 0.025,
                'prior': 1.5,
                'acq_rate': 1,
                'param_names': ['batch_size','learning_rate','momentum_factor'],
                'param_edges': [[20,200],[2,5],[0,1]],
                'param_labels': ['Batch Size', 'Learning Rate', 'Momentum Factor'],
                'precision': 15
}

# GRID SEARCH FOR TRAINING HYPERPARAMETERS
SEARCH_TRAININGPARAMS = {   'alpha': 1.5,
                            'scale': 0.25,
                            'sigma2': 0.25,
                            'prior': 1.5,
                            'acq_rate': 1,
                            'param_names': ['learning_rate','momentum_factor'],
                            'param_edges': [[2,5],[0,1]],
                            'param_labels': ['Learning Rate', 'Momentum Factor'],
                            'precision': 15
}

