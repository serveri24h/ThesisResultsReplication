import os
import sys
import numpy as np

# IF NOT ROOT: SET WD TO ROOT
if __name__=='__main__':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# IMPORT LOCAL PACKAGES
from data_tools import DTI_DATA_TOOL
from training_tools import GP_GRID_SEARCH, hyperparameter_optimizer
import constants as const

    
    ########################################
    ############ DIAGNOSTICS ###############
    ########################################

def GPGS_diagnostics(GPGS):
    try:
        assert len(GPGS.Xp) == GPGS.l
        assert len(GPGS.Xp[0]) == GPGS.dims
    except:
        print("Dimensions did not match")
        return False
    try:
        assert GPGS.mapping[0] == [p[0] for p in GPGS.params] 
    except:
        print("\nMapping failed...")
        print( [p[0] for p in GPGS.params] )
        return False
    try:
        GPGS.get_next_x()
    except:
        print("Failed to obtain the prior distribution")
        return False
    try:
        set_1 = np.array([0,0,0])
        set_2 = np.array([[1,1,1],[0.5,0.5,0.5]] )
        GPGS.add_new_data(set_1,3)
        GPGS.add_new_data(set_2,[3,1])
    except:
        print("Failed adding new data")
        return False
    GPGS.update_posterior()
    try:
        GPGS.update_posterior()
    except:
        print("Failed Udating the posterior")
        return False
    try:
        GPGS.get_next_x()
    except:
        print("Failed to obtain next prediction")
        return False
    try:
        y_new=3
        GPGS.step(y_new)
    except:
        print("Step function failed")
        return False
    return True

    ########################################
    ############ DIAGNOSTICS ###############
    ########################################
        
def HPO_diagnostics(HPO):
    # RUN TESTS
    first_step = False
    nth_step = False
    full_model = False
    loading_file = True
    misc = False

    if first_step:
        ## FIRST STEP
        try:
            assert HPO.GPGS.Xp.shape == (HPO.GPGS.l,HPO.GPGS.dims)
            assert HPO.GPGS.Xp_norm.shape == (HPO.GPGS.l,HPO.GPGS.dims)
        except:
            print("Something wrong with sahpes")
            return False

        print("Testing first iteration")
        try:
            # Initiate parameter space
            lr_0 = 5
            bs_0 = 50
            hs_0 = 25
            X = [lr_0,bs_0,hs_0]
            params0 = HPO.get_params_dict(X)
        except Exception as e:
            print("something went wrong with initiating parameters with error:\n", e)
            return False
        try:
            y = HPO.evaluate_parameters(params0,epochs=1)
            print('\n')
        except Exception as e:
            print("Something went wrong with the training loop or parameter evuluation:\n", e)
            return False
    
    if nth_step:
        ## Nth STEP
        print("Testing second iteration with set guessed parameters")
        try:
            next_values = HPO.GPGS.step(y,np.array(X))
        except:
            print("Something went wrong with the updating the GPGS\n")
            return False
        try:
            params = HPO.get_params_dict(next_values)
            y = HPO.evaluate_parameters(params,epochs=1)
            print('\n')
        except:
            print("Something went wrong with parameter evuluation with updated parameters\n")
            return False
    
    if loading_file:
        try:
            HPO.GPGS.add_data_from_file(const.DUMMY_RESULTS_DIR+'gpgs_grid.csv')
        except:
            print("Loading Data failed")
            return False

        next_values = HPO.GPGS.next_parameters()
        try:
            next_values = HPO.GPGS.next_parameters()
        except:
            print("Lälläslieru")
            return False
    
    if full_model:
        # FULL Module
        print("Testing the main-function \"run()\"")
        try:
            HPO.run(next_values,N_search=2)
        except Exception as e: 
            print("Failed to run the main function with error:", e)
            return False
    
    if misc:
        try:
            if len(HPO.GPGS.DATA_X)==0:
                HPO.load_data('results/saved_data/saved_data_10.csv')
        except:
            print("Something when loading the data...")
            return False
        try:
            HPO.predict_minimum()
        except:
            print("Something failed when trying to obtain minimum")
            return False
            
    return True
    
def run_tests():
    # Make test variables
    precision1 = 6
    par1 = np.linspace(0,1,precision1)
    par2 = np.linspace(0,1,precision1)
    par3 = np.linspace(0,1,precision1)
    parameters1 = [par1,par2,par3]

    # Hypers
    acquasition_rate = 2

    # Testing Gaussian GRID SEARCH
    test_gpgs = False
    if test_gpgs:
        print("\nRunning tests for gaussiang grid search method..\n")
        # Initialize the class
        try:
            GPGS1 = GP_GRID_SEARCH(parameters1,acquasition_rate=acquasition_rate)
        except Exception as e:
            print("Initialising the class failed with following error\n", e)
            return
        # Run diagnostics
        TEST1 = GPGS_diagnostics(GPGS1)

    # Testing hyperparameter optimizer
    test_hpo = True
    if test_hpo:
        # DATA
        print("\nRunning tests for the hyperparameter optimizer\n")
        try:
            TESTING_DATA = DTI_DATA_TOOL(const.DATA_DIR,prints=False)
            TESTING_DATA.read_training_data(const.TRAINING_FILE,index_range=(1,40000))
            TESTING_DATA.prepare_training_data()
        except Exception as e:
            print("Testing failed when obtaining test data with error:\n",e)
        
        # GPGS
        try:
            precision2 = 5
            grid_lr = np.linspace(6,10,precision2)
            grid_bs = np.linspace(10,100,precision2,dtype=int) 
            grid_hs = np.linspace(10,30,precision2,dtype=int)
            parameters2 = [grid_lr,grid_bs,grid_hs]
            GPGS2 = GP_GRID_SEARCH(parameters2,acquasition_rate=acquasition_rate,prints=False)
        except:
            print("Testing failed when initializing the GPGS..\n")

        # Initialize the class
        try:
            HPO = hyperparameter_optimizer(TESTING_DATA,GPGS2,outdir=const.DUMMY_RESULTS_DIR)
        except:
            print("Initialising the class failed...\n")

        # Run diagnostics
        TEST2 = HPO_diagnostics(HPO)

    if TEST1 and TEST2:
        print("\nALL THE TESTS PASSED!! - THIS IS A WONDERFUL DAY\n")
    else:
        print("\nSOMETHING FAILED - THERE IS A BUG IN THE CODE\n")


if __name__=="__main__":
    run_tests()