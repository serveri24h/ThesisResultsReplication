import torch
import constants as const

def stop_training_standard(errors,CTX):
    if len(errors)>CTX.free_parameters['epochs']:
        return True
    return False

def stop_training_error_based(errors,CTX):
    min_e = min(errors)
    if len(errors)==const.MAX_EPOCHS+1 or (errors[-1]>min_e and errors[-2]>min_e):
        print(f"\n Conditions for stopping the training have been met after {len(errors)-1} epochs.\n")
        return True
    return False

def compute_test_error(NET,DATA,CTX):
    # SET EVALUATION MODE
    NET.eval()
    
    # PREPARE TEST DATA
    X = DATA.validation_batch()
    
    # PREDICTION
    with torch.no_grad():
        pred = NET.forward(X[0:4])
    
    # ERROR
    true = X[4]
    test_error = CTX.criterion(pred,true).item()
    
    # SET TRAINING MODE
    NET.train()
    
    return test_error

def training_loop(NET,DATA,optimizer,CTX,stop_condition):
    if CTX.prints: print("Starting a training loop for {} epoch(s)".format(CTX.free_parameters['epochs']))
    # SETUP
    Es_train = []
    Es_test = []
    
    # COMPUTE ERROR BEFORE ANY TRAINING
    test_error = compute_test_error(NET,DATA,CTX)
    Es_test.append(test_error)
    if CTX.prints: print("Test error for before training: {:2f}".format(test_error))
    
    # TRAINING LOOP
    #while stop_training(Es_test)
    while not stop_condition(Es_test,CTX):
        epc =  len(Es_test)
        if CTX.prints: print("EPOCH:", epc)
            
        # START EPOCH
        DATA.start_epoch()
        NET.train()
        while DATA.epoch_status():
            # RESET GRADIENT
            NET.zero_grad()
            
            # PREPARE TRAINING DATA
            X = DATA.training_batch()
            
            # FORWARD
            pred = NET.forward(X[0:4])

            # ERROR/LOSS
            true = X[4]
            loss = CTX.criterion(pred,true)
            if torch.isnan(loss) or torch.isinf(loss):
                return [3],[3] # Escape from failed model
            if CTX.prints: print("PROCESS:  {0:.4f} %  ------  ERROR: {1:.2f}".format( DATA.get_progress(),loss.item() ) )
            Es_train.append(loss.item())
            
            # BACKPROPAGATION
            loss.backward()
            optimizer.step()
        
        # EVALUATION
        test_error = compute_test_error(NET,DATA,CTX)
        Es_test.append(test_error)
        if CTX.prints: print("Test error for EPOCH: {0:} = {1:2f}".format(epc,test_error))         
    return Es_train,Es_test