import torch

def get_y_pred_linear_meas(X_estimated_pred_test, Pk_estimated_pred_test, Cw_test, ssm_model_test):
    
    Y_estimated_pred_test = torch.einsum('ij,nkj->nki', torch.from_numpy(ssm_model_test.H).type(torch.FloatTensor), X_estimated_pred_test)
    Py_estimated_pred_test = torch.einsum('nkij,lj->nkil',
        torch.einsum('ij,nkjj->nkij', 
                     torch.from_numpy(ssm_model_test.H).type(torch.FloatTensor), Pk_estimated_pred_test), 
                     torch.from_numpy(ssm_model_test.H).type(torch.FloatTensor)
                    ) + torch.repeat_interleave(Cw_test.unsqueeze(1), Pk_estimated_pred_test.shape[1], 1).type(torch.FloatTensor) 
    
    return Y_estimated_pred_test, Py_estimated_pred_test