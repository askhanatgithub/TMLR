from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MSE=nn.MSELoss(reduction='mean')
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len
    

def get_dataloader(x_data,y_data,batch_size):

    x_train_sr=x_data[x_data['treatment']==0]
    y_train_sr=y_data[x_data['treatment']==0]
    x_train_tr=x_data[x_data['treatment']==1]
    y_train_tr=y_data[x_data['treatment']==1]


    train_data_sr = Data(np.array(x_train_sr), np.array(y_train_sr))
    train_dataloader_sr = DataLoader(dataset=train_data_sr, batch_size=batch_size)

    train_data_tr = Data(np.array(x_train_tr), np.array(y_train_tr))
    train_dataloader_tr = DataLoader(dataset=train_data_tr, batch_size=batch_size)


    return train_dataloader_sr, train_dataloader_tr

def get_data(data_type,file_num):
   

    if(data_type=='train'):
        data=pd.read_csv(f"Dataset/IHDP_a/ihdp_npci_train_{file_num}.csv")
    else:
        data = pd.read_csv(f"Dataset/IHDP_a/ihdp_npci_test_{file_num}.csv")

    x_data=pd.concat([data.iloc[:,0], data.iloc[:, 1:30]], axis = 1)
    x_data.iloc[:,18]=np.where(x_data.iloc[:,18]==2,1,0)
    #x_data_a=x_data.iloc[:,0:5]
    #x_data_b=x_data.iloc[:,5:30]
    #scaler.fit(x_data_b)
    #scaled_b = pd.DataFrame(scaler.fit_transform(x_data_b))
    #x_data=data.iloc[:, 5:30]
    #x_data_trans=pd.concat([x_data_a,scaled_b],axis=1)
    y_data_trans=data.iloc[:, 1]
    #y_data_trans=pd.DataFrame(scaler.fit_transform(data.iloc[:, 1].to_numpy().reshape(-1, 1)))
    #y_data_trans=y_data_trans.to_numpy().reshape(-1,)
    return x_data,y_data_trans
def cal_pehe(data,y,Regressor,Encoder,mask_gamma,mask_delta,mask_upsilon,fstart,fend,sstart,send,tstart,tend):
    #data,y=get_data('test',i)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data=data.to_numpy()
    data=torch.from_numpy(data.astype(np.float32)).to(device)
    #Replace 30 with :

    phi, phi_mean, phi_var=Encoder(data[:,5:])   
    phi_gamma=phi[:,fstart:fend]
    phi_delta=phi[:,sstart:send]
    phi_upsilon=phi[:,tstart:tend]
    #phi_irr=phi[:,frstart:frend]

    del_ups=torch.cat((phi_delta*mask_delta, phi_upsilon*mask_upsilon), 1)
    
   
    #del_ome=torch.cat((phi_delta,phi_irr), 1)
    #ups_ome=torch.cat((phi_upsilon,phi_irr), 1)
    
    # change to del_ups for true evaluation
    concat_pred=Regressor(del_ups)
    
    
    
    t=data[:,0]
   
    
    
    
    predicted_y=torch.where(t.squeeze() == 0, concat_pred[:,0], concat_pred[:,1])
    
    #print(y)
    #print('mae test',np.mean(np.abs(predicted_y.detach().numpy()-y)))
    
    #print('mae train',np.mean(np.abs(predicted_yt.detach().numpy()-ty)))
    
    #concat_num=scaler.inverse_transform(pd.DataFrame(concat.detach().numpy() ))
    #concat_pred=torch.from_numpy(concat_num.astype(np.float32))
    #dont forget to rescale the outcome before estimation!
    #y0_pred = data['y_scaler'].inverse_transform(concat_pred[:, 0].reshape(-1, 1))
    #y1_pred = data['y_scaler'].inverse_transform(concat_pred[:, 1].reshape(-1, 1))

  

    cate_pred=concat_pred[:,1]-concat_pred[:,0]
    
    cate_true=data[:,4]-data[:,3]#Hill's noiseless true values

    #cate_true=((pd.DataFrame(data[:,4].cpu().detach().numpy()))/
    #-(pd.DataFrame(data[:,3].cpu().detach().numpy())) )#Hill's noiseless true values

    #cate_true=((data_ori.iloc[:,1])-(data_ori.iloc[:,0])).to_numpy().reshape(-1,1)

   
    cate_err=torch.mean( torch.square( ( (cate_true) - (cate_pred) ) ) )
    #cate_err=np.mean( np.square( ( (cate_true) - (cate_pred) ) ) )
    #return np.sqrt(cate_err).item()

    return torch.sqrt(cate_err).item()

def cal_pehe_nn(data,y,Reg,Enc,mask_gamma,mask_delta,mask_upsilon,fstart,fend,sstart,send,tstart,tend):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        datat=data.to_numpy()
        datat=torch.from_numpy(datat.astype(np.float32)).to(device)
        yt=y.to_numpy()
        yt=torch.from_numpy(yt.astype(np.float32)).to(device)
        df_datac=data[data['treatment']==0]
        df_datat=data[data['treatment']==1]
        
        
        torch_c=df_datac.to_numpy()
        torch_c=torch.from_numpy(torch_c.astype(np.float32))
        torch_t=df_datat.to_numpy()
        torch_t=torch.from_numpy(torch_t.astype(np.float32))
        
        
        
        
        
        phi, phi_mean, phi_var=Enc(datat[:,5:])   
        phi_gamma=phi[:,fstart:fend]
        phi_delta=phi[:,sstart:send]
        phi_upsilon=phi[:,tstart:tend]
        #phi_irr=phi[:,frstart:frend]
        del_ups=torch.cat((phi_delta*mask_delta, phi_upsilon*mask_upsilon), 1)

        concat_pred=Reg(del_ups)
        
        
       
        dists = torch.sqrt(torch.cdist(torch_c, torch_t))
        
        c_index=torch.argmin(dists, dim=0).tolist()
        t_index=torch.argmin(dists, dim=1).tolist()
    
        yT_nn=df_datac.iloc[c_index]['y_factual']
        yC_nn=df_datat.iloc[t_index]['y_factual']
        yT_nn=yT_nn.to_numpy()
        yT_nn=torch.from_numpy(yT_nn.astype(np.float32)).to(device)
        yC_nn=yC_nn.to_numpy()
        yC_nn=torch.from_numpy(yC_nn.astype(np.float32)).to(device)
        y_nn = torch.cat([yT_nn, yC_nn],0) 
        

        
      
        cate_pred=concat_pred[:,1]-concat_pred[:,0]

        
        cate_nn_err=torch.mean(torch.square((((1 - 2 * datat[:,0]) * (y_nn - yt)) - cate_pred)))
        return cate_nn_err.item()
        #torch.mean( torch.square( (1-2*datat[:,0]) * (y_nn-y) - (concat_pred[:,1]-concat_pred[:,0]) ) )
    
        


def wasserstein(X, t, p=0.5, lam=10, its=10, sq=False, backpropT=False):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    it = torch.where(t > 0)[0]
    ic = torch.where(t < 1)[0]
    Xc = X[ic]
    Xt = X[it]
    nc = torch.tensor(Xc.shape[0], dtype=torch.float)
    nt = torch.tensor(Xt.shape[0], dtype=torch.float)

    if sq:
        M = torch.cdist(Xt, Xc, p=2) ** 2
    else:
        M = torch.sqrt(torch.cdist(Xt, Xc, p=2) ** 2)

    M_mean = torch.mean(M)
    M_drop = torch.nn.functional.dropout(M, p=10 / (nc * nt))
    delta = torch.max(M).detach()
    eff_lam = lam / M_mean

    num_row = M.shape[0]
    num_col = M.shape[1]
    row = delta * torch.ones(1, num_col).to(device)
    col = torch.cat([delta * torch.ones(num_row, 1).to(device), torch.zeros(1, 1).to(device)], dim=0)
    Mt = torch.cat([M, row], dim=0)
    Mt = torch.cat([Mt, col], dim=1)

    a = torch.cat([p * torch.ones(len(it), 1) / nt, (1 - p) * torch.ones(1, 1)], dim=0).to(device)
    b = torch.cat([(1 - p) * torch.ones(len(ic), 1) / nc, p * torch.ones(1, 1)], dim=0).to(device)

    Mlam = eff_lam * Mt
    K = torch.exp(-Mlam).to(device) + 1e-6
    U = K * Mt
    ainvK = K / a

    u = a.clone()
    for i in range(its):
        u = 1.0 / (ainvK @ (b / (u.T @ K).T))
    v = b / (u.T @ K).T

    T = u * (v.T * K)

    if not backpropT:
        T = T.detach()

    E = T * Mt
    D = 2 * torch.sum(E)

    return D, Mlam
def add_dummy_features_shuffle(data, no_features):
    np.random.seed(2)
    data_dummy=pd.DataFrame()
    dummies=no_features

    if(dummies==0):
        return data
    
    elif (dummies==1):
        #data_aux = pd.DataFrame({f"d{0}"  : np.random.randint(low=0, high=2, size=data.shape[0])})
        #arr=data.iloc[:,5].values
        #data_aux = pd.DataFrame({f"d{0}"  : data.iloc[:,5].sample(frac=1).reset_index(drop=True)})
        arr=data.iloc[:,5].values
        np.random.shuffle(arr)
        data_aux = pd.DataFrame({f"d{0}"  :arr.tolist() })
        #data_aux = pd.DataFrame({f"d{0}"  :np.random.shuffle(arr)},index=[0])
        data_dummy = pd.concat([data_dummy, data_aux],axis=1)
    else:
        for i in range(5,5+dummies):
            #np.random.seed(i)
            #data_aux = pd.DataFrame({f"d{i}"  : np.random.randint(low=0, high=2, size=data.shape[0])})
            #data_aux = pd.DataFrame({f"d{i}"  : data.iloc[:,5].sample(frac=1).reset_index(drop=True)})
            arr=data.iloc[:,5].values #22
            #arr=np.random.normal(0,1, size=data.shape[0])
            np.random.shuffle(arr)
            data_aux = pd.DataFrame({f"d{i}"  :arr.tolist() })
            data_dummy = pd.concat([data_dummy, data_aux],axis=1)
            
    new_data=pd.concat([data,data_dummy],axis=1)   

    return new_data
    
def kl_loss_2(mean1, logvar1 , beta,kl_flag):
    
    if(kl_flag==True):
        kl_divergence = -0.5 * (1 + logvar1 - mean1.pow(2) - logvar1.exp()).sum(-1).mean()
        #kl_divergence = -0.5 * torch.mean(1 + logvar1 - mean1.pow(2) - logvar1.exp())
        
    else:
        kl_divergence = -0.5 * (1 + logvar1 - mean1.pow(2) - logvar1.exp()).sum(-1).mean()
        #kl_divergence = -0.5 * torch.mean(1 + logvar1 - mean1.pow(2) - logvar1.exp())
    
    #return ( beta * (kl_divergence))
    return ((kl_divergence))
def kl_loss_3(mean1, logvar1 , beta):
    
    kl_divergence = -0.5 * (1 + logvar1 - mean1.pow(2) - logvar1.exp())
   
    #return torch.mean(( beta * (kl_divergence)), dim=0)
    return torch.mean(((kl_divergence)), dim=0)
    
def Y_GECO(predicted_y,train_y, tol):
    #Reconstruction_loss=(MSE(decoded_space[:,0:6],train_x[:,0:6])+BCE(decoded_space[:,6:25],train_x[:,6:25])+MSE(decoded_space[:,25:],train_x[:,25:]))
    #return  Reconstruction_loss -tol**2
    #y_loss=MSE_2(predicted_y,train_y)- tol**2
   
    y_loss=((predicted_y-train_y)**2- tol)
    mse=MSE(predicted_y,train_y)
    
    return y_loss.mean(), mse.mean().detach().item()
def get_dim_count(tens, threshold):
    
    binary_tensor = ( tens> threshold).float()
    count=torch.sum(binary_tensor).item()
   
    return count
def clear():
    gamma_count.clear()
    delta_count.clear()
    upsilon_count.clear()
    #omega_count.clear()
def get_mask(tens, threshold):
    
    binary_tensor = ( tens> threshold).float()
   
    return binary_tensor
def RC_GECO(train_x, decoded_space, tol):
    #Reconstruction_loss=(MSE(decoded_space[:,0:6],train_x[:,0:6])+BCE(decoded_space[:,6:25],train_x[:,6:25])+MSE(decoded_space[:,25:],train_x[:,25:]))
    #return  Reconstruction_loss -tol**2
    
    y_loss=torch.sum((decoded_space-train_x)**2- tol)

    
    return y_loss