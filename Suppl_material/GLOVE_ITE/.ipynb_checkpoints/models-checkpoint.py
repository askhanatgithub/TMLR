from torch import nn   
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Tclassifier(nn.Module):
    def __init__(self,
                 input_dim,
                 regularization):
        super(Tclassifier, self).__init__()
       
        
        #Classifier to calculate weights
        self.classifier_w1 = nn.Sequential(
            nn.Linear(input_dim, 100),
            
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        self.classifier_w2 = nn.Sequential(
            nn.Linear(input_dim, 100),
           
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        
       
        self.classifier_w3 = nn.Linear(100, 1) #100
        self.sig = nn.Sigmoid()
        
        


    def forward(self, inputs):
       
        
        # classifires
       
        #out_w=self.classifier_w1(inputs)
        out_w=self.classifier_w2(inputs)
        
       
        out_w_f=self.sig(self.classifier_w3(out_w))
        
        
        # Returning arguments

       
        return out_w_f
    
class Regressors(nn.Module):
    def __init__(self,
                 input_dim,hid_dim,
                 regularization):
        super(Regressors, self).__init__()
        
    

        self.regressor1_y0 = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
          
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        
        self.regressor2_y0 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
          
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        self.regressor3_y0 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
          
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
            
        self.regressor4_y0 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
          
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        
        self.regressor5_y0 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
          
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        
                
        self.regressor6_y0 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
          
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
    
        self.regressorO_y0 = nn.Linear(hid_dim, 1)
        
        

        self.regressor1_y1 = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
           
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        self.regressor2_y1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
           
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        
        self.regressor3_y1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
           
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        
        self.regressor4_y1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
           
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        
        self.regressor5_y1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
           
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        
        self.regressor6_y1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
           
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        
       
        self.regressorO_y1 = nn.Linear(hid_dim, 1)
        
        


    def forward(self, inputs):

        # Regressors
        #del_ups=torch.cat((phi_delta, phi_upsilon), 1)
        out_y0 = self.regressor1_y0(inputs)
        out_y0 = self.regressor2_y0(out_y0)
        out_y0 = self.regressor3_y0(out_y0)
        #out_y0 = self.regressor4_y0(out_y0)  
        #out_y0 = self.regressor5_y0(out_y0)
        #out_y0 = self.regressor6_y0(out_y0)
        y0 = self.regressorO_y0(out_y0)

        out_y1 = self.regressor1_y1(inputs)
        out_y1 = self.regressor2_y1(out_y1)
        out_y1 = self.regressor3_y1(out_y1)
        #out_y1 = self.regressor4_y1(out_y1)
        #out_y1 = self.regressor5_y1(out_y1)
        #out_y1 = self.regressor6_y1(out_y1)
        
        y1 = self.regressorO_y1(out_y1)
        
        # classifires
        #gam_del=torch.cat((phi_gamma,phi_delta), 1)
        #out_w=self.classifier_w1(phi_delta)
        #out_w=self.classifier_w2(out_w)
        #out_w_f=self.sig(self.classifier_w3(out_w))
        
        #out_t=self.classifier_t1(gam_del)
        #out_t=self.classifier_t2(out_t)
        #out_t_f=self.sig(self.classifier_t3(out_t))
        
        # Returning arguments

        concat = torch.cat((y0, y1), 1)
        return concat#out_w_f,out_t_f


    
class Decoder(nn.Module):
    def __init__(self, input_dim, decoding_dim,regularization):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 600),  # First decoder layer, 200 dimensions
            nn.ELU(),
            nn.Dropout(p=regularization),
           
            nn.Linear(600, 400),  # First decoder layer, 200 dimensions
            nn.ELU(),
            nn.Dropout(p=regularization),
            

            

            nn.Linear(400, decoding_dim)    # Third decoder layer, output original dimensions
            
            
        )
        self.sig=nn.Sigmoid()

    def forward(self, x):
        decoded = self.decoder(x)
        decoded_a=decoded[:,0:6]
        decoded_b=self.sig(decoded[:,6:25])
        decoded_c=decoded[:,25:]
        return torch.cat((decoded_a, decoded_b,decoded_c), 1)


class Net(nn.Module):
    def __init__(self,
                 input_dim,hid_enc,lat_dim_enc,
                 regularization,fstart,fend,sstart,send,tstart,tend):
        super(Net, self).__init__()
        self.fstart=fstart
        self.fend=fend
        self.sstart=sstart
        self.send=send
        self.tstart=tstart
        self.tend=tend
        self.encoder_gamma_1 = nn.Linear(input_dim,hid_enc)
        self.encoder_gamma_2 = nn.Linear(hid_enc, hid_enc)
        self.encoder_gamma_3_mean = nn.Linear(hid_enc, lat_dim_enc)
        self.encoder_gamma_3_var = nn.Linear(hid_enc, lat_dim_enc)
       
        self.sig = nn.Sigmoid()
        self.BN= nn.BatchNorm1d(lat_dim_enc)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
    

      
    def reparameterize(self, mean, logvar):
        #version 1
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
        
        #version 2
        #eps = torch.randn_like(logvar)
        #return mean + eps * logvar

        #version 3

        #return mean+logvar*self.N.sample(mean.shape)

    def forward(self, inputs):
        x_gamma = nn.functional.elu(self.encoder_gamma_1(inputs))
        x_gamma = nn.functional.elu(self.encoder_gamma_2(x_gamma))
        phi_gamma_mean = self.encoder_gamma_3_mean(x_gamma)
        phi_gamma_var = self.encoder_gamma_3_var(x_gamma)
        

        phi_gamma_z = self.reparameterize(phi_gamma_mean, phi_gamma_var)
       
   
        phi_gamma=phi_gamma_z[:,self.fstart:self.fend]
        phi_delta=phi_gamma_z[:,self.sstart:self.send]
        phi_upsilon=phi_gamma_z[:,self.tstart:self.tend]
        #phi_irr=phi_gamma_z[:,frstart:frend]*mask_omega

        phi=torch.cat((phi_gamma,phi_delta,phi_upsilon),1)
 
        return (phi,phi_gamma_mean,phi_gamma_var)
