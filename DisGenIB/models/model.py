
import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#Encoder
class Encoder(nn.Module):

    def __init__(self, opt,encoder_layer_sizes,latent_size):
        super(Encoder,self).__init__()
        layer_sizes = encoder_layer_sizes
        latent_size = latent_size
        self.opt = opt
        if opt.encoder_use_y:
            layer_sizes[0] += latent_size
        self.fc1=nn.Linear(layer_sizes[0], layer_sizes[-1])
        self.fc3=nn.Linear(layer_sizes[-1], latent_size*2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.linear_means = nn.Linear(latent_size*2, latent_size)
        self.linear_log_var = nn.Linear(latent_size*2, latent_size)
        self.apply(weights_init)

    def forward(self, x, c=None):
        if self.opt.encoder_use_y:
            x = torch.cat((x, c), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

class Generator(nn.Module):
    def __init__(self,opt,decoder_layer_sizes,latent_size,attsize):
        super(Generator,self).__init__()
        layer_sizes = decoder_layer_sizes
        latent_size = latent_size
        input_size = latent_size + attsize
        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc3 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid=nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.apply(weights_init)

    def _forward(self, z, c=None):
        z = torch.cat((z, c), dim=-1)
        x1 = self.lrelu(self.fc1(z))
        # x = self.sigmoid(self.fc3(x1))
        # x = self.tanh(self.fc3(x1))
        x = self.fc3(x1)
        self.out = x1
        return x
    def forward(self, z, a1=None, c=None, feedback_layers=None):
        if feedback_layers is None:
            return self._forward(z,c)
        else:
            z = torch.cat((z, c), dim=-1)
            x1 = self.lrelu(self.fc1(z))
            feedback_out = x1 + a1*feedback_layers
            # x = self.sigmoid(self.fc3(feedback_out))
            # x = self.tanh(self.fc3(feedback_out))
            x = self.fc3(feedback_out)
            
            return x

class Classifier(nn.Module):
    def __init__(self,input_size,hidden,n_class):
        super(Classifier,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden)
        self.fc2 = nn.Linear(hidden,n_class)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)