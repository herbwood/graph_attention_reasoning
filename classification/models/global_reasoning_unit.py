import torch
import torch.nn as nn
from models import attention 
from .attention import GAT, ScaledDotProductAttention


class GCN(nn.Module):

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1) # adjacent matrix 
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # conv1d by columns and rows 
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        h = self.conv2(self.relu(h))
        adj_matrix = self.conv1.weight

        return h, adj_matrix 


class GloRe_Unit(nn.Module):
    
    def __init__(self, num_in, num_mid, 
                 ConvNd=nn.Conv3d,
                 BatchNormNd=nn.BatchNorm3d,
                 normalize=False,
                 out_features=128,
                 n_heads=8,
                 is_concat=True,
                 dropout=0.6):
        super(GloRe_Unit, self).__init__()
        
        self.normalize = normalize
        self.num_s = int(2 * num_mid) # 128 -> feature_dim
        self.num_n = int(1 * num_mid) # 64 -> num_nodes

        #######################Graph Attention##############################
        self.dropout = nn.Dropout(dropout)

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Graph Attention Network
        self.gat = GAT(self.num_s, self.n_hidden, out_features, n_heads, dropout)

        # Self Attention 
        # self.selfattn = attention.ScaledDotProductAttention(d_model=128, d_k=64, d_v=64, h=8)
        ####################################################################

        # dim reduction layer
        self.conv_state = ConvNd(num_in, self.num_s, kernel_size=1)
        
        # projection matrix layer 
        self.conv_proj = ConvNd(num_in, self.num_n, kernel_size=1)

        # Graph Convolution Network 
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        
        # recover channel size 
        self.conv_extend = ConvNd(self.num_s, num_in, kernel_size=1, bias=False)

        # batch norm 
        self.blocker = BatchNormNd(num_in, eps=1e-04) 


    def forward(self, x):
        
        n = x.size(0)

        # hidden state 
        # feature map with reduced dim 
        # output shape : (batch size, feature_dim, channel) -> (5, 128. 1024)
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)

        # projection matrix 
        # output shape : (batch size, num_nodes, channel) -> (5, 64, 1024)
        x_proj_reshaped = self.conv_proj(x).view(n, self.num_n, -1)

        # reverse projection matrix 
        # output shape : (batch size, num_nodes, channel)
        x_rproj_reshaped = x_proj_reshaped

        # matrix multiplication 
        # hidden state x projection matrix 
        # output shape : (batch size, feature_dim, num_nodes)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # graph convolutional network
        # 1) conv1d : by columns
        # 2) element-wise addition : conv1d(x) + x
        # 3) conv1d + relu : by rows 
        # x_n_rel shape : (batch size, feature_dim, num_nodes)
        # adj_matrix shape : (num_nodes, num_nodes)
        x_n_rel, adj_matrix = self.gcn(x_n_state)

        #######################Graph Attention##############################
        x_n_rel = self.gat(x_n_rel, adj_matrix)
        ####################################################################

        #######################Self Attention##############################
        # x_n_rel = x_n_rel.transpose(1, 2)
        # x_n_rel = self.selfattn(x_n_rel, x_n_rel, x_n_rel)
        # x_n_rel = x_n_rel.transpose(1, 2)
        ####################################################################       
        
        # matrix multiplication 
        # graph output x reverse projection 
        # output shape : (batch size, feature_dim, num_nodes)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # resize to original feature map size
        # output shape : (batch size, feature_dim, height, width)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])

        # final output 
        # 1) 1x1 conv2d : back to original channel
        # 2) batch norm 
        # 3) element-wise addition : with original feature map 
        out = x + self.blocker(self.conv_extend(x_state))

        return out


class GloRe_Unit_1D(GloRe_Unit):
    def __init__(self, num_in, num_mid, normalize=False):
        
        super(GloRe_Unit_1D, self).__init__(num_in, num_mid,
                                            ConvNd=nn.Conv1d,
                                            BatchNormNd=nn.BatchNorm1d,
                                            normalize=normalize)

class GloRe_Unit_2D(GloRe_Unit):
    def __init__(self, num_in, num_mid, normalize=False):
        
        super(GloRe_Unit_2D, self).__init__(num_in, num_mid,
                                            ConvNd=nn.Conv2d,
                                            BatchNormNd=nn.BatchNorm2d,
                                            normalize=normalize,
                                            out_features=128,
                                            n_heads=8,
                                            is_concat=True
                                            )

class GloRe_Unit_3D(GloRe_Unit):
    def __init__(self, num_in, num_mid, normalize=False):
        
        super(GloRe_Unit_3D, self).__init__(num_in, num_mid,
                                            ConvNd=nn.Conv3d,
                                            BatchNormNd=nn.BatchNorm3d,
                                            normalize=normalize)

