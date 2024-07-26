import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import os
from .STMGraph import STMGraph
from .utils import Transfer_pytorch_Data
import random
import torch
import torch.backends.cudnn as cudnn
import os
import torch.nn.functional as F

def sce_loss(x, y, alpha=1.0):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

def train_STMGraph(adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001,mask_ratio=0.5,noise=0.05,alpha=1.0,key_added='STMGraph',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=52, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    save_loss
        If True, the training loss is saved in adata.uns['STMGraph_loss'].
    save_reconstrction
        If True, the reconstructed expression profiles are saved in adata.layers['STMGraph_ReX'].
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
    seed=random_seed
    fix_seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    adata.X = sp.csr_matrix(adata.X)
    
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    data = Transfer_pytorch_Data(adata_Vars)

    model = STMGraph(hidden_dims = [data.x.shape[1]] + hidden_dims).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-4)
    # optimizer = torch.optim.Adam(list(model.parameters()), lr=lr, weight_decay=weight_decay)

    #loss_list = []
    for epoch in tqdm(range(1, n_epochs+1)):
        model.train()
        optimizer.zero_grad()
        # z, out = model(data.x, data.edge_index)
        z, out_1, out_2, mask_nodes, keep_nodes = model(data.x, data.edge_index, mask_rate=mask_ratio, replace_rate=noise)
        # loss = F.mse_loss(data.x, out_1)
        loss = sce_loss(data.x[mask_nodes],out_1[mask_nodes],alpha=alpha) + sce_loss(data.x[keep_nodes],out_2[keep_nodes],alpha=alpha)
        # loss = sce_loss(data.x,out_1,alpha=alpha) + sce_loss(data.x,out_2,alpha=alpha)
        # F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        # loss_list.append(loss)
        print(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
        print(loss.item())
    
    model.eval()
    z, out_1, out_2, mask_nodes, keep_nodes = model(data.x, data.edge_index, mask_rate=0.0, replace_rate=0.0)
    STMGraph_rep = z.to('cpu').detach().numpy()
    adata.obsm[key_added] = STMGraph_rep
    torch.cuda.empty_cache()
    if save_loss:
        adata.uns['STMGraph_loss'] = loss
    if save_reconstrction:
        ReX = out.to('cpu').detach().numpy()
        ReX[ReX<0] = 0
        adata.layers['STMGraph_ReX'] = ReX

    return adata

def fix_seed(seed):
    #seed = 2023
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  