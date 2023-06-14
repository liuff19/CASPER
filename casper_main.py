from cProfile import run
from sympy import re
from notears.locally_connected import LocallyConnected
from notears.trace_expm import trace_expm
import torch
import torch.nn as nn
from notears.lbfgsb_scipy import LBFGSBScipy
import numpy as np
import tqdm as tqdm
from notears.loss_func import *
import random
import time
import igraph as ig
import notears.utils as ut
from notears.loss_func import *
import torch.utils.data as data
import os
from runhelps.runhelper import config_parser
from torch.utils.tensorboard import SummaryWriter
from sachs_data.load_sachs import *
import torch.optim as optim
import torch.nn.functional as F
import json

COUNT = 0
IF_figure = 0

parser = config_parser()
args = parser.parse_args()
print(args)
IF_baseline = args.run_mode
run_mode = args.run_mode # 0: casper ; 1: baseline; 2: batch + baseline

Hscore = []
logHscore = []
loss_score = []
class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2    
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)  
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers) 
        layers = []
        # TODO: add the discriminator function for wasserstein loss
        layers.append(LocallyConnected(d, d, 1, bias=bias))
        self.fc3 = nn.ModuleList(layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1] 
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d] 
        return x

    def forward__(self, x):  # [n, d] -> [n, d]
        W = 1 - torch.eye (x.shape [1])
        W = W.reshape (1, W.shape [0], -1).expand (x.shape [0], -1, -1) 
        x = x.reshape (x.shape [0], 1, -1).expand (-1, x.shape [1], -1) 
        x = x * W 
        for _, fc in enumerate (self.fc3) :
            if _ != 0 : 
                x = torch.relu (x)  # [n, d, m1]
            #else :
                #x = torch.nn.functional.dropout (x, 0.5)
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x
    
    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i] 
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        # h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        M = torch.eye(d).to(A.device) + A / d  # (Yu et al. 2019)
        E = torch.matrix_power(M, d - 1)
        h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg 

    def l2_reg__(self):
        """Take 2-norm-squared of all parameters in Discriminator """
        reg = 0.
        for fc in self.fc3:
            reg += torch.sum(fc.weight ** 2)
        return reg
    
    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


def wasserstein_loss(x_distribution, wx_distribution):
    # TODO: we need to pay attention to the axis of the average
    # TODO: check the correctness of the loss function
    return (torch.mean(x_distribution, axis = 0) - torch.mean(wx_distribution, axis = 0)).sum()
    
def dual_ascent_step(model, X, train_loader, lambda1, lambda2, rho, alpha, h, rho_max, adp_flag):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    space_optimizer = optim.RMSprop(model.fc3.parameters(), lr = args.space_lr, momentum = 0.9)
    # X_torch = torch.from_numpy(X)
    while rho < rho_max:
        def closure__():
            space_optimizer.zero_grad()
            primal_obj = torch.tensor(0.).to(args.device)
            
            for _ , tmp_x in enumerate(train_loader):
                batch_x = tmp_x[0].to(args.device)
                x_distribution = model.forward__(batch_x)
                wx_distribution = model.forward__(model(batch_x))
                primal_obj += -wasserstein_loss(x_distribution, wx_distribution)
            
            l2_reg = 10 * 0.5 * lambda2 * model.l2_reg__ ()
            primal_obj += l2_reg
            primal_obj.backward ()
            return primal_obj
        
        def closure():
            global COUNT
            COUNT += 1
            optimizer.zero_grad()
            X_hat = model(X)
            loss = squared_loss(X_hat, X)
            h_val = model.h_func()
            Hscore.append(h_val.detach().numpy())
            logHscore.append(torch.log(1+h_val).detach().numpy())
           
            loss_score.append(loss.detach().numpy()-12.5)
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            return primal_obj
        
        def batch_closure():
            global COUNT
            COUNT += 1
            optimizer.zero_grad()

            primal_obj = torch.tensor(0.).to(args.device)
            loss = torch.tensor(0.).to(args.device)

            for _ , tmp_x in enumerate(train_loader):
                batch_x = tmp_x[0].to(args.device)
                X_hat = model(batch_x)
                primal_obj += squared_loss(X_hat, batch_x)
            
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj += penalty + l2_reg + l1_reg
            primal_obj.backward()
            
            return primal_obj
        
        def w_closure():
            global COUNT
            if COUNT % args.iter_mod == 0 :
                # TODO: add the h_func to control the clip args
                args.clip = torch.sigmoid(model.h_func())
                for i in range (1) :
                    LLL = closure__()
                    space_optimizer.step ()
                    for p in model.fc3.parameters():
                        p.data.clamp_(-args.clip, args.clip)
            COUNT += 1
            optimizer.zero_grad()

            primal_obj = torch.tensor(0.).to(args.device)
            loss_D = torch.tensor(0.).to(args.device)

            for _ , tmp_x in enumerate(train_loader):
                batch_x = tmp_x[0].to(args.device)
                X_hat = model(batch_x)
                x_distribution = model.forward__(batch_x)
                wx_distribution = model.forward__(model(batch_x))
                loss_D += wasserstein_loss(x_distribution, wx_distribution)
                primal_obj += squared_loss(X_hat, batch_x)
                # if COUNT % 10 == 0:
                #     print(loss_D)
            h_val = model.h_func()
            
            # print(h_val)
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj += penalty + l2_reg + l1_reg + 0.01 * loss_D
            primal_obj.backward()
            
            for p in model.fc3.parameters () :
                p.grad.zero_ ()
            return primal_obj
        
        if run_mode == 1:
            optimizer.step(closure)  # NOTE: updates model in-place
        elif run_mode == 2:
            optimizer.step(batch_closure)
        elif run_mode == 0:
            optimizer.step(w_closure)

        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def notears_nonlinear(model: nn.Module,
                    X: np.ndarray,
                    train_loader: data.DataLoader,
                    lambda1: float = 0.,
                    lambda2: float = 0.,
                    max_iter: int = 100,
                    h_tol: float = 1e-8,
                    rho_max: float = 1e+16,
                    w_threshold: float = 0.3
                    ):
    rho, alpha, h = 1.0, 0.0, np.inf
    adp_flag = False
    for j in tqdm.tqdm(range(max_iter)):
        rho, alpha, h = dual_ascent_step(model, X, train_loader, lambda1, lambda2,
                                        rho, alpha, h, rho_max, adp_flag)
        
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def main(trials, seed):
    tot_perf={}
    casper=f"_space_{args.iter_mod}" if not IF_baseline else ""
    import notears.utils as ut
    for trial in range(trials):
        print('==' * 20)
        cur_seed = trial + seed
        set_random_seed(cur_seed)

        if args.modeltype == 'notears':
            if args.linear:
                model = NotearsMLP(dims=[args.d, 1], bias=True) # FIXME: the layer of the Notears MLP
            else:
                model = NotearsMLP(dims=[args.d,10, 1], bias=True) # FIXME: if the nonlinear setting needs to add 10
        
        if args.data_type=="synthetic" and args.linear:
            linearity = "linear"
        else:
            linearity = "non-linear"
        
        datatype = args.data_type
        sem_type = args.sem_type if linearity=="non-linear" else args.linear_sem_type
        data_dir=f'data/{linearity}/{args.graph_type}_{sem_type}/'
        ensureDir(data_dir)
        
        if not args.scaled_noise:
            data_name=f'{args.d}_{args.s0}_{args.n}_{cur_seed}'
        
        try:
            X=np.load(data_dir+data_name+"_X.npy")
            B_true=np.load(data_dir+data_name+"_B.npy")
            print("data loaded...")
        except:
            print("generating data from scratch...")
            if args.data_type == 'real':
                X = np.loadtxt('sachs_path', delimiter=',')
                B_true = np.loadtxt('sachs_path', delimiter=',')
            elif args.data_type == 'synthetic':
                B_true = ut.simulate_dag(args.d, args.s0, args.graph_type)
                if args.linear:
                    W_true=ut.simulate_parameter(B_true)
                    X = ut.simulate_linear_sem(W_true, args.n, args.linear_sem_type)
                    np.save(data_dir+data_name+"_W",W_true)
                else:
                    X = ut.simulate_nonlinear_sem(B_true, args.n, args.sem_type)
                    
            np.save(data_dir+data_name+"_X",X)
            np.save(data_dir+data_name+"_B",B_true)
        
        print("X[-1]====",X.shape[-1])
        print("args.n====",args.n)
        
        if args.run_mode:
            if args.run_mode == 1:
                prefix = f'complexity_ob_{args.n}'
            elif args.run_mode == 2:
                prefix = f'complexity_ob_batch_{args.n}'
        else:
            prefix = f'casper_ob_{args.n}'
            
        if args.data_type == 'real' or args.data_type == 'sachs_full':
            f_dir=f'{prefix}/{linearity}/{args.data_type}/{args.d}_{args.s0}_{args.n}/{args.data_type}/'
        else:
            f_dir=f'{prefix}/{linearity}/{args.data_type}/{args.d}_{args.s0}_{args.n}/{args.graph_type}_{sem_type}/'
        if not os.path.exists(f_dir):
            os.makedirs(f_dir)
            
            
        X = torch.from_numpy(X).float().to(args.device)
        model.to(args.device)
        
        X_data = data.TensorDataset(X)
        train_loader = data.DataLoader(X_data, batch_size=args.batch_size, shuffle=True)

        if args.modeltype == 'notears':
            W_est = notears_nonlinear(model, X, train_loader, args.lambda1, args.lambda2)
        assert ut.is_dag(W_est)
        
        acc = ut.count_accuracy(B_true, W_est != 0)
        print(acc)
        
        f_path=f'{prefix}/{linearity}/{args.data_type}/{args.d}_{args.s0}_{args.n}/{args.graph_type}_{sem_type}/seed_{cur_seed}.txt'

        if args.data_type == 'synthetic':
            with open(f_path, 'a') as f:
                f.write(f'args:{args}\n')
                f.write(f'run_mode: {IF_baseline}\n')
                f.write(f'observation_num: {args.n}\n')
                if not IF_baseline:
                    f.write(f'iter_mod: {args.iter_mod}\n')
                    f.write(f'batch_size:{args.batch_size}\n')
                f.write(f'dataset_type:{args.data_type}\n')
                f.write(f'modeltype:{args.modeltype}\n')
                f.write(f'acc:{acc}\n')
                f.write('-----------------------------------------------------\n')
        for key,value in acc.items():
            if key not in tot_perf:
                tot_perf[key]={"value":[],"mean":[],"std":[]}
            tot_perf[key]["value"].append(value)
    for key,value in tot_perf.items():
        perf=np.array(value["value"])
        tot_perf[key]['mean']=float(np.mean(perf))
        tot_perf[key]['std']=float(np.std(perf))

    # TODO: add the flag to identify casper
    if args.run_mode:
        with open(f_dir+"stats_"+str(args.lambda1)+"_"+str(args.lambda2)+"_"+args.modeltype+".json",'w') as f:
            json.dump(tot_perf,f)
    else:
        with open(f_dir+"stats_"+str(args.lambda1)+"_"+str(args.lambda2)+"_"+"casper"+"_"+"mod"+str(args.iter_mod)+".json",'w') as f:
            json.dump(tot_perf,f)
        
if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=10)
    main(args.trial,args.seed)
    import matplotlib.pyplot as plt
    # 将Hscore和logHscore 转成numpy
    Hscore = np.array(Hscore)
    logHscore = np.array(logHscore)
    # plot the Hscore and logHscore in a single figure
    # 将Hscore和logHscore画在一张图上
    plt.figure()
    # plt.plot(loss_score, label='Loss')
    # plt.plot(Hscore, label='logHscore')
    # 画出拟合曲线，即去掉毛刺
    plt.plot(loss_score, label='Loss')
    plt.plot(Hscore, label='logHscore')
    plt.xlabel('iteration')
    plt.legend()
    # save figure
    plt.savefig('Hscore.png')
