from cProfile import label
import os

import matplotlib

# To avoid displaying the figures
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

MOD = 5 # 1,2,3,4,5,7,10
def notearsData(path, method):
    xs=[[]]
    ys=[{}]
    best_shd_base=1e10
    best_perf_base=""
    for file in os.listdir(path):
        if file.endswith(f"{method}.json"):
            xs[i].append(float(file.split("_")[1]))
            abs_path=os.path.join(path, file)
            with open(abs_path) as f:
                    data=json.load(f)
            perf_str=file.split("_")[1]+"\n"
            flag=0
            for key,value in data.items():
                if key not in ys[i]:
                    ys[i][key]=[[],[]]
                ys[i][key][0].append(value['mean'])
                ys[i][key][1].append(value['std'])
                if key =="shd":
                    if value['mean']<best_shd_base:
                        best_shd_base=value['mean']
                        flag=1
                perf_str+=f"{key}:{value['mean']}+-{value['std']}\n"
            
            if flag==1:
                best_perf_base=perf_str
    return xs, ys, best_perf_base


def casperData(path, method):
    i = 0
    xs=[[]]
    ys=[{}]
    best_shd_base=1e10
    best_perf_base=""
    for file in os.listdir(path):
        if file.endswith(f"{method}_mod{MOD}.json"):
            xs[i].append(float(file.split("_")[1]))
            abs_path=os.path.join(path, file)
            with open(abs_path) as f:
                    data=json.load(f)
            perf_str=file.split("_")[1]+"\n"
            flag=0
            for key,value in data.items():
                if key not in ys[i]:
                    ys[i][key]=[[],[]]
                ys[i][key][0].append(value['mean'])
                ys[i][key][1].append(value['std'])
                if key =="shd":
                    if value['mean']<best_shd_base:
                        best_shd_base=value['mean']
                        flag=1
                perf_str+=f"{key}:{value['mean']}+-{value['std']}\n"
            
            if flag==1:
                best_perf_base=perf_str
    return xs, ys, best_perf_base

if __name__ == '__main__':
    import os
    import json
    xs_ori=[[]]
    ys_ori=[{}]
    xs_casper=[[]]
    ys_casper=[{}]
    i=0
    gf="SF"
    setting="10_20_2000"
    # for notears linear
    method_ori="notears"
    mod=1
    linearty = "non-linear"
    path_ori = f"complexity_ob_2000/{linearty}/synthetic/{setting}/{gf}_gp"
    xs_ori, ys_ori, best_perf_base = notearsData(path_ori, method_ori)
    # print(xs_ori)
    # print(ys_ori)
    
    # for casper linear
    method_new="casper"
    mod=1
    linearty = "non-linear"
    path_new = f"casper_ob_2000/{linearty}/synthetic/{setting}/{gf}_gp"
    xs_casper, ys_casper, best_perf_casper = casperData(path_new, method_new)
    # print(xs_casper)
    # print(ys_casper)
    
    print("baseline:")
    print(best_perf_base)
    print("casper:")
    print(best_perf_casper)
    
    # from min to max and sort the x axis
    idx_ori = np.argsort(np.array(xs_ori[i]))
    xs_ori[i] = np.array(xs_ori[i])[idx_ori]
    
    idx = np.argsort(np.array(xs_casper[i]))
    xs_casper[i] = np.array(xs_casper[i])[idx]
    
    # # the operation for plot
    plt.figure(figsize=(20, 10))
    cnt = 0
    for key, item in ys_ori[i].items():
        plt.subplot(2, 3, cnt + 1)
        means_ori = np.array(item[0])[idx_ori]
        stds_ori = np.array(item[1])[idx_ori]
        

        means = np.array(ys_casper[i][key][0])[idx]
        stds = np.array(ys_casper[i][key][1])[idx]
        
        plt.plot(xs_ori[i], means_ori, label=f"{method_ori}")
        plt.fill_between(xs_ori[i], means_ori - stds_ori, means_ori + stds_ori, alpha=0.05, color='b')
        
        plt.plot(xs_casper[i], means, label=f"{method_new}")
        plt.fill_between(xs_casper[i], means - stds, means + stds, alpha=0.05, color='r')
        plt.ylabel(key.upper())
        plt.xlabel(r'$\lambda$')
        
        plt.title(key)
        plt.legend()
        cnt+=1
    
    plt.savefig(os.path.join('figs_nonlinear/',gf+setting+linearty+"batch_final.png"),bbox_inches='tight')
    plt.cla()

    