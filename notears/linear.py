import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
import matplotlib.pyplot as plt
import os
import tqdm as tqdm
beta = 0.1
reweight_list = []
epoch = 6
def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3, B_true=None):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps #
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        """定义损失函数并且计算损失函数的值和梯度"""
        M = X @ W  # M是X矩阵和W矩阵的乘积，X是样本矩阵，W是权重矩阵，X的维度是[n, d]，W的维度是[d, d]，M的维度是[n, d]
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R  # G_loss是损失函数的梯度 维度是[d, d]

        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _reweighloss(W):
        M = X @ W
        R = X - M
        assert loss_type in ['l2', 'logistic', 'poisson']
        rewei_num = len(reweight_list)
        re_matrix = np.eye(100,100)
        for idx in reweight_list:
            re_matrix[idx][idx] = beta
        loss = 0.5 / X.shape[0] * ((re_matrix @ R) ** 2).sum()
        G_loss = - 1.0 / X.shape[0] * X.T @ ((re_matrix**2) @ R)
        # return loss, G_loss
        return loss,G_loss

    
    def _single_loss(W):
        # 只计算每个节点带来的损失函数
        M = X @ W
        assert loss_type == 'l2'
        R = X - M
        # 矩阵维度是[n, d]，所以每个节点的损失函数是一个[n, 1]的矩阵
        loss = 0.5 / X.shape[0] * (R ** 2)
        # 求出每个节点的损失函数
        return loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        # 检查E中有没有nan元素，如果有则输出1
        if np.any(np.isnan(E)) or np.any(np.isinf(E)):
            print('nan in expm')
            return 1
        h = np.trace(E) - d
        if np.any( np.isnan(h)) or np.any(np.isinf(h)):
            print('nan in h')
            return 1
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate(
            (G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    def _refunc(w):
        W = _adj(w)
        loss, G_loss = _reweighloss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate(
            (G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape  # n为样本数，d为特征数
    # w_est是最终的结果，rho是惩罚系数，alpha是拉格朗日乘子，h是拉格朗日函数的值
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf

    bnds = [(0, 0) if i == j else (0, None) for _ in range(2)
            for i in range(d) for j in range(d)]  # bnds是约束条件，每个元素的第一个元素是下界，第二个元素是上界
    if loss_type == 'l2':
        # 对X进行均值归一化，np.mean(X, axis=0, keepdims=True)表示按列求均值, 输出是一个[1,d]的矩阵
        X = X - np.mean(X, axis=0, keepdims=True)

    ob_loss = []
    total_loss = []
    xepoch = 0
    for iter in tqdm.tqdm(range(max_iter)):
        w_new, h_new = None, None
        while rho < rho_max:
            xepoch+=1
            # 这里的sol是一个字典，sol['x']是更新后的w，sol['fun']是更新后的h
            if xepoch<epoch:
                sol = sopt.minimize(
                    _func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            elif xepoch==epoch:
                w_new = sol.x
                loss_record = _single_loss(_adj(w_new))
                each_loss = loss_record.sum(axis=1)
                # 让each_loss按照每个节点的损失函数进行排序 从小到大
                each_loss_idx = each_loss.argsort()
                reweight_list = each_loss_idx[:int(len(each_loss_idx)*0.1)]
                sol = sopt.minimize(
                    _refunc, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
                # sol = sopt.minimize(
                #     _func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            elif xepoch>epoch:
                sol = sopt.minimize(
                    _refunc, w_est, method='L-BFGS-B', jac=True, bounds=bnds)

            w_new = sol.x  # sol.x是一个[2 d^2]的矩阵，w_new是一个[d, d]的矩阵, 是最优解

            # _adj(w_new)是一个[d, d]的矩阵，h_new是一个数值，是拉格朗日函数的值，_adj是一个函数，_h是一个函数
            loss_record = _single_loss(_adj(w_new))
            # 将loss_record按照行求和，得到一个[n, 1]的矩阵
            
            each_loss = loss_record.sum(axis=1)
            ob_loss.append(each_loss)
            total_loss.append(_loss(_adj(w_new))[0])
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:       # 如果拉格朗日函数的值大于0.25*h，则rho变大
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new     # 更新w_est和h
        alpha += rho * h            # 更新alpha
        if h <= h_tol or rho >= rho_max:   # 如果拉格朗日函数的值小于等于h_tol或者步长大于等于rho_max，则结束循环
            break
    # 把ob_loss列表中的元素concatenate到一起，得到一个[len(ob_loss), n]的矩阵
    observed_loss = ob_loss[0].reshape(100, 1)
    for i in range(1, len(ob_loss)):
        observed_loss = np.concatenate(
            (observed_loss, ob_loss[i].reshape(100, 1)), axis=1)
    # 绘制一个可以放100张图的图
    plt.figure(figsize=(50, 50))
    for i in range(100):
        plt.subplot(10, 10, i + 1)
        plt.plot(observed_loss[i])
        # 调整title字体大小
        plt.title(i, fontsize=10)
        plt.title('node {}'.format(i))
        plt.axis('on')
        plt.box(True)

    if not os.path.exists('linear_notears'):
        os.makedirs('linear_notears')
    plt.savefig('linear_notears/observation.png')    

    plt.figure(figsize=(20, 10))
    plt.plot(total_loss)
    plt.title('total loss(30nodes+noise*10)')
    plt.savefig('linear_notears/total_loss.png')


    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est, iter


if __name__ == '__main__':
    # from notears import utils
    import utils
    utils.set_random_seed(1)

    # n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
    n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    # np.savetxt('W_true.csv', W_true, delimiter=',')

    X = utils.simulate_linear_sem(W_true, n, sem_type)
    # np.savetxt('X.csv', X, delimiter=',')

    W_est, iter_num = notears_linear(
        X, max_iter=100, lambda1=0.1, loss_type='l2', B_true=B_true)
    print(f'iteration: {iter_num}')
    assert utils.is_dag(W_est)
    # np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)
