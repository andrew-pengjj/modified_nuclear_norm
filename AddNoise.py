import torch
import numpy as np
import scipy.io

def soft_thresh(x, soft):
    tmp = abs(x) - soft
    tmp[tmp < 0] = 0.0
    return torch.sign(x) * tmp

def add_sp(image,prob):
    h = image.shape[0]
    w = image.shape[1]
    output = image.copy()
    sp = h*w   # 计算图像像素点个数
    NP = int(sp*prob)   # 计算图像椒盐噪声点个数
    for i in range(NP):
        randx = np.random.randint(1, h-1)   # 生成一个 1 至 h-1 之间的随机整数
        randy = np.random.randint(1, w-1)   # 生成一个 1 至 w-1 之间的随机整数
        if np.random.random() <= 0.5:   # np.random.random()生成一个 0 至 1 之间的浮点数
            output[randx, randy] = 0
        else:
            output[randx, randy] = 1
    return output

def add_sp_noise(data_path, std_e):
    data = scipy.io.loadmat(data_path)
    cln_hsi = data['data'].astype(np.float32)
    Hei, Wid, Band = cln_hsi.shape
    noi_hsi = np.zeros([Hei,Wid,Band])
    print('add sparse noise  (%s)' % std_e)
    for ind in range(Band):
        noi_hsi[:, :, ind] = add_sp(cln_hsi[:, :, ind].copy(),std_e)
    return cln_hsi, noi_hsi

def add_gaussian(image, sigma):
    # add gaussian noise
    # image in [0,1], sigma in [0,1]
    output = image.copy()
    output = output + np.random.normal(0, sigma,image.shape)
    # output = output + np.random.randn(image.shape[0], image.shape[1])*sigma
    return output

def add_Gaussian_noise(data_path, std_e):
    data = scipy.io.loadmat(data_path)
    cln_hsi = data['data'].astype(np.float32)
    Hei, Wid, Band = cln_hsi.shape
    noi_hsi = np.zeros([Hei,Wid,Band])
    print('add Gaussian noise  (%s)' % std_e)
    for ind in range(Band):
        noi_hsi[:, :, ind] = add_gaussian(cln_hsi[:, :, ind].copy(),std_e)
    return cln_hsi, noi_hsi

def add_Mixture_noise(data_path,std_g, std_s):
    data = scipy.io.loadmat(data_path)
    cln_hsi = data['data'].astype(np.float32)
    Hei, Wid, Band = cln_hsi.shape
    noi_hsi = np.zeros([Hei,Wid,Band])
    print('add Gaussian noise  (%s)' % std_g)
    print('add Sparse noise  (%s)' % std_s)
    for ind in range(Band):
        noi_hsi[:, :, ind] = add_gaussian(noi_hsi[:, :, ind].copy(), std_g)
        noi_hsi[:, :, ind] = add_sp(cln_hsi[:, :, ind].copy(), std_s)
    return cln_hsi, noi_hsi

def Mask_R(TensorInput, ratio):
    [M, N, p] = TensorInput.shape
    mask = np.zeros((M, N, p))
    mask[np.random.rand(M, N, p) <= ratio] = 1
    mask = mask.reshape(M*N,p)
    #mask = torch.FloatTensor(mask)
    return mask

def ErrRelGlobAdimSyn(imagery1, imagery2):
    m, n, k = imagery1.shape
    mm, nn, kk = imagery2.shape
    m = min(m, mm)
    n = min(n, nn)
    k = min(k, kk)
    imagery1 = imagery1[:m, :n, :k]
    imagery2 = imagery2[:m, :n, :k]

    ergas = 0
    for i in range(k):
        ergas += np.mean(np.square(imagery1[:, :, i] - imagery2[:, :, i])) / np.mean(imagery1[:, :, i])

    ergas = 100 * np.sqrt(ergas / k)
    return ergas

def CAUC(test_targets, output):
    # 计算AUC值，test_targets为原始样本标签，output为分类器得到的判为正类的概率，均为行或列向量
    test_targets = test_targets.ravel()
    output = output.ravel()
    I = np.argsort(output)
    M = int(sum(test_targets))  # 正类样本数
    N = int(len(output) - M)  # 负类样本数
    sigma = 0
    for i in range(M + N - 1, -1, -1):
        if test_targets[I[i]] == 1:
            sigma += i + 1  # 正类样本rank相加
    result = (sigma - (M + 1) * M / 2) / (M * N)
    return result
