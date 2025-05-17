from RpcaCode import *
import scipy.io as scio
import scipy.io
from SimulatedData import *
from AddNoise import *
# read MATLAB file
filepath = 'new_mobile.mat'
print(filepath)
rk = 10

lr_rate = 1e-3
slevel = 0.7
max_iter = 5001
cln_hsi, noi_hsi = add_sp_noise(filepath, slevel)
size = cln_hsi.shape
n1 = size[0]
n2 = size[1]
n3 = size[2]
outer_matrix = torch.FloatTensor(cln_hsi).reshape(n1*n2, n3)
X_in = torch.FloatTensor(noi_hsi).reshape(n1*n2,n3).cuda()
psnr_list, ssim_list, erg_list, intin_list, finin_list, times_list = GetAllRpcaReal(X_in, outer_matrix, size, rk,
                                                                                    max_iter, lr_rate)


method_len = 4
print('The result is:\n')
print(psnr_list)
print(ssim_list)
print(erg_list)
print(times_list)