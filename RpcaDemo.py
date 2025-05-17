from RpcaCode import *
import scipy.io as scio
import scipy.io
from SimulatedData import *
# generate data
n1 = 10
n2 = 10
n3 = 100
size = [n1,n2,n3]

rk = 10
sr = 0.1
k_num = round(sr*n1*n2*n2)
outer_matrix = torch.FloatTensor(generate_L(n1, n2, n3, rk))
sparse_matrix = torch.FloatTensor(generate_S(n1*n2, n3, k_num))
X = outer_matrix + sparse_matrix
X_in = X.clone().cuda().detach()
max_iter = 1001
lr_rate = 1e-3
error_list,errs_list, intin_list, finin_list, times_list = GetAllRpca(X_in,sparse_matrix, outer_matrix,size,rk,max_iter, lr_rate)
method_len = 5
print('The result is:\n')
for it in range(method_len):
    errL = error_list[it]
    errS = errs_list[it]
    inIn = intin_list[it]
    fiIn = finin_list[it]
    time = times_list[it]
    print('[%d/%d], errL=%.4f, errS=%.4f, inIn=%.2f, fiIn=%.2f, time=%.2f\n' %(it+1, method_len, errL, errS, inIn, fiIn, time))