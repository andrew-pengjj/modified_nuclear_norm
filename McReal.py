from McCode import *
import scipy.io as scio
import scipy.io
from SimulatedData import *
from AddNoise import *
# 读取 MATLAB 文件
filepath = 'chest_pet.mat'
sr = 0.2
rk = 10
data = scipy.io.loadmat(filepath)
cln_hsi = data['data']#.astype(np.float32)
size = cln_hsi.shape
n1 = size[0]
n2 = size[1]
n3 = size[2]
mask = Mask_R(cln_hsi, sr)
mask = torch.FloatTensor(mask)
#mask = torch.FloatTensor(RandomSample(n1*n2, n3, sr))
outer_matrix = torch.FloatTensor(cln_hsi).reshape(n1*n2, n3)
X_in = torch.FloatTensor(outer_matrix*mask).reshape(n1*n2,n3).cuda()
mask = mask.cuda()
max_iter = 3001 #2001
lr_rate = 1e-3# 1e-3
psnr_list, ssim_list, ergas_list, intin_list, finin_list, times_list = GetAllMcReal(X_in, mask, outer_matrix, size, sr, rk, max_iter, lr_rate)
method_len = 5
print('The result is:\n')
for it in range(method_len):
    psnr = psnr_list[it]
    ssim = ssim_list[it]
    erg  = ergas_list[it]
    inIn = intin_list[it]
    fiIn = finin_list[it]
    time = times_list[it]
    print('[%d/%d], psnr=%.4f, ssim=%.4f, erg = %.4f, inIn=%.2f, fiIn=%.2f, time=%.2f\n' %(it+1, method_len, psnr, ssim, erg, inIn, fiIn, time))