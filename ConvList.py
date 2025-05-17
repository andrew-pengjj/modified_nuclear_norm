import torch.nn as nn
import math
import torch.optim as optim
import time
from SimulatedData import *

#torch.manual_seed(42)

# 定义一个简单的神经网络，只有一个卷积层
class SingleConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, given_kernel):
        super(SingleConvNet, self).__init__()
        # 计算 padding
        kernel_size = given_kernel.shape
        padding_height = kernel_size[0] // 2
        padding_width = kernel_size[1] // 2         
        padding = (padding_height, padding_width)
        if np.mod(kernel_size[0],2) == 1:
            self.r_s_h = 0
        else:
            self.r_s_h = 1
        if np.mod(kernel_size[1],2) == 1:
            self.r_s_w = 0
        else:
            self.r_s_w = 1
        # 定义一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.kernel_size = kernel_size
        self.conv1.weight.data = given_kernel.reshape(in_channels, out_channels,kernel_size[0],kernel_size[1])
        self.conv1.weight.requires_grad = False
        
    def forward(self, x):
        # 前向传播
        x = self.conv1(x)
        return x
    
    def normalize_weights(self):
        with torch.no_grad():
            # 获取卷积层权重
            weight = self.conv1.weight.data  
            # 计算F范数
            f_norm = torch.norm(weight)
            self.conv1.weight.data = weight / f_norm
            
class LocalStorage(nn.Module):
    def __init__(self):
        super(LocalStorage, self).__init__()
        self.matrix1 = torch.tensor([
                [1],
            ], dtype=torch.float32)
        self.matrix2 = torch.tensor([
                [-1, 1],
                #[-1], [1],
            ], dtype=torch.float32)
        self.matrix3 = torch.tensor([
                [-2, 1],
                [1, 0]
            ], dtype=torch.float32)
        self.matrix4 = torch.tensor([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0],
            ], dtype=torch.float32)
        self.matrix5 = torch.tensor([
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1],
            ], dtype=torch.float32)
        # self.matrix10 = torch.tensor([
        #         [0, -1, -1, -1, 0],
        #         [-1, -2, -4, -2, -1],
        #         [-1, -4, 20, -4, -1],
        #         [-1, -2, -4, -2, -1],
        #         [0, -1, -1, -1, 0]
        #     ], dtype=torch.float32)
        # self.matrix11 = torch.tensor([
        #         [-1, 0, 1],
        #         [-2, 0, 2],
        #         [-1, 0, 1]
        #     ], dtype=torch.float32)
        # self.matrix12 = torch.tensor([
        #         [-1, 0, 1],
        #         [-1, 0, 1],
        #         [-1, 0, 1]
        #     ], dtype=torch.float32)
    def normalize_matrices(self):
        # 计算矩阵的 F 范数
        norm_matrix1 = torch.norm(self.matrix1, p='fro')
        norm_matrix2 = torch.norm(self.matrix2, p='fro')
        norm_matrix3 = torch.norm(self.matrix3, p='fro')
        norm_matrix4 = torch.norm(self.matrix4, p='fro')
        norm_matrix5 = torch.norm(self.matrix5, p='fro')
        # norm_matrix6 = torch.norm(self.matrix6, p='fro')
        # norm_matrix7 = torch.norm(self.matrix7, p='fro')
        # norm_matrix8 = torch.norm(self.matrix8, p='fro')
        # norm_matrix9 = torch.norm(self.matrix9, p='fro')
        # norm_matrix10 = torch.norm(self.matrix10, p='fro')
        # norm_matrix11 = torch.norm(self.matrix11, p='fro')
        # norm_matrix12 = torch.norm(self.matrix12, p='fro')

        # 归一化操作
        self.matrix1 /= norm_matrix1
        self.matrix2 /= norm_matrix2
        self.matrix3 /= norm_matrix3
        self.matrix4 /= norm_matrix4
        self.matrix5 /= norm_matrix5
    def forward(self):
        # 在这里你可以对存储的矩阵进行操作
        pass
    
storage = LocalStorage()

# 执行归一化操作
storage.normalize_matrices()
               
class ConvNetHsi(nn.Module):
    def __init__(self, size, input_kernel):
        super(ConvNetHsi, self).__init__()
        self.size = size
        self.L = nn.Parameter(torch.zeros(size[0] * size[1], size[2]), requires_grad=True)
        self.conv = SingleConvNet(1, 1, input_kernel)
        ks = input_kernel.shape
        self.r_s_h = self.conv.r_s_h
        self.r_s_w = self.conv.r_s_w
        print(ks,self.r_s_h,self.r_s_w)
    def forward(self):
        out = self.conv(self.L.reshape(1, self.size[0], self.size[1],self.size[2]).permute(3,0,1,2)).permute(1, 2, 3, 0).reshape((self.size[0]+self.r_s_h) * (self.size[1]+self.r_s_w), self.size[2])
        return out

class ConvNetHsiTwo(nn.Module):
    def __init__(self, size, input_kernel1, input_kernel2):
        super(ConvNetHsiTwo, self).__init__()
        self.size = size
        self.L = nn.Parameter(torch.zeros(size[0] * size[1], size[2]), requires_grad=True)
        self.conv1 = SingleConvNet(1, 1, input_kernel1)
        self.conv2 = SingleConvNet(1, 1, input_kernel2)
        ks = input_kernel1.shape
        self.r_s_h = self.conv1.r_s_h
        self.r_s_w = self.conv1.r_s_w
        print(ks,self.r_s_h,self.r_s_w)
    def forward(self):
        out1 = self.conv1(self.L.reshape(1, self.size[0], self.size[1], self.size[2]).permute(3, 0, 1, 2)).permute(1, 2,
                                                                                                                  3,
                                                                                                                  0).reshape(
            (self.size[0] + self.r_s_h) * (self.size[1] + self.r_s_w), self.size[2])
        out2 = self.conv2(self.L.reshape(1, self.size[0], self.size[1], self.size[2]).permute(3, 0, 1, 2)).permute(1, 2,
                                                                                                                   3,
                                                                                                                   0).reshape(
            (self.size[0] + self.r_s_h) * (self.size[1] + self.r_s_w), self.size[2])


        return out1, out2
    
def get_incoherence_new(matrix, rk):
    n1, n2 = matrix.shape
    # 使用 torch.linalg.svd 对矩阵进行奇异值分解
    U, S, Vh = torch.svd(matrix)
    # 计算左右奇异值矩阵中绝对值的最大值
    max_U_abs = torch.abs(U).max()
    max_Vh_abs = torch.abs(Vh).max()
    # 计算左右奇异值矩阵乘积中的绝对值的最大值
    UVT = U @ Vh  # 计算 U 和 V^T 的乘积
    max_UVT_abs = torch.abs(UVT).max()
    # 计算 Umax、Vmax、UVmax
    Umax = max_U_abs * n1 / rk
    Vmax = max_Vh_abs * n2 / rk
    UVmax = (max_UVT_abs ** 2 * n1 * n2) / rk
    # 计算 small_incoherence
    small_incoherence = torch.max(torch.tensor([Umax, Vmax, UVmax], device=matrix.device))
    return small_incoherence

def get_conv_result(input_tensor, input_kernel):
    # 定义卷积层
    ks = input_kernel.shape
    # 计算适当的填充量以保持输出尺寸与输入尺寸一致
    padding = (ks[2] // 2, ks[3] // 2)
    conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(ks[2], ks[3]), stride=1, padding=padding, bias=False)
    conv_layer.weight.data = input_kernel
    output_matrices = []
    for i in range(input_tensor.shape[2]):
        # 获取沿第三个维度的第 i 个矩阵
        matrix = input_tensor[:, :, i].unsqueeze(0).unsqueeze(0)  # 增加批次和通道维度
        output_matrix = conv_layer(matrix)
        output_matrices.append(output_matrix.squeeze())

    # 将卷积后的结果矩阵堆叠在一起，形成一个新的 3 阶张量
    output_tensor = torch.stack(output_matrices, dim=2)
    return output_tensor

def get_init_inco(outer_matrix,rk,model,size):
    input_tensor = outer_matrix.reshape(size[0],size[1],size[2])# X_rpca,outer_matrix
    change_tensor = get_conv_result(input_tensor.cuda(), model.conv.conv1.weight.data)
    [m,n,p]=change_tensor.shape
    change_matrix = change_tensor.reshape(m*n,p)
    small_inco = get_incoherence_new(change_matrix,rk)
    return small_inco