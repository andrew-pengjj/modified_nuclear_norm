import numpy as np
import matplotlib.pyplot as plt
import torch
def min_max_normalize(data):
    """
    按最大最小值进行归一化
    :param data: 需要归一化的数据，可以是一维或二维数组
    :return: 归一化后的数据
    """
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def GenerateSparse(row, col, k_num):
    outer_matrix = np.zeros((row, col))
    rand_index = np.random.permutation(row * col)
    choose_index = rand_index[:k_num]
    for i in range(len(choose_index)):
        if np.random.rand() >= 0.5:
            outer_matrix.flat[choose_index[i]] = 1
        else:
            outer_matrix.flat[choose_index[i]] = -1
    return outer_matrix

def GenerateSalty(input_data, slevel):
    # 确定噪声的形状
    noise_shape = input_data.shape
    noise = (torch.rand(*noise_shape) < slevel).float()
    outer_matrix = input_data + noise * (1 - input_data)
    return outer_matrix

def GenerateGaussian(input_data, glevel):
    # 生成高斯随机数
    noise = torch.randn_like(input) * glevel
    outer_matrix = input_data + noise
    return outer_matrix

def GetNoise(original_data, glevel=0, slevel=0):
    # 如果 gaussian_level 和 sparse_level 均为标量，则转换为与通道数相同的数组
    # 添加 Salt & Pepper 噪声
    outer_matrix = original_data
    # 添加高斯噪声
    if glevel != 0:
        outer_matrix = GenerateGaussian(outer_matrix, glevel)
    # 添加稀疏噪声
    if slevel != 0:
        outer_matrix = GenerateSalty(outer_matrix, slevel)
    return outer_matrix


def RandomSample(n1,n2,samratio):
    mask = np.zeros([n1,n2])
    mask[np.random.rand(n1,n2) <= samratio] = 1
    return mask

def generate_U(n1, n2, rk):
    mask = np.zeros((n1, n2), dtype=int)
    class_size = rk
    coef_tensor = np.zeros((n1, n2, rk))
    support = np.random.normal(0, 1/rk, (class_size, rk))
    init_center = np.random.permutation(n1 * n2)[:class_size]
    center_axis = np.zeros((class_size, 2))
    for j in range(class_size):
        row_id = init_center[j] % n1
        col_id = (init_center[j] - row_id) // n1 + 1
        center_axis[j, 0] = row_id
        center_axis[j, 1] = col_id
    for i in range(n1):
        for j in range(n2):
            a = np.tile(np.array([i + 1, j + 1]), (class_size, 1))
            dist = np.sum((a - center_axis) ** 2, axis=1)
            c_id = np.argmin(dist)
            mask[i, j] = c_id
            coef_tensor[i, j, :] = support[c_id, :]

    coef_matrix = np.reshape(coef_tensor, (n1 * n2, rk))
    return mask, coef_matrix

def generate_L(h, w, band, rk):
    mask, U = generate_U(h, w, rk)
    V = np.random.normal(0, 1/rk, (band, rk))
    data = np.dot(U, V.T)
    #data = min_max_normalize(data)
    return data

def generate_S(row, col, k_num):
    outer_matrix = np.zeros((row, col))
    rand_index = np.random.permutation(row * col)
    choose_index = rand_index[:k_num]
    for i in range(len(choose_index)):
        if np.random.rand() >= 0.5:
            outer_matrix.flat[choose_index[i]] = 1
        else:
            outer_matrix.flat[choose_index[i]] = -1
    return outer_matrix


# def get_incoherence(matrix, rk):
#     n1, n2 = matrix.shape
#     # 对矩阵进行奇异值分解
#     U, S, V = torch.linalg.svd(matrix, full_matrices=False)

#     # 计算左右奇异值矩阵中绝对值的最大值
#     max_U_abs = torch.abs(U).max().item()
#     max_V_abs = torch.abs(V).max().item()
#     # 计算左右奇异值矩阵乘积中的绝对值的最大值
#     UVT = U @ V  # 计算 U 和 V^T 的乘积
#     max_UVT_abs = torch.abs(UVT).max().item()
#     Umax = max_U_abs*n1/rk
#     Vmax = max_V_abs*n2/rk
#     UVmax = pow(max_UVT_abs,2)*n1*n2/rk
#     small_incoherence = max(Umax, Vmax, UVmax)
#     return small_incoherence

def get_incoherence(matrix, rk):
    n1, n2 = matrix.shape
    
    # 使用 torch.linalg.svd 对矩阵进行奇异值分解
    U, S, Vh = torch.svd(matrix)
    
    # 计算左右奇异值矩阵中绝对值的最大值
    max_U_abs = torch.abs(U).max()
    max_Vh_abs = torch.abs(Vh).max()
    
    # 计算左右奇异值矩阵乘积中的绝对值的最大值
    UVT = U @ Vh.T  # 计算 U 和 V^T 的乘积
    max_UVT_abs = torch.abs(UVT).max()
    
    # 计算 Umax、Vmax、UVmax
    Umax = max_U_abs * n1 / rk
    Vmax = max_Vh_abs * n2 / rk
    UVmax = max_UVT_abs ** 2 * n1 * n2 / rk
    
    # 计算 small_incoherence
    small_incoherence = torch.max(torch.stack([Umax, Vmax, UVmax]))
    
    return small_incoherence


# def get_incoherence(matrix, rk):
#     n1, n2 = matrix.shape
#     U, S, V = torch.linalg.svd(matrix, full_matrices=False) # U*S*V
#     # 计算左右奇异向量与标准单位向量的内积
#     inner_product_U = torch.matmul(U.T, U)
#     Unorm = torch.norm(inner_product_U, p=2, dim=1)
#     Umax  = torch.max(Unorm)*n1/rk
#     inner_product_V = torch.matmul(V, V.T)
#     Vnorm = torch.norm(inner_product_V, p=2, dim=1)
#     Vmax  = torch.max(Vnorm)*n2/rk
#     UVmax = pow(torch.max(torch.abs(torch.matmul(U, V))),2)*n1*n2/rk
#     #print("incoherence:", Umax, Vmax, UVmax)
#     small_incoherence = max(Umax, Vmax, UVmax)
#     return small_incoherence

# #使用示例
# n1 = 100
# n2 = 100
# n3 = 100
# rk = 10
# generated_data = generate_L(n1, n2, n3, rk)
# matrix = np.reshape(generated_data[:, 1], [n1, n2])
# # 使用imshow显示矩阵
# plt.imshow(matrix, cmap='viridis', interpolation='nearest', origin='upper')
# # 添加颜色条
# plt.colorbar()
# # 添加标题
# plt.title('Matrix imshow Example')
# # 显示图像
# plt.show()