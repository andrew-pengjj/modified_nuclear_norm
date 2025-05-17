from ConvList import *
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from AddNoise import *
def MNN_MC(sizeD, input_kernel, X_in, outer_matrix, mask, sr, max_iter, lr_rate, Is_show):
    sigma = 1e-6
    mu = 1*(math.sqrt(sizeD[0] * sizeD[1])+ math.sqrt(sizeD[2])) * math.sqrt(sr) * sigma
    normD = torch.norm(outer_matrix, p='fro')
    model = ConvNetHsi(sizeD, input_kernel).cuda()
    params = []
    params += [x for x in model.parameters()]
    lr_ = lr_rate  # 1e-4,fixed
    flag = 0
    #optimizer = optim.SGD(params, lr=lr_rate, momentum=0.1)
    optimizer = optim.Adam(params, lr=lr_, weight_decay=1e-8)
    #optimizer = optim.Adam(params, lr=lr_)
    start = time.time()
    errX_best = 1.0
    err_old = 1.0
    change_iter = 500
    loss_list = []
    errX_list = []
    for iteri in range(max_iter):
        #if flag == 0 and iteri > change_iter:
            # lr_ = 0.1 * lr_
            # optimizer = optim.Adam(params, lr=lr_, weight_decay=1e-8)  # 0.0001
            # #optimizer = optim.Adam(params, lr=lr_)
            # flag = 1
        X_Out = model()
        #loss = mu * torch.norm(X_Out, 'nuc') + 0.5 * torch.pow(torch.norm(X_in * mask - model.L * mask, 'fro'), 2)
        loss = mu * torch.norm(X_Out, 'nuc') + 0.5 * torch.pow(torch.norm(X_in * mask - model.L * mask, 'fro'), 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteri % 50 == 0:
            errX = torch.norm(model.L.cpu() - outer_matrix, p='fro') / normD
            loss_list.append(errX.item())
            errX_list.append(loss.detach().cpu().data)
            # if (errX - err_old) / err_old <= 1e-10:
            #     break
            # else:
            #     err_old = errX
            if errX < errX_best:
                errX_best = errX.item()
            if Is_show == 1:
                print('MNN iteration: %d, errX: %.4f, errX_best: %.4f, loss: %.4f' % (
                iteri, errX.item(), errX_best, loss.detach().cpu().data))
    end = time.time()
    print('MNN runtime: ', end - start)
    torch.cuda.empty_cache()
    return model.L.cpu(), X_Out, errX_best, end - start, model, loss_list, errX_list

def MNN_MC_Two(sizeD, input_kernel1, input_kernel2, X_in, outer_matrix, mask, sr, max_iter, lr_rate, Is_show):
    sigma = 1e-4
    mu = (math.sqrt(sizeD[0] * sizeD[1])+ math.sqrt(sizeD[2])) * math.sqrt(sr) * sigma

    normD = torch.norm(outer_matrix, p='fro')
    model = ConvNetHsiTwo(sizeD, input_kernel1, input_kernel2).cuda()
    params = []
    params += [x for x in model.parameters()]
    lr_ = lr_rate  # 1e-4,fixed
    flag = 0
    #optimizer = optim.SGD(params, lr=lr_rate, momentum=0.1)
    optimizer = optim.Adam(params, lr=lr_, weight_decay=1e-10)
    #optimizer = optim.Adam(params, lr=lr_)
    start = time.time()
    errX_best = 1.0
    err_old = 1.0
    change_iter = 4000
    for iteri in range(max_iter):
        if flag == 0 and iteri > change_iter:
            lr_ = 0.1 * lr_
            optimizer = optim.Adam(params, lr=lr_, weight_decay=10e-10)  # 0.0001
            #optimizer = optim.Adam(params, lr=lr_)
            flag = 1
        X_Out1, X_Out2 = model()
        #loss = mu * torch.norm(X_Out, 'nuc') + 0.5 * torch.pow(torch.norm(X_in * mask - model.L * mask, 'fro'), 2)
        loss = mu * (torch.norm(X_Out1, 'nuc')+torch.norm(X_Out2, 'nuc')) + 1 * torch.pow(torch.norm(X_in * mask - model.L * mask, 'fro'),1.5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteri % 100 == 0:
            errX = torch.norm(model.L.cpu() - outer_matrix, p='fro') / normD
            # if (errX - err_old) / err_old <= 1e-10:
            #     break
            # else:
            #     err_old = errX
            if errX < errX_best:
                errX_best = errX.item()
            if Is_show == 1:
                print('MNN iteration: %d, errX: %.4f, errX_best: %.4f, loss: %.4f' % (
                iteri, errX.item(), errX_best, loss.detach().cpu().data))
    end = time.time()
    print('MNN runtime: ', end - start)
    torch.cuda.empty_cache()
    return model.L.cpu(), X_Out1, errX_best, end - start, model


def GetAllMc(X_in, mask, outer_matrix, size, sr, rk, max_iter, lr_rate):
    Is_show = 1
    normD = torch.norm(outer_matrix, p='fro')
    method_len = 5
    error_list = torch.zeros(method_len, 1)
    intin_list = torch.zeros(method_len, 1)
    finin_list = torch.zeros(method_len, 1)
    times_list = torch.zeros(method_len, 1)

    it = 0
    input_kernel = storage.matrix1
    rec, mnn_data, errX, times, model, loss_list, errX_list = MNN_MC(size, input_kernel, X_in, outer_matrix, mask, sr, max_iter, lr_rate,
                                               Is_show)
    error_list[it] = torch.norm(rec.cpu() - outer_matrix, p='fro') / normD
    finin_list[it] = get_incoherence_new(mnn_data, rk)
    intin_list[it] = get_init_inco(outer_matrix, rk, model, size)
    times_list[it] = times
    scipy.io.savemat('mc_evaluate_1.mat', {'Loss_all': loss_list, 'ErrX_all': errX_list})

    it = 1
    input_kernel = storage.matrix2
    rec, mnn_data, errX, times, model, loss_list, errX_list = MNN_MC(size, input_kernel, X_in, outer_matrix, mask, sr, max_iter, lr_rate,
                                               Is_show)
    error_list[it] = torch.norm(rec.cpu() - outer_matrix, p='fro') / normD
    finin_list[it] = get_incoherence_new(mnn_data, rk)
    intin_list[it] = get_init_inco(outer_matrix, rk, model, size)
    times_list[it] = times
    scipy.io.savemat('mc_evaluate_2.mat', {'Loss_all': loss_list, 'ErrX_all': errX_list})

    it = 2
    input_kernel = storage.matrix3
    rec, mnn_data, errX, times, model, loss_list, errX_list = MNN_MC(size, input_kernel, X_in, outer_matrix, mask, sr, max_iter, lr_rate,
                                               Is_show)
    error_list[it] = torch.norm(rec.cpu() - outer_matrix, p='fro') / normD
    finin_list[it] = get_incoherence_new(mnn_data, rk)
    intin_list[it] = get_init_inco(outer_matrix, rk, model, size)
    times_list[it] = times
    scipy.io.savemat('mc_evaluate_3.mat', {'Loss_all': loss_list, 'ErrX_all': errX_list})

    it = 3
    input_kernel = storage.matrix4
    rec, mnn_data, errX, times, model, loss_list, errX_list = MNN_MC(size, input_kernel, X_in, outer_matrix, mask, sr, max_iter, lr_rate,
                                               Is_show)
    error_list[it] = torch.norm(rec.cpu() - outer_matrix, p='fro') / normD
    finin_list[it] = get_incoherence_new(mnn_data, rk)
    intin_list[it] = get_init_inco(outer_matrix, rk, model, size)
    times_list[it] = times
    scipy.io.savemat('mc_evaluate_4.mat', {'Loss_all': loss_list, 'ErrX_all': errX_list})

    it = 4
    input_kernel = storage.matrix5
    rec, mnn_data, errX, times, model, loss_list, errX_list = MNN_MC(size, input_kernel, X_in, outer_matrix, mask, sr,  max_iter, lr_rate, Is_show)
    error_list[it] = torch.norm(rec.cpu() - outer_matrix, p='fro') / normD
    finin_list[it] = get_incoherence_new(mnn_data, rk)
    intin_list[it] = get_init_inco(outer_matrix, rk, model, size)
    times_list[it] = times
    scipy.io.savemat('mc_evaluate_5.mat', {'Loss_all': loss_list, 'ErrX_all': errX_list})
    del model
    torch.cuda.empty_cache()
    return error_list, intin_list, finin_list, times_list

def GetAllMcReal(X_in, mask, outer_matrix, size, sr, rk, max_iter, lr_rate):
    real_data = outer_matrix.reshape(size).numpy().astype(np.float32)
    Is_show = 1
    method_len = 5
    psnr = torch.zeros(method_len, 1)
    ssim = torch.zeros(method_len, 1)
    ergas = torch.zeros(method_len, 1)
    intin_list = torch.zeros(method_len, 1)
    finin_list = torch.zeros(method_len, 1)
    times_list = torch.zeros(method_len, 1)

    it = 0
    input_kernel = storage.matrix1
    rec, mnn_data, errX, times, model, _, _ = MNN_MC(size, input_kernel, X_in, outer_matrix, mask, sr, max_iter,
                                                     lr_rate, Is_show)
    mspnr = 0
    mssim = 0
    rec_hsi = rec.reshape(size).detach().numpy().astype(np.float32)
    for ind in range(size[2]):
        mspnr += PSNR(real_data[:, :, ind], rec_hsi[:, :, ind], data_range=1.0) / size[2]
        mssim += SSIM(real_data[:, :, ind], rec_hsi[:, :, ind], ) / size[2]
    psnr[it] = mspnr
    ssim[it] = mssim
    ergas[it] = ErrRelGlobAdimSyn(255 * real_data, 255 * rec_hsi)
    finin_list[it] = get_incoherence_new(mnn_data, rk)
    intin_list[it] = get_init_inco(outer_matrix, rk, model, size)
    times_list[it] = times

    it = 1
    input_kernel = storage.matrix2
    rec, mnn_data, errX, times, model,_,_ = MNN_MC(size, input_kernel, X_in, outer_matrix, mask, sr, max_iter, lr_rate, Is_show)
    mspnr = 0
    mssim = 0
    rec_hsi = rec.reshape(size).detach().numpy().astype(np.float32)
    for ind in range(size[2]):
        mspnr += PSNR(real_data[:, :, ind], rec_hsi[:, :, ind], data_range=1.0) / size[2]
        mssim += SSIM(real_data[:, :, ind], rec_hsi[:, :, ind], ) / size[2]
    psnr[it] = mspnr
    ssim[it] = mssim
    ergas[it] = ErrRelGlobAdimSyn(255 * real_data, 255 * rec_hsi)
    finin_list[it] = get_incoherence_new(mnn_data, rk)
    intin_list[it] = get_init_inco(outer_matrix, rk, model, size)
    times_list[it] = times

    it = 2
    input_kernel = storage.matrix3
    rec, mnn_data, errX, times, model, _, _ = MNN_MC(size, input_kernel, X_in, outer_matrix, mask, sr, max_iter,
                                                     lr_rate, Is_show)
    mspnr = 0
    mssim = 0
    rec_hsi = rec.reshape(size).detach().numpy().astype(np.float32)
    for ind in range(size[2]):
        mspnr += PSNR(real_data[:, :, ind], rec_hsi[:, :, ind], data_range=1.0) / size[2]
        mssim += SSIM(real_data[:, :, ind], rec_hsi[:, :, ind], ) / size[2]
    psnr[it] = mspnr
    ssim[it] = mssim
    ergas[it] = ErrRelGlobAdimSyn(255 * real_data, 255 * rec_hsi)
    finin_list[it] = get_incoherence_new(mnn_data, rk)
    intin_list[it] = get_init_inco(outer_matrix, rk, model, size)
    times_list[it] = times

    it = 3
    input_kernel = storage.matrix4
    rec, mnn_data, errX, times, model,_,_ = MNN_MC(size, input_kernel, X_in, outer_matrix, mask, sr, max_iter, lr_rate,
                                               Is_show)
    mspnr = 0
    mssim = 0
    rec_hsi = rec.reshape(size).detach().numpy().astype(np.float32)
    for ind in range(size[2]):
        mspnr += PSNR(real_data[:, :, ind], rec_hsi[:, :, ind], data_range=1.0) / size[2]
        mssim += SSIM(real_data[:, :, ind], rec_hsi[:, :, ind], ) / size[2]
    psnr[it] = mspnr
    ssim[it] = mssim
    ergas[it] = ErrRelGlobAdimSyn(255 * real_data, 255 * rec_hsi)
    finin_list[it] = get_incoherence_new(mnn_data, rk)
    intin_list[it] = get_init_inco(outer_matrix, rk, model, size)
    times_list[it] = times

    it = 4
    input_kernel = storage.matrix5
    rec, mnn_data, errX, times, model, _, _ = MNN_MC(size, input_kernel, X_in, outer_matrix, mask, sr, max_iter,
                                                     lr_rate,
                                                     Is_show)
    mspnr = 0
    mssim = 0
    rec_hsi = rec.reshape(size).detach().numpy().astype(np.float32)
    for ind in range(size[2]):
        mspnr += PSNR(real_data[:, :, ind], rec_hsi[:, :, ind], data_range=1.0) / size[2]
        mssim += SSIM(real_data[:, :, ind], rec_hsi[:, :, ind], ) / size[2]
    psnr[it] = mspnr
    ssim[it] = mssim
    ergas[it] = ErrRelGlobAdimSyn(255 * real_data, 255 * rec_hsi)
    finin_list[it] = get_incoherence_new(mnn_data, rk)
    intin_list[it] = get_init_inco(outer_matrix, rk, model, size)
    times_list[it] = times

    del model
    torch.cuda.empty_cache()
    return psnr, ssim, ergas, intin_list, finin_list, times_list


