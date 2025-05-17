from ConvList import *
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from AddNoise import *
import numpy as np
def MNN_RPCA(sizeD, input_kernel, X_in, outer_matrix, max_iter, lr_rate,Is_show):
    lamb = 1 / math.sqrt(sizeD[0] * sizeD[1])
    normD = torch.norm(outer_matrix, p='fro')
    model = ConvNetHsi(sizeD, input_kernel).cuda()
    params = []
    params += [x for x in model.parameters()]
    lr_ = lr_rate  # 1e-4,fixed
    flag = 0
    optimizer = optim.Adam(params, lr=lr_, weight_decay=1e-8)
    start = time.time()
    errX_best = 1.0
    incount = 0
    loss_list = []
    errX_list = []
    for iteri in range(max_iter):
        if flag == 0 and iteri > round(min(0.5*max_iter,500)):
            lr_ = 0.1 * lr_
            optimizer = optim.Adam(params, lr=lr_, weight_decay=10e-8)  # 0.0001
            flag = 1
        X_Out = model()
        loss = torch.norm(X_Out, 'nuc') + lamb * torch.norm(X_in - model.L, 1)
        #loss = torch.norm(model.L, 'nuc') + lamb * torch.norm(X_in - X_Out, 1)
        # loss = incoh
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteri % 50 == 0:
            errX = torch.norm(model.L.cpu() - outer_matrix, p='fro') / normD
            loss_list.append(errX.item())
            errX_list.append(loss.detach().cpu().data)
            #errX = torch.norm(X_Out.cpu() - outer_matrix, p='fro') / normD
            if errX < errX_best:
                errX_best = errX.item()
                incount
            if Is_show == 1:
                print('MNN iteration: %d, errX: %.4f, errX_best: %.4f, loss: %.4f' % (
                iteri, errX.item(), errX_best, loss.detach().cpu().data))
    end = time.time()
    print('MNN runtime: ', end - start)
    torch.cuda.empty_cache()
    return model.L.cpu(), X_Out, errX_best, end - start, model, loss_list, errX_list

def MNN_RPCA_Fore(sizeD, input_kernel, X_in, outer_matrix, max_iter, lr_rate,lamb, Is_show):
    normD = torch.norm(outer_matrix, p='fro')
    model = ConvNetHsi(sizeD, input_kernel).cuda()
    params = []
    params += [x for x in model.parameters()]
    lr_ = lr_rate  # 1e-4,fixed
    flag = 0
    optimizer = optim.Adam(params, lr=lr_, weight_decay=1e-8)
    start = time.time()
    errX_best = 1.0
    incount = 0
    for iteri in range(max_iter):
        if flag == 0 and iteri > round(min(0.5*max_iter,500)):
            lr_ = 0.1 * lr_
            optimizer = optim.Adam(params, lr=lr_, weight_decay=10e-8)  # 0.0001
            flag = 1
        X_Out = model()
        loss = torch.norm(X_Out, 'nuc') + lamb * torch.norm(X_in - model.L, 1)
        #loss = torch.norm(model.L, 'nuc') + lamb * torch.norm(X_in - X_Out, 1)
        # loss = incoh
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteri % 100 == 0:
            errX = torch.norm(model.L.cpu() - outer_matrix, p='fro') / normD
            #errX = torch.norm(X_Out.cpu() - outer_matrix, p='fro') / normD
            if errX < errX_best:
                errX_best = errX.item()
                incount
            if Is_show == 1:
                print('MNN iteration: %d, errX: %.4f, errX_best: %.4f, loss: %.4f' % (
                iteri, errX.item(), errX_best, loss.detach().cpu().data))
    end = time.time()
    print('MNN runtime: ', end - start)
    torch.cuda.empty_cache()
    return model.L.cpu(), X_Out, errX_best, end - start, model


def GetAllRpca(X_in, sparse_matrix, outer_matrix, size, rk, max_iter, lr_rate):
    Is_show = 1
    normD = torch.norm(outer_matrix, p='fro')
    normS = torch.norm(sparse_matrix, p='fro')
    error_list = torch.zeros(5, 1)
    errs_list = torch.zeros(5, 1)
    intin_list = torch.zeros(5, 1)
    finin_list = torch.zeros(5, 1)
    times_list = torch.zeros(5, 1)
    it = 0
    input_kernel = storage.matrix1
    rec, mnn_data, errX, times, model, loss_list, errX_list = MNN_RPCA(size, input_kernel, X_in, outer_matrix, max_iter, lr_rate,Is_show)
    error_list[it] = torch.norm(rec.cpu() - outer_matrix, p='fro') / normD
    errs_list[it] = torch.norm(X_in.cpu() - rec.cpu() - sparse_matrix, p='fro') / normS
    finin_list[it] = get_incoherence_new(mnn_data, rk)
    intin_list[it] = get_init_inco(outer_matrix, rk, model, size)
    times_list[it] = times
    scipy.io.savemat('rpca_evaluate_1.mat', {'Loss_all': loss_list, 'ErrX_all': errX_list})


    it = 1
    input_kernel = storage.matrix2
    rec, mnn_data, errX, times, model, loss_list, errX_list = MNN_RPCA(size, input_kernel, X_in, outer_matrix, max_iter, lr_rate, Is_show)
    error_list[it] = torch.norm(rec.cpu() - outer_matrix, p='fro') / normD
    errs_list[it] = torch.norm(X_in.cpu() - rec.cpu() - sparse_matrix, p='fro') / normS
    finin_list[it] = get_incoherence_new(mnn_data, rk)
    intin_list[it] = get_init_inco(outer_matrix, rk, model, size)
    times_list[it] = times
    scipy.io.savemat('rpca_evaluate_2.mat', {'Loss_all': loss_list, 'ErrX_all': errX_list})

    it = 2
    input_kernel = storage.matrix3
    rec, mnn_data, errX, times, model, loss_list, errX_list = MNN_RPCA(size, input_kernel, X_in, outer_matrix, max_iter, lr_rate, Is_show)
    error_list[it] = torch.norm(rec.cpu() - outer_matrix, p='fro') / normD
    errs_list[it] = torch.norm(X_in.cpu() - rec.cpu() - sparse_matrix, p='fro') / normS
    finin_list[it] = get_incoherence_new(mnn_data, rk)
    intin_list[it] = get_init_inco(outer_matrix, rk, model, size)
    times_list[it] = times
    scipy.io.savemat('rpca_evaluate_3.mat', {'Loss_all': loss_list, 'ErrX_all': errX_list})

    it = 3
    input_kernel = storage.matrix4
    rec, mnn_data, errX, times, model, loss_list, errX_list = MNN_RPCA(size, input_kernel, X_in, outer_matrix, max_iter, lr_rate, Is_show)
    error_list[it] = torch.norm(rec.cpu() - outer_matrix, p='fro') / normD
    errs_list[it] = torch.norm(X_in.cpu() - rec.cpu() - sparse_matrix, p='fro') / normS
    finin_list[it] = get_incoherence_new(mnn_data, rk)
    intin_list[it] = get_init_inco(outer_matrix, rk, model, size)
    times_list[it] = times
    scipy.io.savemat('rpca_evaluate_4.mat', {'Loss_all': loss_list, 'ErrX_all': errX_list})

    it = 4
    input_kernel = storage.matrix5
    rec, mnn_data, errX, times, model, loss_list, errX_list = MNN_RPCA(size, input_kernel, X_in, outer_matrix, max_iter, lr_rate, Is_show)
    error_list[it] = torch.norm(rec.cpu() - outer_matrix, p='fro') / normD
    errs_list[it] = torch.norm(X_in.cpu() - rec.cpu() - sparse_matrix, p='fro') / normS
    finin_list[it] = get_incoherence_new(mnn_data, rk)
    intin_list[it] = get_init_inco(outer_matrix, rk, model, size)
    times_list[it] = times
    scipy.io.savemat('rpca_evaluate_5.mat', {'Loss_all': loss_list, 'ErrX_all': errX_list})

    del model
    torch.cuda.empty_cache()
    return error_list, errs_list, intin_list, finin_list, times_list


def GetAllRpcaReal(X_in, outer_matrix, size, rk, max_iter, lr_rate):
    real_data = outer_matrix.reshape(size).numpy().astype(np.float32)
    Is_show = 1
    psnr = torch.zeros(4, 1)
    ssim = torch.zeros(4, 1)
    ergas = torch.zeros(4,1)
    intin_list = torch.zeros(4, 1)
    finin_list = torch.zeros(4, 1)
    times_list = torch.zeros(4, 1)

    it = 0
    input_kernel = storage.matrix5
    rec, mnn_data, errX, times, model, _, _ = MNN_RPCA(size, input_kernel, X_in, outer_matrix, max_iter, lr_rate,
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
    scipy.io.savemat('mobile_mnn1.mat', {'data': rec_hsi})
    del model
    torch.cuda.empty_cache()

    it = 1
    input_kernel = storage.matrix3
    rec, mnn_data, errX, times, model, _, _ = MNN_RPCA(size, input_kernel, X_in, outer_matrix, max_iter, lr_rate,
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
    scipy.io.savemat('mobile_mnn2.mat', {'data': rec_hsi})
    del model
    torch.cuda.empty_cache()

    it = 2
    input_kernel = storage.matrix4
    rec, mnn_data, errX, times, model, _, _ = MNN_RPCA(size, input_kernel, X_in, outer_matrix, max_iter, lr_rate,
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
    scipy.io.savemat('mobile_mnn3.mat', {'data': rec_hsi})
    del model
    torch.cuda.empty_cache()

    it = 3
    input_kernel = storage.matrix5
    rec, mnn_data, errX, times, model, _, _ = MNN_RPCA(size, input_kernel, X_in, outer_matrix, max_iter, lr_rate, Is_show)
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
    scipy.io.savemat('mobile_mnn4.mat', {'data': rec_hsi})
    del model
    torch.cuda.empty_cache()
    return psnr, ssim, ergas, intin_list, finin_list, times_list

def ForeExtract(X_in, outer_matrix, size, rk, max_iter, lr_rate, lamb):
    real_data = outer_matrix.reshape(size).numpy().astype(np.float32)
    Is_show = 1
    mauc = torch.zeros(4, 1)
    intin_list = torch.zeros(4, 1)
    finin_list = torch.zeros(4, 1)
    times_list = torch.zeros(4, 1)
    it = 0
    input_kernel = storage.matrix2
    rec, mnn_data, errX, times, model = MNN_RPCA_Fore(size, input_kernel, X_in, outer_matrix, max_iter, lr_rate, lamb, Is_show)
    rec_hsi = (X_in.cpu()-rec).reshape(size).detach().numpy().astype(np.float32)
    auc = 0
    for ind in range(size[2]):
        auc += CAUC(real_data[:, :, ind], abs(rec_hsi[:, :, ind]))
    mauc[it] = auc/size[2]
    finin_list[it] = get_incoherence_new(mnn_data, rk)
    intin_list[it] = get_init_inco(outer_matrix, rk, model, size)
    times_list[it] = times

    it = 1
    input_kernel = storage.matrix3
    rec, mnn_data, errX, times, model = MNN_RPCA_Fore(size, input_kernel, X_in, outer_matrix, max_iter, lr_rate, lamb,
                                                      Is_show)
    rec_hsi = (X_in.cpu() - rec).reshape(size).detach().numpy().astype(np.float32)
    auc = 0
    for ind in range(size[2]):
        auc += CAUC(real_data[:, :, ind], abs(rec_hsi[:, :, ind]))
    mauc[it] = auc / size[2]
    finin_list[it] = get_incoherence_new(mnn_data, rk)
    intin_list[it] = get_init_inco(outer_matrix, rk, model, size)
    times_list[it] = times

    it = 2
    input_kernel = storage.matrix4
    rec, mnn_data, errX, times, model = MNN_RPCA_Fore(size, input_kernel, X_in, outer_matrix, max_iter, lr_rate, lamb,
                                                      Is_show)
    rec_hsi = (X_in.cpu() - rec).reshape(size).detach().numpy().astype(np.float32)
    auc = 0
    for ind in range(size[2]):
        auc += CAUC(real_data[:, :, ind], abs(rec_hsi[:, :, ind]))
    mauc[it] = auc / size[2]
    finin_list[it] = get_incoherence_new(mnn_data, rk)
    intin_list[it] = get_init_inco(outer_matrix, rk, model, size)
    times_list[it] = times

    it = 3
    input_kernel = storage.matrix5
    rec, mnn_data, errX, times, model = MNN_RPCA_Fore(size, input_kernel, X_in, outer_matrix, max_iter, lr_rate, lamb,
                                                      Is_show)
    rec_hsi = (X_in.cpu() - rec).reshape(size).detach().numpy().astype(np.float32)
    auc = 0
    for ind in range(size[2]):
        auc += CAUC(real_data[:, :, ind], abs(rec_hsi[:, :, ind]))
    mauc[it] = auc / size[2]
    finin_list[it] = get_incoherence_new(mnn_data, rk)
    intin_list[it] = get_init_inco(outer_matrix, rk, model, size)
    times_list[it] = times
    del model
    torch.cuda.empty_cache()
    return mauc, intin_list, finin_list, times_list
