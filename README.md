# 🌈 Modified Nuclear Norm

🧠 **MNN code** published in 🏅 **IJCAI**.

📄 **Theoretical proofs** and more **experimental data** are placed in the 📦 **Supplemental Material** file!

---

## 📊 Results on PaviaU Data (Sparse Noise)

Run `HSIDenoising_Sparse_Demo.m`. The following table summarizes the denoising performance:

| Method | Description                         | MPSNR   | MSSIM   | ERGAS   | TIME (s) |
|--------|-------------------------------------|---------|---------|---------|----------|
| Noise  | Noise Data                          |  8.5220 | 0.0460  | 1388.43 |   0.00   |
| TNN    | Tensor Nuclear Norm                 | 35.7100 | 0.9420  |   86.44 | 100.67   |
| ATNN   | Adaptive Tensor Nuclear Norm (Ours) | 41.5500 | 0.9920  |   32.42 |  38.80   |
| CTV    | Correlated Total Variation          | 43.0400 | 0.9950  |   27.38 |  56.60   |
| TCTV   | Tensor Correlated Total Variation   | 40.7400 | 0.9800  |   54.35 | 193.85   |

---

## 📊 Results on PaviaU Data (Gaussian Noise)

Run `HSIDenoising_Gaussian_Demo.m`. The following table summarizes the denoising performance:

| Method | Description                         | MPSNR   | MSSIM   | ERGAS   | TIME (s) |
|--------|-------------------------------------|---------|---------|---------|----------|
| Noise  | Noise Data                          |  9.6830 | 0.0560  | 1165.63 |   0.00   |
| TNN    | Tensor Nuclear Norm                 | 18.1000 | 0.2020  |  451.53 |  90.45   |
| ATNN   | Adaptive Tensor Nuclear Norm (Ours) | 22.1900 | 0.4210  |  279.23 |  16.32   |
| CTV    | Correlated Total Variation          | 27.1300 | 0.7390  |  162.07 |  54.86   |
| TCTV   | Tensor Correlated Total Variation   | 22.3600 | 0.4480  |  275.79 | 184.83   |

---

## 📈 Results on Simulated Data (RPCA Task)
Run `RPCA_Simulated_DEMO.m`. The following table summarizes the denoising performance:

| Method | Description                         | NMSE     | TIME (s) |
|--------|-------------------------------------|----------|----------|
| NN     | Nuclear Norm                        | 0.3487   | 0.17     |
| TNN    | Tensor Nuclear Norm                 | 0.9456   | 1.88     |
| ATNN   | Adaptive Tensor Nuclear Norm (Ours) | 0.0419   | 1.02     |
| CTV    | Correlated Total Variation          | 0.1704   | 1.03     |
| TCTV   | Tensor Correlated Total Variation   | 0.2693   | 6.08     |

---

## 📈 Results on Simulated Data (MC Task)
Run `TC_Simulated_DEMO.m`. The following table summarizes the denoising performance:

| Method | Description                         | NMSE     | TIME (s) |
|--------|-------------------------------------|----------|----------|
| NN     | Nuclear Norm                        | 0.0021   | 0.12     |
| TNN    | Tensor Nuclear Norm                 | 0.2126   | 3.48     |
| ATNN   | Adaptive Tensor Nuclear Norm (Ours) | 0.0002   | 0.80     |
| CTV    | Correlated Total Variation          | 0.0001   | 2.53     |
| TCTV   | Tensor Correlated Total Variation   | 0.1126   | 9.78     |

---
