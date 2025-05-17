# 🌈 Modified Nuclear Norm

🧠 **MNN code** published in 🏅 **IJCAI**.

📄 **Theoretical proofs** and more **experimental data** are placed in the 📦 **Supplemental Material** file!

## 📊 Results on Simulated RPCA tasks

Run `python RpcaDemo.py`. The following table summarizes the denoising performance under a certain experiment:

| Method | Description                         | ErrorL   | ErrorS   | TIME (s) |
|--------|-------------------------------------|---------|---------|----------|
| NN  | Nuclear Norm                          |  0.0237 | 0.0077  |   7.85   |
| MNN-Diff   | First-Order Difference  | 0.0186 | 0.0060  | 5.48   |
| MNN-Sobel   | Sobel Operator | 0.0011 | 0.0004  |  5.52   |
| MNN-Lap1    | Laplacian type 1   | 0.0006 | 0.0002  |  4.83   |
| MNN-Lap2    | Laplacian type 2   | 0.0100 | 0.0032  | 4.86   |

---

## 📊 Results on Simulated MC tasks

Run `python McDemo.py`. The following table summarizes the denoising performance under a certain experiment:

| Method | Description                         | ErrorL   |TIME (s) |
|--------|-------------------------------------|---------|-----------|
| NN  | Nuclear Norm                          |  0.0039 |   8.19   |
| MNN-Diff   | First-Order Difference  | 0.0038 |5.38   |
| MNN-Sobel   | Sobel Operator | 0.0021 |   5.42   |
| MNN-Lap1    | Laplacian type 1   | 0.0020 |  4.86   |
| MNN-Lap2    | Laplacian type 2   | 0.0023 |  4.83   |

---

## 📊 Results on chest_pet Data (MC)

Run `python McReal.m`. The following table summarizes the denoising performance:

| Method | Description                         | MPSNR   | MSSIM   | ERGAS   | TIME (s) |
|--------|-------------------------------------|---------|---------|---------|----------|
| NN  | Nuclear Norm                           |  22.7923 | 0.6767  | 242.6524 |   61.74   |
| MNN-Diff    | First-Order Difference                 | 29.8989 | 0.9241  |  107.8069 |  58.38   |
| MNN-Sobel   | Sobel Operator | 33.3498 | 0.9648  |  69.3290 |  59.83   |
| MNN-Lap1    | Laplacian type 1          | 34.6774 | 0.9726  |  59.2194 |  58.32   |
| MNN-Lap2    | Laplacian type 2   | 34.2451 | 0.9707  |  62.1497 | 58.38   |

---

