import os
import time
import math
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import textwrap
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.special import hankel2, j1
import torch.backends.cudnn as cudnn

# =============================================================================
# 0. 全局统一配置与硬件信息
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype_c = torch.complex64
dtype_r = torch.float32

if torch.cuda.is_available():
    cudnn.benchmark = True

print(f"Executing Deterministic Benchmark (CSI vs CC-CSI vs DBIM) on: {device}")

EPS0 = 8.8541878e-12
C_SPEED = 299792458.0
GRID_SIZE = 64
PAD_MULT = 4
CALIB_NUM = 3

NUM_RUNS = 1


# =============================================================================
# 1. 共享辅助函数与评估、绘图函数
# =============================================================================
def generate_ground_truth(r_grid, shape_type):
    x, y = r_grid[:, 0], r_grid[:, 1]
    eps_true = np.ones_like(x)
    sigma_true = np.zeros_like(x)

    if shape_type == 'Austria':
        mask_c1 = (x + 0.15) ** 2 + (y - 0.3) ** 2 <= 0.1 ** 2
        mask_c2 = (x - 0.15) ** 2 + (y - 0.3) ** 2 <= 0.1 ** 2
        r_sq = x ** 2 + (y + 0.1) ** 2
        mask_ring = (r_sq >= 0.15 ** 2) & (r_sq <= 0.3 ** 2)
        eps_true[mask_c1], eps_true[mask_c2], eps_true[mask_ring] = RANGE_EPS_1, RANGE_EPS_2, RANGE_EPS_3
        sigma_true[mask_c1], sigma_true[mask_c2], sigma_true[mask_ring] = RANGE_SIG_1, RANGE_SIG_2, RANGE_SIG_3
        range_eps = max(RANGE_EPS_1, RANGE_EPS_2, RANGE_EPS_3)
        range_sig = max(RANGE_SIG_1, RANGE_SIG_2, RANGE_SIG_3)

    elif shape_type == 'FoamTwinDiel':
        mask_L = (x + 2.5e-3) ** 2 + (y - 0.0) ** 2 <= 40e-3 ** 2
        mask_Ext = (x + 58e-3) ** 2 + (y - 4e-3) ** 2 <= (31e-3 / 2) ** 2
        mask_Int = (x + 7.5e-3) ** 2 + (y - 0.0) ** 2 <= (31e-3 / 2) ** 2
        eps_true[mask_L], eps_true[mask_Int], eps_true[mask_Ext] = RANGE_EPS_1, RANGE_EPS_2, RANGE_EPS_3
        sigma_true[mask_L], sigma_true[mask_Int], sigma_true[mask_Ext] = RANGE_SIG_1, RANGE_SIG_2, RANGE_SIG_3
        range_eps = max(RANGE_EPS_1, RANGE_EPS_2, RANGE_EPS_3)
        range_sig = max(RANGE_SIG_1, RANGE_SIG_2, RANGE_SIG_3)

    elif shape_type == 'bowtie_cross':
        mask_h = (np.abs(x) <= 0.36) & (np.abs(y) <= 0.1)
        mask_top = (y >= 0) & (y <= 0.36) & (np.abs(x) <= y * (0.2 / 0.36))
        mask_bot = (y <= 0) & (y >= -0.36) & (np.abs(x) <= -y * (0.2 / 0.36))
        eps_true[mask_h], eps_true[mask_top], eps_true[mask_bot] = RANGE_EPS_1, RANGE_EPS_2, RANGE_EPS_3
        sigma_true[mask_h], sigma_true[mask_top], sigma_true[mask_bot] = RANGE_SIG_1, RANGE_SIG_2, RANGE_SIG_3
        range_eps = max(RANGE_EPS_1, RANGE_EPS_2, RANGE_EPS_3)
        range_sig = max(RANGE_SIG_1, RANGE_SIG_2, RANGE_SIG_3)
    else:
        raise ValueError("Unsupported shape_type")

    if range_sig == 0.0: range_sig = 1.0
    return eps_true.reshape(GRID_SIZE, GRID_SIZE), sigma_true.reshape(GRID_SIZE, GRID_SIZE), range_eps, range_sig


def add_gt_contours(ax, shape_type, ext):
    if shape_type == 'Austria':
        ax.add_patch(plt.Circle((-0.15, 0.3), 0.1, fill=False, linestyle='--', linewidth=1.5, edgecolor='cyan'))
        ax.add_patch(plt.Circle((0.15, 0.3), 0.1, fill=False, linestyle='--', linewidth=1.5, edgecolor='cyan'))
        ax.add_patch(plt.Circle((0, -0.1), 0.3, fill=False, linestyle='--', linewidth=1.5, edgecolor='cyan'))
        ax.add_patch(plt.Circle((0, -0.1), 0.15, fill=False, linestyle='--', linewidth=1.5, edgecolor='cyan'))
    elif shape_type == 'FoamTwinDiel':
        ax.add_patch(plt.Circle((-2.5e-3, 0.0), 40e-3, fill=False, linestyle='--', linewidth=1.5, edgecolor='cyan'))
        ax.add_patch(plt.Circle((-58e-3, 4e-3), 31e-3 / 2, fill=False, linestyle='--', linewidth=1.5, edgecolor='cyan'))
        ax.add_patch(plt.Circle((-7.5e-3, 0.0), 31e-3 / 2, fill=False, linestyle='--', linewidth=1.5, edgecolor='cyan'))
    elif shape_type == 'bowtie_cross':
        ax.add_patch(plt.Rectangle((-0.36, -0.1), 0.72, 0.2, fill=False, linestyle='--', edgecolor='cyan'))
        ax.add_patch(plt.Polygon([(-0.2, 0.36), (0.2, 0.36), (0, 0)], fill=False, linestyle='--', edgecolor='cyan'))
        ax.add_patch(plt.Polygon([(-0.2, -0.36), (0.2, -0.36), (0, 0)], fill=False, linestyle='--', edgecolor='cyan'))


def save_reconstruction_plots(eps_np, sigma_np, ext, suffix, save_dir, shape_type, title_val_eps, title_val_sig):
    FONT_TITLE = 28
    FONT_LABEL = 28
    FONT_TICK = 24

    fig = plt.figure(figsize=(7, 6))
    plt.imshow(eps_np.T, extent=ext, origin='lower', cmap='hot_r')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=FONT_TICK)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().set_axisbelow(True)
    add_gt_contours(plt.gca(), shape_type, ext)
    plt.xlabel("x [m]", fontsize=FONT_LABEL)
    plt.ylabel("y [m]", fontsize=FONT_LABEL)
    plt.title(f"Epsilon ($\\epsilon_r={title_val_eps:g}$)", fontsize=FONT_TITLE, pad=15)
    plt.tick_params(axis='both', which='major', labelsize=FONT_TICK)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"reconstruction_{suffix}_eps.pdf"), format='pdf', dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 6))
    plt.imshow((sigma_np * 1000).T, extent=ext, origin='lower', cmap='hot_r')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=FONT_TICK)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().set_axisbelow(True)
    add_gt_contours(plt.gca(), shape_type, ext)
    plt.xlabel("x [m]", fontsize=FONT_LABEL)
    plt.ylabel("y [m]", fontsize=FONT_LABEL)
    plt.title(f"Sigma ($\\sigma={title_val_sig:g}$ mS/m)", fontsize=FONT_TITLE, pad=15)
    plt.tick_params(axis='both', which='major', labelsize=FONT_TICK)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"reconstruction_{suffix}_sig.pdf"), format='pdf', dpi=150)
    plt.close(fig)


def calc_psnr(true_img, pred_img, data_range):
    true_img, pred_img = true_img.astype(np.float64), pred_img.astype(np.float64)
    mse = np.mean((true_img - pred_img) ** 2)
    return 100.0 if mse < 1e-10 else 10 * np.log10((data_range ** 2) / mse)


def calc_ssim(true_img, pred_img, data_range):
    true_img, pred_img = true_img.astype(np.float64), pred_img.astype(np.float64)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_true, mu_pred = np.mean(true_img), np.mean(pred_img)
    var_true, var_pred = np.var(true_img), np.var(pred_img)
    covar = np.mean((true_img - mu_true) * (pred_img - mu_pred))
    den = (mu_true ** 2 + mu_pred ** 2 + c1) * (var_true + var_pred + c2)
    return 0.0 if den == 0 else ((2 * mu_true * mu_pred + c1) * (2 * covar + c2)) / den


# =============================================================================
# 2. 物理基础算子与数据加载
# =============================================================================
def build_fft_green_kernel(k0, dx, dy, N_grid, a_eff, pad_mult=4):
    N_pad = pad_mult * N_grid
    x_shift = np.fft.fftfreq(N_pad, d=1 / (N_pad * dx))
    y_shift = np.fft.fftfreq(N_pad, d=1 / (N_pad * dy))
    X, Y = np.meshgrid(x_shift, y_shift, indexing='ij')
    rho = np.sqrt(X ** 2 + Y ** 2)
    j1_term = j1(k0 * a_eff)
    coef = -1j * np.pi * k0 * a_eff / 2.0
    g_kernel = np.zeros((N_pad, N_pad), dtype=np.complex64)
    mask_off = rho > 1e-6
    g_kernel[mask_off] = coef * j1_term * hankel2(0, k0 * rho[mask_off])
    g_kernel[~mask_off] = coef * hankel2(1, k0 * a_eff) - 1.0
    return torch.from_numpy(np.fft.fft2(g_kernel)).to(dtype=dtype_c, device=device)


def build_physics_matrices_nocc(k0, r_grid, rx_positions, tx_pos, a_eff):
    diff_SD = rx_positions[:, None, :] - r_grid[None, :, :]
    rho_SD = np.linalg.norm(diff_SD, axis=-1)
    rho_inc = np.linalg.norm(r_grid - tx_pos, axis=-1)
    j1_term = j1(k0 * a_eff)
    coef = -1j * np.pi * k0 * a_eff / 2.0
    Gs_mat = (coef * j1_term * hankel2(0, k0 * rho_SD)).astype(np.complex64)
    E_inc_base = (-hankel2(0, k0 * rho_inc)).astype(np.complex64)
    return torch.from_numpy(Gs_mat).to(device), torch.from_numpy(E_inc_base).to(device)


def build_C_mat(k0, r_grid, a_eff):
    N = r_grid.shape[0]
    diff_DD = r_grid[:, None, :] - r_grid[None, :, :]
    rho_DD = np.linalg.norm(diff_DD, axis=-1)
    j1_term = j1(k0 * a_eff)
    coef = -1j * np.pi * k0 * a_eff / 2.0
    C_mat = np.zeros((N, N), dtype=np.complex64)
    mask_off = ~np.eye(N, dtype=bool)
    C_mat[mask_off] = coef * j1_term * hankel2(0, k0 * rho_DD[mask_off])
    C_mat[np.eye(N, dtype=bool)] = coef * hankel2(1, k0 * a_eff) - 1.0
    return torch.tensor(C_mat, dtype=dtype_c, device=device)


def compute_internal_scattered_fft_batched(W_batched, G_hat_batched, N_grid, pad_mult=4, adjoint=False):
    N_f, _, N_tx = W_batched.shape
    W_padded = torch.nn.functional.pad(W_batched.view(N_f, N_grid, N_grid, N_tx).permute(0, 3, 1, 2),
                                       (0, pad_mult * N_grid - N_grid, 0, pad_mult * N_grid - N_grid))
    W_hat = torch.fft.fft2(W_padded)
    kernel = G_hat_batched.conj() if adjoint else G_hat_batched
    Es_padded = torch.fft.ifft2(W_hat * kernel.unsqueeze(1))
    return Es_padded[:, :, :N_grid, :N_grid].permute(0, 2, 3, 1).reshape(N_f, -1, N_tx)


def load_data_cc(file_path):
    data = np.loadtxt(file_path)
    dx = dy = (2 * ROI_RANGE) / GRID_SIZE
    a_eff = np.sqrt((dx * dy) / np.pi)
    X, Y = np.meshgrid(np.linspace(-ROI_RANGE + dx / 2, ROI_RANGE - dx / 2, GRID_SIZE),
                       np.linspace(-ROI_RANGE + dy / 2, ROI_RANGE - dy / 2, GRID_SIZE), indexing='ij')
    r_grid = np.stack((X.flatten(), Y.flatten()), axis=1)
    dataset = {}

    for f_val in F_SEL:
        f_subset = data[data[:, 0] == f_val]
        omega = 2 * np.pi * f_val * 1e9
        k0_f = omega / C_SPEED
        u_tx = np.unique(np.round(f_subset[:, 1:3], 4), axis=0)
        G_hat = build_fft_green_kernel(k0_f, dx, dy, GRID_SIZE, a_eff, pad_mult=PAD_MULT)
        E_inc_list, Gs_list, Es_meas_list = [], [], []

        for tx in u_tx:
            idx = np.where(np.linalg.norm(f_subset[:, 1:3] - tx, axis=1) < 1e-4)[0]
            sub_tx = f_subset[idx]
            rx = sub_tx[:, 3:5]
            nt, nr = np.linalg.norm(tx), np.linalg.norm(rx, axis=1)
            angles = np.rad2deg(
                np.arccos(np.clip((tx[0] * rx[:, 0] + tx[1] * rx[:, 1]) / (nt * nr + 1e-12), -1.0, 1.0)))
            valid = angles >= 30.0
            if not np.any(valid): continue

            rx_valid = rx[valid]
            ei_m = sub_tx[valid, 5] + 1j * sub_tx[valid, 6]
            et_m = sub_tx[valid, 7] + 1j * sub_tx[valid, 8]
            es_meas = (et_m - ei_m).astype(np.complex64)
            dist_rx = np.linalg.norm(rx_valid - tx, axis=1)
            g_a = -hankel2(0, k0_f * dist_rx)

            cal_idx = np.argsort(np.abs(angles[valid] - 180.0))[:min(CALIB_NUM, len(angles[valid]))]
            a_coeff = np.sum(ei_m[cal_idx] * np.conj(g_a[cal_idx])) / (np.sum(np.abs(g_a[cal_idx]) ** 2) + 1e-12)

            Gs_mat, E_inc_base = build_physics_matrices_nocc(k0_f, r_grid, rx_valid, tx, a_eff)
            E_inc_list.append(
                (E_inc_base * torch.tensor(np.complex64(a_coeff), dtype=dtype_c, device=device)).unsqueeze(1))
            Gs_list.append(Gs_mat)
            Es_meas_list.append(torch.from_numpy(es_meas).to(device))

        max_M = max([gs.shape[0] for gs in Gs_list])
        Gs_padded = torch.zeros((len(Gs_list), max_M, GRID_SIZE ** 2), dtype=dtype_c, device=device)
        Es_padded = torch.zeros((len(Gs_list), max_M), dtype=dtype_c, device=device)
        for i in range(len(Gs_list)):
            M_i = Gs_list[i].shape[0]
            Gs_padded[i, :M_i, :] = Gs_list[i]
            Es_padded[i, :M_i] = Es_meas_list[i]

        num = torch.bmm(Gs_padded.conj().transpose(1, 2).contiguous(), Es_padded.unsqueeze(2).contiguous())
        W_init = (num / (Gs_padded.shape[0] * Gs_padded.shape[1] + 1e-12)).squeeze(2).T

        dataset[f_val] = {
            'omega': omega, 'inv_omega_eps': torch.tensor(1.0 / (omega * EPS0), dtype=dtype_r, device=device),
            'G_hat': G_hat, 'E_inc_mat': torch.cat(E_inc_list, dim=1),
            'Gs_tensor': Gs_padded, 'Es_meas_tensor': Es_padded, 'W_init': W_init,
            'norm_D': torch.mean(torch.abs(Es_padded) ** 2) + 1e-10,
            'norm_S': torch.mean(torch.abs(torch.cat(E_inc_list, dim=1)) ** 2) + 1e-10
        }
    return dataset, r_grid


def load_data_es(file_path):
    os.makedirs("cached_matrices_Cmat", exist_ok=True)
    data = np.loadtxt(file_path)
    dx = dy = (2 * ROI_RANGE) / GRID_SIZE
    a_eff = np.sqrt((dx * dy) / np.pi)
    X, Y = np.meshgrid(np.linspace(-ROI_RANGE + dx / 2, ROI_RANGE - dx / 2, GRID_SIZE),
                       np.linspace(-ROI_RANGE + dy / 2, ROI_RANGE - dy / 2, GRID_SIZE), indexing='ij')
    r_grid = np.stack((X.flatten(), Y.flatten()), axis=1)
    dataset = {}

    for f_val in F_SEL:
        cache_path = os.path.join("cached_matrices_Cmat",
                                  f"VIE_Cache_Cmat_Grid{GRID_SIZE}_F{str(f_val).replace('.', 'd')}.pt")
        k0_f = (2 * np.pi * f_val * 1e9) / C_SPEED
        if os.path.exists(cache_path):
            C_mat_tensor = torch.load(cache_path, map_location=device, weights_only=False)['C_mat']
        else:
            C_mat_tensor = build_C_mat(k0_f, r_grid, a_eff)
            torch.save({'C_mat': C_mat_tensor}, cache_path)

        f_subset = data[data[:, 0] == f_val]
        u_tx = np.unique(np.round(f_subset[:, 1:3], 4), axis=0)
        E_inc_list, Gs_list, Es_meas_list, es_all_meas = [], [], [], []

        for tx in u_tx:
            idx = np.where(np.linalg.norm(f_subset[:, 1:3] - tx, axis=1) < 1e-4)[0]
            sub_tx = f_subset[idx]
            rx = sub_tx[:, 3:5]
            nt, nr = np.linalg.norm(tx), np.linalg.norm(rx, axis=1)
            angles = np.rad2deg(
                np.arccos(np.clip((tx[0] * rx[:, 0] + tx[1] * rx[:, 1]) / (nt * nr + 1e-12), -1.0, 1.0)))
            valid = angles >= 30.0
            if not np.any(valid): continue

            rx_valid = rx[valid]
            ei_m = sub_tx[valid, 5] + 1j * sub_tx[valid, 6]
            et_m = sub_tx[valid, 7] + 1j * sub_tx[valid, 8]
            es_meas = (et_m - ei_m).astype(np.complex64)
            dist_rx = np.linalg.norm(rx_valid - tx, axis=1)
            g_a = -hankel2(0, k0_f * dist_rx)

            cal_idx = np.argsort(np.abs(angles[valid] - 180.0))[:min(CALIB_NUM, len(angles[valid]))]
            a_coeff = np.sum(ei_m[cal_idx] * np.conj(g_a[cal_idx])) / (np.sum(np.abs(g_a[cal_idx]) ** 2) + 1e-12)

            Gs_mat, E_inc_base = build_physics_matrices_nocc(k0_f, r_grid, rx_valid, tx, a_eff)
            E_inc_list.append(
                (E_inc_base * torch.tensor(np.complex64(a_coeff), dtype=dtype_c, device=device)).unsqueeze(1))
            Gs_list.append(Gs_mat)
            Es_meas_list.append(torch.from_numpy(es_meas).to(device))
            es_all_meas.append(Es_meas_list[-1])

        max_rx = max([g.shape[0] for g in Gs_list])
        Gs_batch = torch.zeros((len(Gs_list), max_rx, GRID_SIZE ** 2), dtype=dtype_c, device=device)
        Es_meas_batch = torch.zeros((len(Es_meas_list), max_rx), dtype=dtype_c, device=device)
        mask_batch = torch.zeros((len(Es_meas_list), max_rx), dtype=dtype_r, device=device)

        for i, (g, e) in enumerate(zip(Gs_list, Es_meas_list)):
            num_rx = g.shape[0]
            Gs_batch[i, :num_rx, :] = g
            Es_meas_batch[i, :num_rx] = e
            mask_batch[i, :num_rx] = 1.0

        dataset[f_val] = {
            'omega': 2 * np.pi * f_val * 1e9, 'C_mat': C_mat_tensor,
            'E_inc_mat': torch.cat(E_inc_list, dim=1),
            'Gs_batch': Gs_batch, 'Es_meas_batch': Es_meas_batch, 'mask_batch': mask_batch,
            'norm_D': torch.mean(torch.abs(torch.cat(es_all_meas, dim=0)) ** 2) + 1e-10
        }
    return dataset, r_grid


# =============================================================================
# 3. 算法执行器 (Runners)
# =============================================================================

def run_csi(run_dir):
    """
    经典 CSI 算法执行器。
    去除了 CC-CSI 中的 Data + State 交叉关联物理约束，完全对齐经典的对比源反演方法数学表达。
    """
    start_time = time.time()
    TOTAL_EPOCHS = 10000
    WW_NUM = 0
    dataset, r_grid = load_data_cc(FILE_NAME)
    eps_true, sigma_true, DATA_RANGE_EPS, DATA_RANGE_SIG = generate_ground_truth(r_grid, SHAPE_TYPE)

    chi_master_real = torch.zeros((GRID_SIZE ** 2, 1), dtype=dtype_r, device=device)
    chi_master_imag = torch.zeros((GRID_SIZE ** 2, 1), dtype=dtype_r, device=device)
    W_tensors = {f: data['W_init'].to(dtype_c).clone() for f, data in dataset.items()}

    all_freqs = sorted(dataset.keys())
    freq_stages = [[all_freqs[idx] for idx in si if idx < len(all_freqs)] for si in CONFIG_STAGES]
    if len(freq_stages) == 1:
        stage_epochs_list = [TOTAL_EPOCHS]
    else:
        stage_epochs_list = [
            int(TOTAL_EPOCHS * 0.6) if i == len(freq_stages) - 1 else int((TOTAL_EPOCHS * 0.4) / (len(freq_stages) - 1))
            for i in range(len(freq_stages))
        ]

    metrics = {'step': [], 'psnr_eps': [], 'ssim_eps': [], 'psnr_sig': [], 'ssim_sig': []}
    global_step = 0
    j_complex = torch.tensor(1j, dtype=dtype_c, device=device)

    omega_0 = dataset[all_freqs[0]]['omega'] if len(all_freqs) > 0 else 0

    for stage, active_freqs in enumerate(freq_stages):
        epochs = stage_epochs_list[stage]
        N_f, N_tx, max_M = len(active_freqs), dataset[active_freqs[0]]['Es_meas_tensor'].shape[0], \
            dataset[active_freqs[0]]['Es_meas_tensor'].shape[1]

        raw_weights = {f: (max(all_freqs) / f) ** WW_NUM for f in active_freqs}
        norm_weights_batched = torch.tensor([(raw_weights[f] / sum(raw_weights.values())) * N_f for f in active_freqs],
                                            dtype=dtype_r, device=device).view(N_f)

        Gs_batched = torch.stack([dataset[f]['Gs_tensor'] for f in active_freqs], dim=0)
        Gs_flat = Gs_batched.view(N_f * N_tx, max_M, GRID_SIZE ** 2)
        Es_meas_batched = torch.stack([dataset[f]['Es_meas_tensor'] for f in active_freqs], dim=0)
        G_hat_batched = torch.stack([dataset[f]['G_hat'] for f in active_freqs], dim=0)
        E_inc_batched = torch.stack([dataset[f]['E_inc_mat'] for f in active_freqs], dim=0)

        omega_batched = torch.tensor([dataset[f]['omega'] for f in active_freqs], dtype=dtype_r, device=device)
        omega_0 = dataset[active_freqs[0]]['omega']

        J = torch.stack([W_tensors[f] for f in active_freqs], dim=0)
        Etot = E_inc_batched + compute_internal_scattered_fft_batched(J, G_hat_batched, GRID_SIZE, PAD_MULT)

        if stage == 0:
            chi_init = torch.sum(J * torch.conj(Etot), dim=2) / (torch.sum(torch.abs(Etot) ** 2, dim=2) + 1e-12)
            chi_master_complex = torch.mean(chi_init * (omega_batched / omega_0).view(N_f, 1), dim=0).view(-1, 1)
            chi_master_real = torch.clamp(torch.real(chi_master_complex), min=0.0)
            chi_master_imag = torch.clamp(torch.imag(chi_master_complex), max=0.0)

        v_prev_batched = torch.zeros_like(J)
        g_prev_batched = torch.zeros_like(J)
        is_first_cg_step = True

        WARMUP_STEPS = 100 if stage > 0 else 0

        for epoch in range(epochs):
            chi_f = chi_master_real.unsqueeze(0) + j_complex * chi_master_imag.unsqueeze(0) * (
                    omega_0 / omega_batched).view(N_f, 1, 1)

            Es_pred = torch.bmm(Gs_flat, J.transpose(1, 2).reshape(N_f * N_tx, GRID_SIZE ** 2, 1)).squeeze(2).view(N_f, N_tx, max_M)
            vrho = Es_meas_batched - Es_pred
            chieTot = chi_f * Etot
            vr = chieTot - J

            # 经典 CSI 删除了 Es_cross_pred 相关的约束
            ctmp_S = torch.sum(torch.abs(Es_meas_batched) ** 2, dim=(1, 2), keepdim=True)
            etaS = norm_weights_batched.view(N_f, 1, 1) * ctmp_S / (ctmp_S ** 2 + 1e-12)
            ctmp_D = torch.sum(torch.abs(chieTot) ** 2, dim=(1, 2), keepdim=True)
            etaD = norm_weights_batched.view(N_f, 1, 1) * ctmp_D / (ctmp_D ** 2 + 1e-12)

            # 经典 CSI 删除了 Phixi 项，恢复标准的 chiA 计算
            chiA = compute_internal_scattered_fft_batched(torch.conj(chi_f) * (etaD * vr), G_hat_batched,
                                                          GRID_SIZE, PAD_MULT, adjoint=True)
            vgs = -etaS * torch.bmm(Gs_flat.conj().transpose(1, 2), vrho.view(N_f * N_tx, max_M, 1)).view(N_f, N_tx,
                                                                                                          GRID_SIZE ** 2).transpose(
                1, 2)
            g_W = vgs - etaD * vr + chiA

            if is_first_cg_step:
                v = g_W.clone()
                is_first_cg_step = False
            else:
                ctmp_J = torch.sum(torch.abs(g_prev_batched) ** 2, dim=(1, 2), keepdim=True)
                t_vJ = (ctmp_J / (ctmp_J + 1e-12)) * torch.sum(torch.conj(g_W) * (g_W - g_prev_batched), dim=(1, 2),
                                                               keepdim=True) / (ctmp_J + 1e-12)
                beta_J = ((torch.sqrt(torch.sum(torch.abs(v_prev_batched) ** 2, dim=(1, 2), keepdim=True)) / (
                        torch.sqrt(torch.sum(torch.abs(g_W) ** 2, dim=(1, 2),
                                             keepdim=True)) + 1e-12)) < 1000.0).float() * torch.clamp(
                    torch.real(t_vJ), min=0.0)
                v = g_W + beta_J * v_prev_batched

            eNu = compute_internal_scattered_fft_batched(v, G_hat_batched, GRID_SIZE, PAD_MULT)
            nuchie = v - (chi_f * eNu)
            Fvnu = torch.bmm(Gs_flat, v.transpose(1, 2).reshape(N_f * N_tx, GRID_SIZE ** 2, 1)).squeeze(2).view(N_f, N_tx, max_M)
            Fchie = torch.bmm(Gs_flat, (chi_f * eNu).transpose(1, 2).reshape(N_f * N_tx, GRID_SIZE ** 2, 1)).squeeze(2).view(N_f, N_tx, max_M)

            va = -torch.sum(torch.real(v) * torch.real(g_W) + torch.imag(v) * torch.imag(g_W), dim=(1, 2), keepdim=True) / (etaS * torch.sum(
                torch.real(Fvnu) ** 2 + torch.imag(Fvnu) ** 2 + torch.real(Fchie) ** 2 + torch.imag(Fchie) ** 2,
                dim=(1, 2), keepdim=True) + etaD * torch.sum(torch.real(nuchie) ** 2 + torch.imag(nuchie) ** 2,
                                                             dim=(1, 2), keepdim=True) + 1e-12)
            J, Etot = J + va * v, Etot + va * eNu
            g_prev_batched, v_prev_batched = g_W, v

            # ----------------------------------------------------
            # Phase 2: 更新对比度 Chi (增加预热判断)
            # ----------------------------------------------------
            if epoch < WARMUP_STEPS:
                # 在预热期内，冻结 chi 不更新！
                # 让新加入频率的 J 有足够的时间根据当前优秀的 chi 进行自我修正
                
                global_step += 1
                # 如果需要记录，直接记录当前冻结的 chi 对应的 PSNR
                # 继续下一次 epoch 即可
                continue  

            chi_f_curr = chi_master_real.unsqueeze(0) + j_complex * chi_master_imag.unsqueeze(0) * (
                    omega_0 / omega_batched).view(N_f, 1, 1)
            chieTot_curr, vr_curr = chi_f_curr * Etot, (chi_f_curr * Etot) - J

            etaD_curr = (norm_weights_batched * torch.sum(torch.abs(chieTot_curr) ** 2, dim=(1, 2)) / (
                    torch.sum(torch.abs(chieTot_curr) ** 2, dim=(1, 2)) ** 2 + 1e-12)).view(N_f, 1, 1)
            
            # 经典 CSI 删除了 Phixi_curr 相关的约束
            vgChiF_z = 2 * torch.sum(torch.conj(Etot) * (etaD_curr * vr_curr), dim=2)
            
            vgChiCom = torch.sum(
                torch.real(vgChiF_z) + 1j * (omega_0 / omega_batched).view(N_f, 1) * torch.imag(vgChiF_z),
                dim=0).unsqueeze(1)

            ETot2_z = torch.sum(torch.real(Etot) ** 2 + torch.imag(Etot) ** 2, dim=2)
            vgChi = torch.real(vgChiCom) / (torch.sum(ETot2_z, dim=0).unsqueeze(1) + 1e-12) + 1j * torch.imag(
                vgChiCom) / (torch.sum(ETot2_z * ((omega_0 / omega_batched).view(N_f, 1) ** 2), dim=0).unsqueeze(
                1) + 1e-12)

            if epoch == 0:
                vnuChiCom = vgChi.clone()
            else:
                vnuChiCom = vgChi + vnuChiCom_prev * torch.conj((torch.sum(torch.abs(vgChi_prev) ** 2) / (
                        torch.sum(torch.abs(vgChi_prev) ** 2) + 1e-12)) * torch.sum(
                    torch.conj(vgChi) * (vgChi - vgChi_prev)) / (torch.sum(torch.abs(vgChi_prev) ** 2) + 1e-12))

            vnuChi_f = torch.real(vnuChiCom).unsqueeze(0) + j_complex * torch.imag(vnuChiCom).unsqueeze(0) * (
                    omega_0 / omega_batched).view(N_f, 1, 1)
            enuchi = vnuChi_f * Etot

            a0 = torch.sum(torch.abs(vr_curr) ** 2, dim=(1, 2))
            a1 = 2.0 * torch.sum(torch.real(enuchi) * torch.real(vr_curr) + torch.imag(enuchi) * torch.imag(vr_curr), dim=(1, 2))
            a2 = torch.sum(torch.abs(enuchi) ** 2, dim=(1, 2))

            b1 = 2.0 * torch.sum(torch.real(enuchi) * torch.real(chieTot_curr) + torch.imag(enuchi) * torch.imag(chieTot_curr), dim=(1, 2))
            b2 = a2
            b0 = torch.sum(torch.abs(chieTot_curr) ** 2, dim=(1, 2))

            etaD_sq = norm_weights_batched * b0 / (b0 ** 2 + 1e-12)
            etaD_curr = etaD_sq.view(N_f, 1, 1)

            # 经典 CSI 删除了 c0, c1, c2 相关项
            e1, e2 = torch.sum(etaD_sq * a1).item(), torch.sum(etaD_sq * a2).item()
            sb0 = -e1 / (2 * e2 + 1e-12)

            x_vec = torch.linspace(float(-10.0 * abs(sb0)), 0.0, steps=100000, device=device)
            f_val = torch.zeros_like(x_vec)
            for i in range(N_f):
                num_term = a2[i].item() * (x_vec ** 2) + a1[i].item() * x_vec + a0[i].item()
                den_term = b2[i].item() * (x_vec ** 2) + b1[i].item() * x_vec + b0[i].item()
                f_val += norm_weights_batched[i].item() * (num_term / (den_term + 1e-12))

            min_idx = torch.argmin(f_val)
            sb_opt = x_vec[min_idx].item()

            if sb_opt >= 0: sb_opt = min(sb0, 0.0)

            chi_master_real = torch.clamp(chi_master_real + sb_opt * torch.real(vnuChiCom), min=0.0)
            chi_master_imag = torch.clamp(chi_master_imag + sb_opt * torch.imag(vnuChiCom), max=0.0)
            vgChi_prev, vnuChiCom_prev = vgChi, vnuChiCom

            global_step += 1
            if global_step % 10 == 0:
                eps_np = (chi_master_real.detach().cpu().numpy() + 1.0).reshape(GRID_SIZE, GRID_SIZE)
                sig_np = (-chi_master_imag.detach().cpu().numpy() * (omega_0 * EPS0)).reshape(GRID_SIZE, GRID_SIZE)
                metrics['step'].append(global_step)
                metrics['psnr_eps'].append(calc_psnr(eps_true, eps_np, DATA_RANGE_EPS))
                metrics['ssim_eps'].append(calc_ssim(eps_true, eps_np, DATA_RANGE_EPS))
                metrics['psnr_sig'].append(calc_psnr(sigma_true, sig_np, DATA_RANGE_SIG))
                metrics['ssim_sig'].append(calc_ssim(sigma_true, sig_np, DATA_RANGE_SIG))

        for idx, f_val in enumerate(active_freqs): W_tensors[f_val] = J[idx].detach().clone()

    eps_final_np = (chi_master_real.detach().cpu().numpy() + 1.0).reshape(GRID_SIZE, GRID_SIZE)
    sigma_final_np = (-chi_master_imag.detach().cpu().numpy() * (omega_0 * EPS0)).reshape(GRID_SIZE, GRID_SIZE)

    metrics['run_time'] = time.time() - start_time
    metrics['eps_recon'] = eps_final_np
    metrics['sig_recon'] = sigma_final_np

    torch.save({'chi_master_real': chi_master_real, 'chi_master_imag': chi_master_imag, 'W_tensors': W_tensors},
               os.path.join(run_dir, "model_csi.pth"))

    true_max_eps_val = np.max(eps_true)
    true_max_sig_val = np.max(sigma_true) * 1000
    save_reconstruction_plots(eps_final_np, sigma_final_np, EXT, "final", run_dir, SHAPE_TYPE, true_max_eps_val, true_max_sig_val)

    np.savez(os.path.join(run_dir, "metrics.npz"), **metrics)
    del dataset
    torch.cuda.empty_cache()


def run_cc_csi(run_dir):
    """ CC-CSI: 加入了额外的 Data+State 物理关联项。与上方 run_csi 框架对齐，用于精确控制变量对比。 """
    start_time = time.time()
    TOTAL_EPOCHS = 10000
    WW_NUM = 0
    dataset, r_grid = load_data_cc(FILE_NAME)
    eps_true, sigma_true, DATA_RANGE_EPS, DATA_RANGE_SIG = generate_ground_truth(r_grid, SHAPE_TYPE)

    chi_master_real = torch.zeros((GRID_SIZE ** 2, 1), dtype=dtype_r, device=device)
    chi_master_imag = torch.zeros((GRID_SIZE ** 2, 1), dtype=dtype_r, device=device)
    W_tensors = {f: data['W_init'].to(dtype_c).clone() for f, data in dataset.items()}

    all_freqs = sorted(dataset.keys())
    freq_stages = [[all_freqs[idx] for idx in si if idx < len(all_freqs)] for si in CONFIG_STAGES]
    if len(freq_stages) == 1:
        stage_epochs_list = [TOTAL_EPOCHS]
    else:
        stage_epochs_list = [
            int(TOTAL_EPOCHS * 0.6) if i == len(freq_stages) - 1 else int((TOTAL_EPOCHS * 0.4) / (len(freq_stages) - 1))
            for i in range(len(freq_stages))
        ]

    metrics = {'step': [], 'psnr_eps': [], 'ssim_eps': [], 'psnr_sig': [], 'ssim_sig': []}
    global_step = 0
    j_complex = torch.tensor(1j, dtype=dtype_c, device=device)

    omega_0 = dataset[all_freqs[0]]['omega'] if len(all_freqs) > 0 else 0

    for stage, active_freqs in enumerate(freq_stages):
        epochs = stage_epochs_list[stage]
        N_f, N_tx, max_M = len(active_freqs), dataset[active_freqs[0]]['Es_meas_tensor'].shape[0], \
            dataset[active_freqs[0]]['Es_meas_tensor'].shape[1]

        raw_weights = {f: (max(all_freqs) / f) ** WW_NUM for f in active_freqs}
        norm_weights_batched = torch.tensor([(raw_weights[f] / sum(raw_weights.values())) * N_f for f in active_freqs],
                                            dtype=dtype_r, device=device).view(N_f)

        Gs_batched = torch.stack([dataset[f]['Gs_tensor'] for f in active_freqs], dim=0)
        Gs_flat = Gs_batched.view(N_f * N_tx, max_M, GRID_SIZE ** 2)
        Es_meas_batched = torch.stack([dataset[f]['Es_meas_tensor'] for f in active_freqs], dim=0)
        G_hat_batched = torch.stack([dataset[f]['G_hat'] for f in active_freqs], dim=0)
        E_inc_batched = torch.stack([dataset[f]['E_inc_mat'] for f in active_freqs], dim=0)

        omega_batched = torch.tensor([dataset[f]['omega'] for f in active_freqs], dtype=dtype_r, device=device)
        omega_0 = dataset[active_freqs[0]]['omega']

        J = torch.stack([W_tensors[f] for f in active_freqs], dim=0)
        Etot = E_inc_batched + compute_internal_scattered_fft_batched(J, G_hat_batched, GRID_SIZE, PAD_MULT)

        if stage == 0:
            chi_init = torch.sum(J * torch.conj(Etot), dim=2) / (torch.sum(torch.abs(Etot) ** 2, dim=2) + 1e-12)
            chi_master_complex = torch.mean(chi_init * (omega_batched / omega_0).view(N_f, 1), dim=0).view(-1, 1)
            chi_master_real = torch.clamp(torch.real(chi_master_complex), min=0.0)
            chi_master_imag = torch.clamp(torch.imag(chi_master_complex), max=0.0)

        v_prev_batched = torch.zeros_like(J)
        g_prev_batched = torch.zeros_like(J)
        is_first_cg_step = True

        WARMUP_STEPS = 100 if stage > 0 else 0

        for epoch in range(epochs):
            chi_f = chi_master_real.unsqueeze(0) + j_complex * chi_master_imag.unsqueeze(0) * (
                    omega_0 / omega_batched).view(N_f, 1, 1)

            Es_pred = torch.bmm(Gs_flat, J.transpose(1, 2).reshape(N_f * N_tx, GRID_SIZE ** 2, 1)).squeeze(2).view(N_f, N_tx, max_M)
            vrho = Es_meas_batched - Es_pred
            chieTot = chi_f * Etot
            vr = chieTot - J
            Es_cross_pred = torch.bmm(Gs_flat, chieTot.transpose(1, 2).reshape(N_f * N_tx, GRID_SIZE ** 2, 1)).squeeze(2).view(N_f, N_tx, max_M)
            vxi = Es_meas_batched - Es_cross_pred

            ctmp_S = torch.sum(torch.abs(Es_meas_batched) ** 2, dim=(1, 2), keepdim=True)
            etaS = norm_weights_batched.view(N_f, 1, 1) * ctmp_S / (ctmp_S ** 2 + 1e-12)
            ctmp_D = torch.sum(torch.abs(chieTot) ** 2, dim=(1, 2), keepdim=True)
            etaD = norm_weights_batched.view(N_f, 1, 1) * ctmp_D / (ctmp_D ** 2 + 1e-12)

            Phixi = torch.bmm(Gs_flat.conj().transpose(1, 2), vxi.view(N_f * N_tx, max_M, 1)).view(N_f, N_tx, GRID_SIZE ** 2).transpose(1, 2)
            chiA = compute_internal_scattered_fft_batched(torch.conj(chi_f) * (etaD * vr - etaS * Phixi), G_hat_batched, GRID_SIZE, PAD_MULT, adjoint=True)
            
            vgs = -etaS * torch.bmm(Gs_flat.conj().transpose(1, 2), vrho.view(N_f * N_tx, max_M, 1)).view(N_f, N_tx, GRID_SIZE ** 2).transpose(1, 2)
            g_W = vgs - etaD * vr + chiA

            if is_first_cg_step:
                v = g_W.clone()
                is_first_cg_step = False
            else:
                ctmp_J = torch.sum(torch.abs(g_prev_batched) ** 2, dim=(1, 2), keepdim=True)
                t_vJ = (ctmp_J / (ctmp_J + 1e-12)) * torch.sum(torch.conj(g_W) * (g_W - g_prev_batched), dim=(1, 2),
                                                               keepdim=True) / (ctmp_J + 1e-12)
                beta_J = ((torch.sqrt(torch.sum(torch.abs(v_prev_batched) ** 2, dim=(1, 2), keepdim=True)) / (
                        torch.sqrt(torch.sum(torch.abs(g_W) ** 2, dim=(1, 2),
                                             keepdim=True)) + 1e-12)) < 1000.0).float() * torch.clamp(
                    torch.real(t_vJ), min=0.0)
                v = g_W + beta_J * v_prev_batched

            eNu = compute_internal_scattered_fft_batched(v, G_hat_batched, GRID_SIZE, PAD_MULT)
            nuchie = v - (chi_f * eNu)
            Fvnu = torch.bmm(Gs_flat, v.transpose(1, 2).reshape(N_f * N_tx, GRID_SIZE ** 2, 1)).squeeze(2).view(N_f, N_tx, max_M)
            Fchie = torch.bmm(Gs_flat, (chi_f * eNu).transpose(1, 2).reshape(N_f * N_tx, GRID_SIZE ** 2, 1)).squeeze(2).view(N_f, N_tx, max_M)

            va = -torch.sum(torch.real(v) * torch.real(g_W) + torch.imag(v) * torch.imag(g_W), dim=(1, 2), keepdim=True) / (etaS * torch.sum(
                torch.real(Fvnu) ** 2 + torch.imag(Fvnu) ** 2 + torch.real(Fchie) ** 2 + torch.imag(Fchie) ** 2,
                dim=(1, 2), keepdim=True) + etaD * torch.sum(torch.real(nuchie) ** 2 + torch.imag(nuchie) ** 2,
                                                             dim=(1, 2), keepdim=True) + 1e-12)
            J, Etot = J + va * v, Etot + va * eNu
            g_prev_batched, v_prev_batched = g_W, v

            # ----------------------------------------------------
            # Phase 2: 更新对比度 Chi (增加预热判断)
            # ----------------------------------------------------
            if epoch < WARMUP_STEPS:
                # 在预热期内，冻结 chi 不更新！
                # 让新加入频率的 J 有足够的时间根据当前优秀的 chi 进行自我修正
                
                global_step += 1
                # 如果需要记录，直接记录当前冻结的 chi 对应的 PSNR
                # 继续下一次 epoch 即可
                continue 

            chi_f_curr = chi_master_real.unsqueeze(0) + j_complex * chi_master_imag.unsqueeze(0) * (
                    omega_0 / omega_batched).view(N_f, 1, 1)
            chieTot_curr, vr_curr = chi_f_curr * Etot, (chi_f_curr * Etot) - J
            vxi_curr = Es_meas_batched - torch.bmm(Gs_flat, chieTot_curr.transpose(1, 2).reshape(N_f * N_tx, GRID_SIZE ** 2, 1)).squeeze(2).view(N_f, N_tx, max_M)
            Phixi_curr = torch.bmm(Gs_flat.conj().transpose(1, 2), vxi_curr.view(N_f * N_tx, max_M, 1)).view(N_f, N_tx, GRID_SIZE ** 2).transpose(1, 2)

            etaD_curr = (norm_weights_batched * torch.sum(torch.abs(chieTot_curr) ** 2, dim=(1, 2)) / (
                    torch.sum(torch.abs(chieTot_curr) ** 2, dim=(1, 2)) ** 2 + 1e-12)).view(N_f, 1, 1)
            vgChiF_z = 2 * torch.sum(torch.conj(Etot) * (etaD_curr * vr_curr - etaS * Phixi_curr), dim=2)
            vgChiCom = torch.sum(
                torch.real(vgChiF_z) + 1j * (omega_0 / omega_batched).view(N_f, 1) * torch.imag(vgChiF_z),
                dim=0).unsqueeze(1)

            ETot2_z = torch.sum(torch.real(Etot) ** 2 + torch.imag(Etot) ** 2, dim=2)
            vgChi = torch.real(vgChiCom) / (torch.sum(ETot2_z, dim=0).unsqueeze(1) + 1e-12) + 1j * torch.imag(
                vgChiCom) / (torch.sum(ETot2_z * ((omega_0 / omega_batched).view(N_f, 1) ** 2), dim=0).unsqueeze(
                1) + 1e-12)

            if epoch == 0:
                vnuChiCom = vgChi.clone()
            else:
                vnuChiCom = vgChi + vnuChiCom_prev * torch.conj((torch.sum(torch.abs(vgChi_prev) ** 2) / (
                        torch.sum(torch.abs(vgChi_prev) ** 2) + 1e-12)) * torch.sum(
                    torch.conj(vgChi) * (vgChi - vgChi_prev)) / (torch.sum(torch.abs(vgChi_prev) ** 2) + 1e-12))

            vnuChi_f = torch.real(vnuChiCom).unsqueeze(0) + j_complex * torch.imag(vnuChiCom).unsqueeze(0) * (
                    omega_0 / omega_batched).view(N_f, 1, 1)
            enuchi = vnuChi_f * Etot
            phienuchi = torch.bmm(Gs_flat, enuchi.transpose(1, 2).reshape(N_f * N_tx, GRID_SIZE ** 2, 1)).squeeze(2).view(N_f, N_tx, max_M)

            a0 = torch.sum(torch.abs(vr_curr) ** 2, dim=(1, 2))
            a1 = 2.0 * torch.sum(torch.real(enuchi) * torch.real(vr_curr) + torch.imag(enuchi) * torch.imag(vr_curr), dim=(1, 2))
            a2 = torch.sum(torch.abs(enuchi) ** 2, dim=(1, 2))

            b1 = 2.0 * torch.sum(
                torch.real(enuchi) * torch.real(chieTot_curr) + torch.imag(enuchi) * torch.imag(chieTot_curr), dim=(1, 2))
            b2 = a2
            b0 = torch.sum(torch.abs(chieTot_curr) ** 2, dim=(1, 2))

            etaS_sq = etaS.squeeze(1).squeeze(1)
            etaD_sq = norm_weights_batched * b0 / (b0 ** 2 + 1e-12)
            etaD_curr = etaD_sq.view(N_f, 1, 1)

            c0 = etaS_sq * torch.sum(torch.abs(vxi_curr) ** 2, dim=(1, 2))
            c1 = -2.0 * etaS_sq * torch.sum(
                torch.real(phienuchi) * torch.real(vxi_curr) + torch.imag(phienuchi) * torch.imag(vxi_curr), dim=(1, 2))
            c2 = etaS_sq * torch.sum(torch.abs(phienuchi) ** 2, dim=(1, 2))

            c0_sum, c1_sum, c2_sum = torch.sum(c0).item(), torch.sum(c1).item(), torch.sum(c2).item()
            e1, e2 = torch.sum(etaD_sq * a1 + c1).item(), torch.sum(etaD_sq * a2 + c2).item()
            sb0 = -e1 / (2 * e2 + 1e-12)

            x_vec = torch.linspace(float(-10.0 * abs(sb0)), 0.0, steps=100000, device=device)
            f_val = c2_sum * (x_vec ** 2) + c1_sum * x_vec + c0_sum
            for i in range(N_f):
                num_term = a2[i].item() * (x_vec ** 2) + a1[i].item() * x_vec + a0[i].item()
                den_term = b2[i].item() * (x_vec ** 2) + b1[i].item() * x_vec + b0[i].item()
                f_val += norm_weights_batched[i].item() * (num_term / (den_term + 1e-12))

            min_idx = torch.argmin(f_val)
            sb_opt = x_vec[min_idx].item()

            if sb_opt >= 0: sb_opt = min(sb0, 0.0)

            chi_master_real = torch.clamp(chi_master_real + sb_opt * torch.real(vnuChiCom), min=0.0)
            chi_master_imag = torch.clamp(chi_master_imag + sb_opt * torch.imag(vnuChiCom), max=0.0)
            vgChi_prev, vnuChiCom_prev = vgChi, vnuChiCom

            global_step += 1
            if global_step % 10 == 0:
                eps_np = (chi_master_real.detach().cpu().numpy() + 1.0).reshape(GRID_SIZE, GRID_SIZE)
                sig_np = (-chi_master_imag.detach().cpu().numpy() * (omega_0 * EPS0)).reshape(GRID_SIZE, GRID_SIZE)
                metrics['step'].append(global_step)
                metrics['psnr_eps'].append(calc_psnr(eps_true, eps_np, DATA_RANGE_EPS))
                metrics['ssim_eps'].append(calc_ssim(eps_true, eps_np, DATA_RANGE_EPS))
                metrics['psnr_sig'].append(calc_psnr(sigma_true, sig_np, DATA_RANGE_SIG))
                metrics['ssim_sig'].append(calc_ssim(sigma_true, sig_np, DATA_RANGE_SIG))

        for idx, f_val in enumerate(active_freqs): W_tensors[f_val] = J[idx].detach().clone()

    eps_final_np = (chi_master_real.detach().cpu().numpy() + 1.0).reshape(GRID_SIZE, GRID_SIZE)
    sigma_final_np = (-chi_master_imag.detach().cpu().numpy() * (omega_0 * EPS0)).reshape(GRID_SIZE, GRID_SIZE)

    metrics['run_time'] = time.time() - start_time
    metrics['eps_recon'] = eps_final_np
    metrics['sig_recon'] = sigma_final_np

    torch.save({'chi_master_real': chi_master_real, 'chi_master_imag': chi_master_imag, 'W_tensors': W_tensors},
               os.path.join(run_dir, "model_cc_csi.pth"))

    true_max_eps_val = np.max(eps_true)
    true_max_sig_val = np.max(sigma_true) * 1000
    save_reconstruction_plots(eps_final_np, sigma_final_np, EXT, "final", run_dir, SHAPE_TYPE, true_max_eps_val, true_max_sig_val)

    np.savez(os.path.join(run_dir, "metrics.npz"), **metrics)
    del dataset
    torch.cuda.empty_cache()


def run_dbim(run_dir):
    """
    Distorted Born Iteration Method (DBIM) 执行器。
    基于物理模型利用 Gauss-Newton / LM 算法优化介电常数与电导率。
    """
    start_time = time.time()
    dataset, r_grid = load_data_es(FILE_NAME)
    eps_true, sigma_true, DATA_RANGE_EPS, DATA_RANGE_SIG = generate_ground_truth(r_grid, SHAPE_TYPE)

    eps_r = torch.ones((GRID_SIZE ** 2, 1), dtype=dtype_r, device=device)
    sigma = torch.zeros((GRID_SIZE ** 2, 1), dtype=dtype_r, device=device)
    I_mat = torch.eye(GRID_SIZE ** 2, dtype=dtype_c, device=device)

    all_freqs = sorted(dataset.keys())
    freq_stages = [[all_freqs[idx] for idx in si if idx < len(all_freqs)] for si in CONFIG_STAGES]

    # --- 新增：与 CC-CSI 相同的阶段分配逻辑 ---
    TOTAL_EPOCHS = 150  # DBIM 收敛极快，总步数设为 150 足矣。如果是3个阶段，则为 30, 30, 90
    if len(freq_stages) == 1:
        stage_epochs_list = [TOTAL_EPOCHS]
    else:
        stage_epochs_list = [
            int(TOTAL_EPOCHS * 0.6) if i == len(freq_stages) - 1 else int((TOTAL_EPOCHS * 0.4) / (len(freq_stages) - 1))
            for i in range(len(freq_stages))
        ]
    # ----------------------------------------

    metrics = {'step': [], 'psnr_eps': [], 'ssim_eps': [], 'psnr_sig': [], 'ssim_sig': []}
    global_step = 0

    for stage, active_freqs in enumerate(freq_stages):
        epochs = stage_epochs_list[stage]  # 使用计算好的阶段步数
        gamma = 1.0

        for epoch in range(epochs):
            J_list_real, J_list_imag = [], []
            dE_list_real, dE_list_imag = [], []

            with torch.no_grad():
                for f_val in active_freqs:
                    data = dataset[f_val]
                    omega = data['omega']
                    chi_c = (eps_r.flatten() - 1.0 - 1j * sigma.flatten() / (omega * EPS0)).to(dtype_c)

                    M = I_mat - data['C_mat'] * chi_c.unsqueeze(0)
                    E_tot = torch.linalg.solve(M, data['E_inc_mat'])

                    Es_pred = torch.bmm(data['Gs_batch'], (chi_c.unsqueeze(1) * E_tot).T.unsqueeze(-1)).squeeze(-1)
                    dE = (data['Es_meas_batch'] - Es_pred) * data['mask_batch']

                    Gs_flat = data['Gs_batch'].view(-1, GRID_SIZE ** 2)
                    U_flat_T = torch.linalg.solve(M, Gs_flat.t())
                    U_batch = U_flat_T.t().view(-1, data['Gs_batch'].shape[1], GRID_SIZE ** 2)

                    J_chi = U_batch * E_tot.t().unsqueeze(1)
                    J_chi = J_chi * data['mask_batch'].unsqueeze(-1)
                    J_chi_flat = J_chi.view(-1, GRID_SIZE ** 2)

                    J_eps = J_chi_flat
                    J_sig = J_chi_flat * (-1j / (omega * EPS0))

                    J_list_real.append(torch.cat([J_eps.real, J_sig.real], dim=1))
                    J_list_imag.append(torch.cat([J_eps.imag, J_sig.imag], dim=1))

                    dE_flat = dE.view(-1)
                    dE_list_real.append(dE_flat.real)
                    dE_list_imag.append(dE_flat.imag)

            A = torch.cat(J_list_real + J_list_imag, dim=0)
            b = torch.cat(dE_list_real + dE_list_imag, dim=0)
            error_current = torch.sum(b ** 2).item()

            norm_A = torch.norm(A, dim=0)
            norm_A = torch.clamp(norm_A, min=1e-8)
            A_scaled = A / norm_A.unsqueeze(0)

            A_T_A = torch.matmul(A_scaled.t(), A_scaled)
            A_T_b = torch.matmul(A_scaled.t(), b)
            max_diag = torch.max(torch.diag(A_T_A))

            step_accepted = False
            inner_iters = 0

            while not step_accepted and inner_iters < 5:
                lambda_LM = gamma * max_diag + 1e-6
                dp_scaled = torch.linalg.solve(A_T_A + lambda_LM * torch.eye(A_scaled.shape[1], device=device), A_T_b)

                dp = dp_scaled / norm_A
                dp_eps = dp[:GRID_SIZE ** 2].unsqueeze(1)
                dp_sig = dp[GRID_SIZE ** 2:].unsqueeze(1)

                dp_eps = torch.clamp(dp_eps, min=-2.0, max=2.0)
                dp_sig = torch.clamp(dp_sig, min=-0.1, max=0.1)

                trial_eps = torch.clamp(eps_r + dp_eps, min=1.0, max=80.0)
                trial_sig = torch.clamp(sigma + dp_sig, min=0.0, max=10.0)

                trial_error = 0.0
                with torch.no_grad():
                    for f_val in active_freqs:
                        data = dataset[f_val]
                        omega = data['omega']
                        chi_test = (trial_eps.flatten() - 1.0 - 1j * trial_sig.flatten() / (omega * EPS0)).to(dtype_c)
                        M_test = I_mat - data['C_mat'] * chi_test.unsqueeze(0)
                        E_tot_test = torch.linalg.solve(M_test, data['E_inc_mat'])
                        Es_pred_test = torch.bmm(data['Gs_batch'], (chi_test.unsqueeze(1) * E_tot_test).T.unsqueeze(-1)).squeeze(-1)
                        trial_dE = (data['Es_meas_batch'] - Es_pred_test) * data['mask_batch']
                        trial_error += torch.sum(torch.abs(trial_dE) ** 2).item()

                if trial_error <= error_current:
                    eps_r = trial_eps
                    sigma = trial_sig
                    gamma = max(1e-5, gamma * 0.3)
                    step_accepted = True
                else:
                    gamma = min(1e5, gamma * 4.0)
                    inner_iters += 1

            global_step += 1
            eps_np = eps_r.cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
            sig_np = sigma.cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
            metrics['step'].append(global_step)
            metrics['psnr_eps'].append(calc_psnr(eps_true, eps_np, DATA_RANGE_EPS))
            metrics['ssim_eps'].append(calc_ssim(eps_true, eps_np, DATA_RANGE_EPS))
            metrics['psnr_sig'].append(calc_psnr(sigma_true, sig_np, DATA_RANGE_SIG))
            metrics['ssim_sig'].append(calc_ssim(sigma_true, sig_np, DATA_RANGE_SIG))

            del A, b, A_scaled, A_T_A, A_T_b, dp_scaled, M, E_tot, U_flat_T
            torch.cuda.empty_cache()
            
            if not step_accepted and inner_iters >= 5:
                pass  

    eps_final_np = eps_r.cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
    sigma_final_np = sigma.cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)

    metrics['run_time'] = time.time() - start_time
    metrics['eps_recon'] = eps_final_np
    metrics['sig_recon'] = sigma_final_np

    torch.save({'eps_r': eps_r, 'sigma': sigma}, os.path.join(run_dir, "model_dbim.pth"))

    true_max_eps_val = np.max(eps_true)
    true_max_sig_val = np.max(sigma_true) * 1000
    save_reconstruction_plots(eps_final_np, sigma_final_np, EXT, "final", run_dir, SHAPE_TYPE, true_max_eps_val,
                              true_max_sig_val)

    np.savez(os.path.join(run_dir, "metrics.npz"), **metrics)
    del dataset
    torch.cuda.empty_cache()


# =============================================================================
# 4. 绘图与可视化
# =============================================================================
# =============================================================================
# 4. 绘图与可视化
# =============================================================================
def plot_robustness_results(root_dir):
    # --- 1. 调整算法绘制顺序：CC-CSI 优先，然后是 CSI，最后是 DBIM ---
    algs = [('CC-CSI', 'CC-CSI'), ('CSI', 'CSI'), ('DBIM', 'DBIM')]

    # --- 2. 颜色与线型严格对应上面的算法顺序 ---
    # CC-CSI: 红色 (red) 实线 (-)
    # CSI: 蓝色 (blue) 虚线 (--)
    # DBIM: 紫色 (purple) 点划线 (-.)
    colors = ['red', 'blue', 'purple']
    linestyles = ['-', '--', '-.']

    # --- 3. 统一定义字号 ---
    FONT_TITLE = 28
    FONT_LABEL = 26
    FONT_TICK = 26
    FONT_LEGEND = 23
    LINE_WIDTH = 2.5

    def apply_axis_formatting(ax, is_x_twin=False, is_boxplot=False):
        if not is_boxplot:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            x_formatter = ScalarFormatter(useMathText=True)
            x_formatter.set_scientific(True)
            x_formatter.set_powerlimits((-3, 3))
            ax.xaxis.set_major_formatter(x_formatter)
            ax.xaxis.get_offset_text().set_fontsize(FONT_TICK)

        if not is_x_twin:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            y_formatter = ScalarFormatter(useMathText=True)
            y_formatter.set_scientific(True)
            y_formatter.set_powerlimits((-3, 3))
            ax.yaxis.set_major_formatter(y_formatter)
            ax.yaxis.get_offset_text().set_fontsize(FONT_TICK)

    fig_eps_step, ax_eps_step = plt.subplots(figsize=(10, 10))
    ax_eps_step_twin = ax_eps_step.twiny()
    ax_eps_step_twin.spines['top'].set_position(('outward', 15))

    fig_sig_step, ax_sig_step = plt.subplots(figsize=(10, 10))
    ax_sig_step_twin = ax_sig_step.twiny()
    ax_sig_step_twin.spines['top'].set_position(('outward', 15))

    fig_eps_time, ax_eps_time = plt.subplots(figsize=(10, 9))
    fig_sig_time, ax_sig_time = plt.subplots(figsize=(10, 9))

    final_psnrs_eps, final_psnrs_sig, run_times_list, labels = [], [], [], []

    for i, (display_name, folder_name) in enumerate(algs):
        alg_dir = os.path.join(root_dir, folder_name)
        if not os.path.exists(alg_dir): continue

        runs_eps, runs_sig, runs_time = [], [], []
        steps = None

        for run in sorted(os.listdir(alg_dir)):
            if not run.startswith('run'): continue
            metrics_file = os.path.join(alg_dir, run, 'metrics.npz')
            if not os.path.exists(metrics_file): continue

            data = np.load(metrics_file)
            if steps is None: steps = data['step']
            runs_eps.append(data['psnr_eps'])
            runs_sig.append(data['psnr_sig'])
            if 'run_time' in data: runs_time.append(float(data['run_time']))

        if not runs_eps: continue

        runs_eps, runs_sig = np.array(runs_eps), np.array(runs_sig)
        mean_eps, std_eps = np.mean(runs_eps, axis=0), np.std(runs_eps, axis=0)
        mean_sig, std_sig = np.mean(runs_sig, axis=0), np.std(runs_sig, axis=0)

        # ---------------- 绘制 Step-PSNR 收敛图 ----------------
        if display_name == 'DBIM':
            ax_eps_step_twin.plot(steps, mean_eps, label=display_name, color=colors[i], linestyle=linestyles[i],
                                   linewidth=LINE_WIDTH)
            if len(runs_eps) > 1: ax_eps_step_twin.fill_between(steps, mean_eps - std_eps, mean_eps + std_eps,
                                                                 color=colors[i], alpha=0.2)

            ax_sig_step_twin.plot(steps, mean_sig, label=display_name, color=colors[i], linestyle=linestyles[i],
                                   linewidth=LINE_WIDTH)
            if len(runs_sig) > 1: ax_sig_step_twin.fill_between(steps, mean_sig - std_sig, mean_sig + std_sig,
                                                                 color=colors[i], alpha=0.2)

        else:
            # 这里由于 CC-CSI 在列表第一位，它将被最先绘制；CSI 在第二位，绘制在 CC-CSI 上方
            ax_eps_step.plot(steps, mean_eps, label=display_name, color=colors[i], linestyle=linestyles[i],
                             linewidth=LINE_WIDTH)
            if len(runs_eps) > 1: ax_eps_step.fill_between(steps, mean_eps - std_eps, mean_eps + std_eps,
                                                           color=colors[i], alpha=0.2)

            ax_sig_step.plot(steps, mean_sig, label=display_name, color=colors[i], linestyle=linestyles[i],
                             linewidth=LINE_WIDTH)
            if len(runs_sig) > 1: ax_sig_step.fill_between(steps, mean_sig - std_sig, mean_sig + std_sig,
                                                           color=colors[i], alpha=0.2)

        # ---------------- 绘制 Time-PSNR 收敛图 ----------------
        avg_time = np.mean(runs_time) if runs_time else 0.0
        time_axis = (steps / np.max(steps)) * avg_time if avg_time > 0 else steps

        ax_eps_time.plot(time_axis, mean_eps, label=display_name, color=colors[i], linestyle=linestyles[i],
                         linewidth=LINE_WIDTH)
        if len(runs_eps) > 1: ax_eps_time.fill_between(time_axis, mean_eps - std_eps, mean_eps + std_eps,
                                                       color=colors[i], alpha=0.2)

        ax_sig_time.plot(time_axis, mean_sig, label=display_name, color=colors[i], linestyle=linestyles[i],
                         linewidth=LINE_WIDTH)
        if len(runs_sig) > 1: ax_sig_time.fill_between(time_axis, mean_sig - std_sig, mean_sig + std_sig,
                                                       color=colors[i], alpha=0.2)

        final_psnrs_eps.append(runs_eps[:, -1] if len(runs_eps) > 1 else np.repeat(runs_eps[0, -1], 1))
        final_psnrs_sig.append(runs_sig[:, -1] if len(runs_sig) > 1 else np.repeat(runs_sig[0, -1], 1))
        run_times_list.append(runs_time if runs_time else [0.0] * max(1, len(runs_eps)))
        labels.append(display_name)

    # -- Epsilon Step 收敛图 --
    ax_eps_step.set_title('Epsilon (Relative Permittivity) PSNR vs Steps', fontsize=FONT_TITLE, pad=20)
    ax_eps_step.set_xlabel('Training Steps\n(CC-CSI / CSI)', fontsize=FONT_LABEL, labelpad=15)
    ax_eps_step_twin.set_xlabel('Training Steps (DBIM)', color='purple', fontsize=FONT_LABEL, labelpad=15)
    ax_eps_step.set_ylabel('PSNR (dB)', fontsize=FONT_LABEL)

    ax_eps_step.tick_params(axis='both', which='major', labelsize=FONT_TICK)
    ax_eps_step_twin.tick_params(axis='x', which='major', colors='purple', labelsize=FONT_TICK)
    ax_eps_step.grid(True, linestyle='--', alpha=0.7)

    apply_axis_formatting(ax_eps_step)
    apply_axis_formatting(ax_eps_step_twin, is_x_twin=True)

    h1, l1 = ax_eps_step.get_legend_handles_labels()
    h2, l2 = ax_eps_step_twin.get_legend_handles_labels()
    ax_eps_step.legend(h1 + h2, l1 + l2, loc='best', fontsize=FONT_LEGEND)
    fig_eps_step.tight_layout(pad=1.5)
    fig_eps_step.savefig(os.path.join(root_dir, 'PSNR_Convergence_eps.pdf'), format='pdf')
    plt.close(fig_eps_step)

    # -- Sigma Step 收敛图 --
    ax_sig_step.set_title('Sigma (Conductivity) PSNR vs Steps', fontsize=FONT_TITLE, pad=20)
    ax_sig_step.set_xlabel('Training Steps\n(CC-CSI / CSI)', fontsize=FONT_LABEL, labelpad=15)
    ax_sig_step_twin.set_xlabel('Training Steps (DBIM)', color='purple', fontsize=FONT_LABEL, labelpad=15)
    ax_sig_step.set_ylabel('PSNR (dB)', fontsize=FONT_LABEL)

    ax_sig_step.tick_params(axis='both', which='major', labelsize=FONT_TICK)
    ax_sig_step_twin.tick_params(axis='x', which='major', colors='purple', labelsize=FONT_TICK)
    ax_sig_step.grid(True, linestyle='--', alpha=0.7)

    apply_axis_formatting(ax_sig_step)
    apply_axis_formatting(ax_sig_step_twin, is_x_twin=True)

    h1, l1 = ax_sig_step.get_legend_handles_labels()
    h2, l2 = ax_sig_step_twin.get_legend_handles_labels()
    ax_sig_step.legend(h1 + h2, l1 + l2, loc='best', fontsize=FONT_LEGEND)
    fig_sig_step.tight_layout(pad=1.5)
    fig_sig_step.savefig(os.path.join(root_dir, 'PSNR_Convergence_sig.pdf'), format='pdf')
    plt.close(fig_sig_step)

    # -- Epsilon Time 收敛图 --
    ax_eps_time.set_title('Epsilon (Relative Permittivity) PSNR vs Running Time', fontsize=FONT_TITLE, pad=15)
    ax_eps_time.set_xlabel('Running Time (Seconds)', fontsize=FONT_LABEL, labelpad=15)
    ax_eps_time.set_ylabel('PSNR (dB)', fontsize=FONT_LABEL)
    ax_eps_time.tick_params(axis='both', which='major', labelsize=FONT_TICK)
    apply_axis_formatting(ax_eps_time)
    ax_eps_time.legend(loc='best', fontsize=FONT_LEGEND)
    ax_eps_time.grid(True, linestyle='--', alpha=0.7)
    fig_eps_time.tight_layout(pad=1.5)
    fig_eps_time.savefig(os.path.join(root_dir, 'Time_PSNR_Convergence_eps.pdf'), format='pdf')
    plt.close(fig_eps_time)

    # -- Sigma Time 收敛图 --
    ax_sig_time.set_title('Sigma (Conductivity) PSNR vs Running Time', fontsize=FONT_TITLE, pad=15)
    ax_sig_time.set_xlabel('Running Time (Seconds)', fontsize=FONT_LABEL, labelpad=15)
    ax_sig_time.set_ylabel('PSNR (dB)', fontsize=FONT_LABEL)
    ax_sig_time.tick_params(axis='both', which='major', labelsize=FONT_TICK)
    apply_axis_formatting(ax_sig_time)
    ax_sig_time.legend(loc='best', fontsize=FONT_LEGEND)
    ax_sig_time.grid(True, linestyle='--', alpha=0.7)
    fig_sig_time.tight_layout(pad=1.5)
    fig_sig_time.savefig(os.path.join(root_dir, 'Time_PSNR_Convergence_sig.pdf'), format='pdf')
    plt.close(fig_sig_time)

    # =============== Boxplots =================
    wrapped_labels = [textwrap.fill(lbl, width=10, break_long_words=False) for lbl in labels]

    fig_bp_eps, ax_bp_eps = plt.subplots(figsize=(11, 8))
    bplot_eps = ax_bp_eps.boxplot(final_psnrs_eps, tick_labels=wrapped_labels, patch_artist=True,
                                  boxprops=dict(facecolor='lightgray'))
    for median in bplot_eps['medians']:
        x, y = median.get_xdata(), median.get_ydata()
        ax_bp_eps.text(x.mean(), y[0], f'{y[0]:.2f}', ha='center', va='bottom', color='red', fontsize=20,
                       fontweight='bold')
    ax_bp_eps.set_title('Epsilon Final PSNR Boxplot', fontsize=FONT_TITLE, pad=15)
    ax_bp_eps.set_ylabel('PSNR (dB)', fontsize=FONT_LABEL)
    ax_bp_eps.tick_params(axis='both', which='major', labelsize=FONT_TICK)
    ax_bp_eps.grid(True, linestyle='--', alpha=0.7)
    apply_axis_formatting(ax_bp_eps, is_boxplot=True)
    fig_bp_eps.tight_layout(pad=1.5)
    fig_bp_eps.savefig(os.path.join(root_dir, 'PSNR_Boxplots_eps.pdf'), format='pdf')
    plt.close(fig_bp_eps)

    fig_bp_sig, ax_bp_sig = plt.subplots(figsize=(11, 8))
    bplot_sig = ax_bp_sig.boxplot(final_psnrs_sig, tick_labels=wrapped_labels, patch_artist=True,
                                  boxprops=dict(facecolor='lightgray'))
    for median in bplot_sig['medians']:
        x, y = median.get_xdata(), median.get_ydata()
        ax_bp_sig.text(x.mean(), y[0], f'{y[0]:.2f}', ha='center', va='bottom', color='red', fontsize=20,
                       fontweight='bold')
    ax_bp_sig.set_title('Sigma Final PSNR Boxplot', fontsize=FONT_TITLE, pad=15)
    ax_bp_sig.set_ylabel('PSNR (dB)', fontsize=FONT_LABEL)
    ax_bp_sig.tick_params(axis='both', which='major', labelsize=FONT_TICK)
    ax_bp_sig.grid(True, linestyle='--', alpha=0.7)
    apply_axis_formatting(ax_bp_sig, is_boxplot=True)
    fig_bp_sig.tight_layout(pad=1.5)
    fig_bp_sig.savefig(os.path.join(root_dir, 'PSNR_Boxplots_sig.pdf'), format='pdf')
    plt.close(fig_bp_sig)

    if any(np.sum(times) > 0 for times in run_times_list):
        fig_time, ax_time = plt.subplots(figsize=(11, 8))
        bplot_time = ax_time.boxplot(run_times_list, tick_labels=wrapped_labels, patch_artist=True,
                                     boxprops=dict(facecolor='lightgray'))
        for median in bplot_time['medians']:
            x, y = median.get_xdata(), median.get_ydata()
            ax_time.text(x.mean(), y[0], f'{y[0]:.2f}', ha='center', va='bottom', color='red', fontsize=20,
                         fontweight='bold')
        ax_time.set_title('Algorithm Running Time Boxplot', fontsize=FONT_TITLE, pad=15)
        ax_time.set_ylabel('Time (Seconds)', fontsize=FONT_LABEL)
        ax_time.tick_params(axis='both', which='major', labelsize=FONT_TICK)
        ax_time.grid(True, linestyle='--', alpha=0.7)
        apply_axis_formatting(ax_time, is_boxplot=True)
        fig_time.tight_layout(pad=1.5)
        fig_time.savefig(os.path.join(root_dir, 'Time_Boxplots.pdf'), format='pdf')
        plt.close(fig_time)


# =============================================================================
# 5. 主调控循环
# =============================================================================
if __name__ == "__main__":

    # "Austria" Target
    datasets_to_test = [
        ("AustriaDiel_2_3freqs_20dB", "Data_Sim/AustriaDiel_2_20dB.txt", 'Austria', 0.5, [0.3, 0.4, 0.5],
         [[0], [0, 1], [0, 1, 2]], (2, 2, 2), (0e-3, 0e-3, 0e-3)),
        ("AustriaDiel_3_3freqs_20dB", "Data_Sim/AustriaDiel_3_20dB.txt", 'Austria', 0.5, [0.3, 0.4, 0.5],
         [[0], [0, 1], [0, 1, 2]], (3, 3, 3), (0e-3, 0e-3, 0e-3)),
        ("AustriaDiel_4_3freqs_20dB", "Data_Sim/AustriaDiel_4_20dB.txt", 'Austria', 0.5, [0.3, 0.4, 0.5],
         [[0], [0, 1], [0, 1, 2]], (4, 4, 4), (0e-3, 0e-3, 0e-3)),
        ("AustriaDiel_5_3freqs_20dB", "Data_Sim/AustriaDiel_5_20dB.txt", 'Austria', 0.5, [0.3, 0.4, 0.5],
         [[0], [0, 1], [0, 1, 2]], (5, 5, 5), (0e-3, 0e-3, 0e-3)),
        ("AustriaDiel_6_3freqs_20dB", "Data_Sim/AustriaDiel_6_20dB.txt", 'Austria', 0.5, [0.3, 0.4, 0.5],
         [[0], [0, 1], [0, 1, 2]], (6, 6, 6), (0e-3, 0e-3, 0e-3)),
        ("AustriaDiel_7_3freqs_20dB", "Data_Sim/AustriaDiel_7_20dB.txt", 'Austria', 0.5, [0.3, 0.4, 0.5],
         [[0], [0, 1], [0, 1, 2]], (7, 7, 7), (0e-3, 0e-3, 0e-3))
    ]

    for d_name, f_name, SHAPE_TYPE, ROI_RANGE, F_SEL, CONFIG_STAGES, eps_vals, sig_vals in datasets_to_test:
        DATASET_NAME = d_name
        FILE_NAME = f_name
        EXT = [-ROI_RANGE, ROI_RANGE, -ROI_RANGE, ROI_RANGE]

        RANGE_EPS_1, RANGE_EPS_2, RANGE_EPS_3 = eps_vals
        RANGE_SIG_1, RANGE_SIG_2, RANGE_SIG_3 = sig_vals

        if len(CONFIG_STAGES) == 1:
            TEST_ROOT_DIR = os.path.join("robustness_test_CSI_DBIM_sim", DATASET_NAME)
        else:
            TEST_ROOT_DIR = os.path.join("robustness_test_CSI_DBIM_hop", DATASET_NAME)

        print(f"\n======================================================================")
        print(f"Starting experiments for dataset: {DATASET_NAME}")
        print(f"File: {FILE_NAME}")
        print(f"======================================================================")

        os.makedirs(TEST_ROOT_DIR, exist_ok=True)

        dx = dy = (2 * ROI_RANGE) / GRID_SIZE
        X_mat, Y_mat = np.meshgrid(np.linspace(-ROI_RANGE + dx / 2, ROI_RANGE - dx / 2, GRID_SIZE),
                                   np.linspace(-ROI_RANGE + dy / 2, ROI_RANGE - dy / 2, GRID_SIZE), indexing='ij')
        r_grid_main = np.stack((X_mat.flatten(), Y_mat.flatten()), axis=1)
        eps_true_main, sigma_true_main, _, _ = generate_ground_truth(r_grid_main, SHAPE_TYPE)
        true_max_eps_main = np.max(eps_true_main)
        true_max_sig_main = np.max(sigma_true_main) * 1000

        save_reconstruction_plots(eps_true_main, sigma_true_main, EXT, "GroundTruth", TEST_ROOT_DIR, SHAPE_TYPE,
                                  true_max_eps_main, true_max_sig_main)

        def check_and_run(alg_dir, run_func, *args, **kwargs):
            os.makedirs(alg_dir, exist_ok=True)
            metrics_path = os.path.join(alg_dir, "metrics.npz")
            if os.path.exists(metrics_path):
                print(f"    -> Found existing data in {alg_dir}. Plotting saved results directly...")
                data = np.load(metrics_path)
                if 'eps_recon' in data and 'sig_recon' in data:
                    save_reconstruction_plots(data['eps_recon'], data['sig_recon'], EXT, "final", alg_dir, SHAPE_TYPE,
                                              true_max_eps_main, true_max_sig_main)
            else:
                run_func(alg_dir, *args, **kwargs)

        # 经典计算方法为确定性算法，只需要执行1次即可。
        print("\n--- 1. Executing Deterministic Benchmarks (CSI vs CC-CSI vs DBIM) ---")
        
        print(" -> Executing Classic CSI ...")
        csi_dir = os.path.join(TEST_ROOT_DIR, "CSI", "run01")
        check_and_run(csi_dir, run_csi)
        
        print(" -> Executing CC-CSI ...")
        cccsi_dir = os.path.join(TEST_ROOT_DIR, "CC-CSI", "run01")
        check_and_run(cccsi_dir, run_cc_csi)
        
        print(" -> Executing DBIM ...")
        dbim_dir = os.path.join(TEST_ROOT_DIR, "DBIM", "run01")
        check_and_run(dbim_dir, run_dbim)

        print(f"\nAll experiments for {DATASET_NAME} checked/completed. Generating comparative Plots...")
        plot_robustness_results(TEST_ROOT_DIR)
        print(f"Finished testing {DATASET_NAME}! Plots saved as PDF in '{TEST_ROOT_DIR}' directory.\n")