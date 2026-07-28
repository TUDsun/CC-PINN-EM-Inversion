import os
import time
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.special import hankel2, j1
import torch.backends.cudnn as cudnn

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm

# =============================================================================
# 0. 全局统一配置与硬件信息
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype_c = torch.complex64
dtype_r = torch.float32

if torch.cuda.is_available():
    cudnn.benchmark = True

print(f"Executing Robustness Benchmark on: {device}")

EPS0 = 8.8541878e-12
C_SPEED = 299792458.0
GRID_SIZE = 64
MAPPING_SIZE = 64
B_FILE = f"fixed_B_{MAPPING_SIZE}.pt"
PAD_MULT = 4
FC_SCALE = 1.0
CALIB_NUM = 3

# ----------------- 鲁棒性实验配置 -----------------
NUM_RUNS = 11


# =============================================================================
# 1. 共享辅助函数与评估、绘图函数
# =============================================================================
def generate_perfect_normal_B(m, n, scale=1.0, shuffle_seed=123):
    p = torch.linspace(0.001, 0.999, m * n)
    z = math.sqrt(2) * torch.erfinv(2 * p - 1) * scale
    g = torch.Generator()
    g.manual_seed(shuffle_seed)
    perm = torch.randperm(m * n, generator=g)
    return z[perm].reshape(m, n)


def get_fixed_B_perfect(mapping_size, scale, filepath=B_FILE):
    if os.path.exists(filepath):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
            B = torch.load(filepath, weights_only=True)
    else:
        B = generate_perfect_normal_B(2, mapping_size, scale)
        torch.save(B, filepath)
    return B


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
        # mask_total = mask_h | mask_top | mask_bot
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
    """独立的绘图函数，将Epsilon与Sigma画在各自独立的PDF中"""
    FONT_TITLE = 24
    FONT_LABEL = 24
    FONT_TICK = 22

    # ---------------- 绘制 Epsilon 独立图 ----------------
    fig = plt.figure(figsize=(7, 6))  # 适当放大画板尺寸以容纳大字号标签
    plt.imshow(eps_np.T, extent=ext, origin='lower', cmap='hot_r')

    # 色条刻度字号放大
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=FONT_TICK)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().set_axisbelow(True)
    add_gt_contours(plt.gca(), shape_type, ext)

    # 设置横纵轴名称及字号
    plt.xlabel("x [m]", fontsize=FONT_LABEL)
    plt.ylabel("y [m]", fontsize=FONT_LABEL)

    # 设置标题及字号
    plt.title(f"Epsilon ($\\epsilon_r={title_val_eps:g}$)", fontsize=FONT_TITLE, pad=15)

    # 设置坐标轴刻度的字号
    plt.tick_params(axis='both', which='major', labelsize=FONT_TICK)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"reconstruction_{suffix}_eps.pdf"), format='pdf', dpi=150)
    plt.close(fig)

    # ---------------- 绘制 Sigma 独立图 ----------------
    fig = plt.figure(figsize=(7, 6))
    plt.imshow((sigma_np * 1000).T, extent=ext, origin='lower', cmap='hot_r')

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=FONT_TICK)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().set_axisbelow(True)
    add_gt_contours(plt.gca(), shape_type, ext)

    # 设置横纵轴名称及字号
    plt.xlabel("x [m]", fontsize=FONT_LABEL)
    plt.ylabel("y [m]", fontsize=FONT_LABEL)

    # 设置标题及字号
    plt.title(f"Sigma ($\\sigma={title_val_sig:g}$ mS/m)", fontsize=FONT_TITLE, pad=15)

    # 设置坐标轴刻度的字号
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


class MaterialNet(nn.Module):
    def __init__(self, eps_scale, mapping_size=64, scale=FC_SCALE, B_file=B_FILE):
        super().__init__()
        self.eps_scale = eps_scale
        B_tensor = get_fixed_B_perfect(mapping_size, scale, B_file).to(torch.float32)
        self.B = nn.Parameter(B_tensor, requires_grad=False)
        self.net = nn.Sequential(
            weight_norm(nn.Linear(mapping_size * 2, 256)), nn.SiLU(),
            weight_norm(nn.Linear(256, 256)), nn.SiLU(),
            weight_norm(nn.Linear(256, 128)), nn.SiLU(),
            nn.Linear(128, 2)
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.net[-1].bias, -3.0)

    def forward(self, x, y):
        pts = torch.cat([x, y], dim=1).to(torch.float32)
        proj = (2.0 * np.pi) * torch.matmul(pts, self.B)
        x_pe = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        out = self.net(x_pe)
        # 严格遵守各自参数设定的范围动态缩放
        eps_r = torch.sigmoid(out[:, 0:1]) * self.eps_scale + 1.0
        sigma = torch.sigmoid(out[:, 1:2]) * 1.0
        return eps_r, sigma


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
def run_cc_pinn_variant(run_dir, classic_mode=False):
    start_time = time.time()
    TOTAL_EPOCHS = 75000

    WW_NUM, ALPHA_VALUE = 0, 10

    dataset, r_grid = load_data_cc(FILE_NAME)
    eps_true, sigma_true, DATA_RANGE_EPS, DATA_RANGE_SIG = generate_ground_truth(r_grid, SHAPE_TYPE)

    net_eps = MaterialNet(eps_scale=79.0).to(device)
    x_t = torch.from_numpy(r_grid[:, 0:1]).to(dtype_r).to(device) / ROI_RANGE
    y_t = torch.from_numpy(r_grid[:, 1:2]).to(dtype_r).to(device) / ROI_RANGE

    W_params = nn.ParameterDict()
    for f_val, data in dataset.items():
        W_params[f"f_{str(f_val).replace('.', '_')}"] = nn.Parameter(data['W_init'].to(dtype_c))

    optimizer = torch.optim.Adam(
        [{'params': net_eps.parameters(), 'lr': 1e-3}, {'params': W_params.parameters(), 'lr': 2e-3}])

    all_freqs = sorted(dataset.keys())
    f_max_global = max(all_freqs)
    raw_weights = {f: (f_max_global / f) ** WW_NUM for f in all_freqs}

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

    for stage, active_freqs in enumerate(freq_stages):
        epochs = stage_epochs_list[stage]
        N_f_stage = len(active_freqs)

        Gs_batched = torch.stack([dataset[f]['Gs_tensor'] for f in active_freqs], dim=0)
        Es_meas_batched = torch.stack([dataset[f]['Es_meas_tensor'] for f in active_freqs], dim=0)
        G_hat_batched = torch.stack([dataset[f]['G_hat'] for f in active_freqs], dim=0)
        E_inc_batched = torch.stack([dataset[f]['E_inc_mat'] for f in active_freqs], dim=0)
        inv_omega_eps_batched = torch.stack([dataset[f]['inv_omega_eps'] for f in active_freqs], dim=0).view(N_f_stage,
                                                                                                             1, 1)
        norm_D_batched = torch.stack([dataset[f]['norm_D'] for f in active_freqs], dim=0).view(N_f_stage, 1, 1)
        norm_S_batched = torch.stack([dataset[f]['norm_S'] for f in active_freqs], dim=0).view(N_f_stage, 1, 1)

        w_stage_sum = sum([raw_weights[f] for f in active_freqs])
        norm_weights_batched = torch.tensor([(raw_weights[f] / w_stage_sum) * N_f_stage for f in active_freqs],
                                            dtype=dtype_r, device=device).view(N_f_stage)
        Gs_flat = Gs_batched.view(N_f_stage * Gs_batched.shape[1], Gs_batched.shape[2], GRID_SIZE ** 2).contiguous()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        for epoch in range(epochs):
            optimizer.zero_grad()
            eps_curr, sigma_curr = net_eps(x_t, y_t)
            chi_batched = eps_curr.to(dtype_c).unsqueeze(0) - j_complex * sigma_curr.to(dtype_c).unsqueeze(
                0) * inv_omega_eps_batched - 1.0

            W_f_batched = torch.stack([W_params[f"f_{str(f).replace('.', '_')}"] for f in active_freqs], dim=0)
            W_f_vec_batched = W_f_batched.transpose(1, 2).reshape(N_f_stage * Gs_batched.shape[1], GRID_SIZE ** 2,
                                                                  1).contiguous()
            Es_pred_batched = torch.bmm(Gs_flat, W_f_vec_batched).squeeze(2).view(N_f_stage, Gs_batched.shape[1],
                                                                                  Gs_batched.shape[2])
            loss_Data = torch.mean(torch.abs(Es_pred_batched - Es_meas_batched) ** 2,
                                   dim=(1, 2)) / norm_D_batched.squeeze()

            E_s_dom_batched = compute_internal_scattered_fft_batched(W_f_batched, G_hat_batched, GRID_SIZE)
            W_pred_batched = chi_batched * (E_inc_batched + E_s_dom_batched)
            loss_State = torch.mean(torch.abs(W_f_batched - W_pred_batched) ** 2, dim=(1, 2)) / norm_S_batched.squeeze()

            if classic_mode:
                total_loss = torch.sum(norm_weights_batched * (loss_Data + loss_State))
            else:
                W_pred_vec_batched = W_pred_batched.transpose(1, 2).reshape(N_f_stage * Gs_batched.shape[1],
                                                                            GRID_SIZE ** 2, 1).contiguous()
                Es_cross_pred = torch.bmm(Gs_flat, W_pred_vec_batched).squeeze(2).view(N_f_stage, Gs_batched.shape[1],
                                                                                       Gs_batched.shape[2])
                loss_Cross = torch.mean(torch.abs(Es_cross_pred - Es_meas_batched) ** 2,
                                        dim=(1, 2)) / norm_D_batched.squeeze()

                beta_cross = np.exp(-ALPHA_VALUE * (epoch / stage_epochs_list[0]))
                total_loss = torch.sum(norm_weights_batched * (loss_Data + loss_State + beta_cross * loss_Cross))

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net_eps.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(W_params.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % 100 == 0:
                eps_np = eps_curr.detach().cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
                sig_np = sigma_curr.detach().cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
                metrics['step'].append(global_step)
                metrics['psnr_eps'].append(calc_psnr(eps_true, eps_np, DATA_RANGE_EPS))
                metrics['ssim_eps'].append(calc_ssim(eps_true, eps_np, DATA_RANGE_EPS))
                metrics['psnr_sig'].append(calc_psnr(sigma_true, sig_np, DATA_RANGE_SIG))
                metrics['ssim_sig'].append(calc_ssim(sigma_true, sig_np, DATA_RANGE_SIG))

    # 生成最终的反演结果并保存参数模型
    net_eps.eval()
    with torch.no_grad():
        eps_final, sigma_final = net_eps(x_t, y_t)

    eps_final_np = eps_final.detach().cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
    sigma_final_np = sigma_final.detach().cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)

    metrics['run_time'] = time.time() - start_time
    metrics['eps_recon'] = eps_final_np
    metrics['sig_recon'] = sigma_final_np

    # 存入模型权重
    torch.save(net_eps.state_dict(), os.path.join(run_dir, "model_net_eps.pth"))
    torch.save(W_params.state_dict(), os.path.join(run_dir, "model_W_params.pth"))

    true_max_eps_val = np.max(eps_true)
    true_max_sig_val = np.max(sigma_true) * 1000
    save_reconstruction_plots(eps_final_np, sigma_final_np, EXT, "final", run_dir, SHAPE_TYPE, true_max_eps_val,
                              true_max_sig_val)

    np.savez(os.path.join(run_dir, "metrics.npz"), **metrics)
    del dataset, net_eps, W_params, optimizer
    torch.cuda.empty_cache()


def run_es_pinn(run_dir):
    start_time = time.time()
    TOTAL_EPOCHS = 10000
    dataset, r_grid = load_data_es(FILE_NAME)
    eps_true, sigma_true, DATA_RANGE_EPS, DATA_RANGE_SIG = generate_ground_truth(r_grid, SHAPE_TYPE)

    net_eps = MaterialNet(eps_scale=15.0).to(device)
    optimizer = torch.optim.Adam(net_eps.parameters(), lr=1e-3)
    x_t, y_t = torch.tensor(r_grid[:, 0], dtype=dtype_r, device=device).unsqueeze(1) / ROI_RANGE, torch.tensor(
        r_grid[:, 1], dtype=dtype_r, device=device).unsqueeze(1) / ROI_RANGE
    I_mat = torch.eye(GRID_SIZE ** 2, dtype=dtype_c, device=device)

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

    for stage, active_freqs in enumerate(freq_stages):
        epochs = stage_epochs_list[stage]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        for epoch in range(epochs):
            optimizer.zero_grad()
            eps_r, sigma = net_eps(x_t, y_t)
            eps_r_c, sigma_c = eps_r.flatten().to(dtype_c), sigma.flatten().to(dtype_c)
            total_loss = 0.0

            for f_val in active_freqs:
                data = dataset[f_val]
                chi = (eps_r_c - 1j * sigma_c / (data['omega'] * EPS0)) - 1.0
                E_tot_mat = torch.linalg.solve(I_mat - data['C_mat'] * chi.unsqueeze(0), data['E_inc_mat'])
                J_internal = (chi.unsqueeze(1) * E_tot_mat).t()
                Es_pred_batch = torch.bmm(data['Gs_batch'], J_internal.unsqueeze(-1)).squeeze(-1)

                diff = torch.abs(Es_pred_batch - data['Es_meas_batch']) ** 2
                freq_loss = torch.sum(diff * data['mask_batch'], dim=1) / (torch.sum(data['mask_batch'], dim=1) + 1e-12)
                total_loss += torch.mean(freq_loss) / data['norm_D']

            (total_loss / len(active_freqs)).backward()
            torch.nn.utils.clip_grad_norm_(net_eps.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % 20 == 0:
                eps_np = eps_r.detach().cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
                sig_np = sigma.detach().cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
                metrics['step'].append(global_step)
                metrics['psnr_eps'].append(calc_psnr(eps_true, eps_np, DATA_RANGE_EPS))
                metrics['ssim_eps'].append(calc_ssim(eps_true, eps_np, DATA_RANGE_EPS))
                metrics['psnr_sig'].append(calc_psnr(sigma_true, sig_np, DATA_RANGE_SIG))
                metrics['ssim_sig'].append(calc_ssim(sigma_true, sig_np, DATA_RANGE_SIG))

    # 生成最终的反演结果并保存参数模型
    net_eps.eval()
    with torch.no_grad():
        eps_final, sigma_final = net_eps(x_t, y_t)

    eps_final_np = eps_final.detach().cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
    sigma_final_np = sigma_final.detach().cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)

    metrics['run_time'] = time.time() - start_time
    metrics['eps_recon'] = eps_final_np
    metrics['sig_recon'] = sigma_final_np

    # 存入模型权重
    torch.save(net_eps.state_dict(), os.path.join(run_dir, "model_net_eps.pth"))

    true_max_eps_val = np.max(eps_true)
    true_max_sig_val = np.max(sigma_true) * 1000
    save_reconstruction_plots(eps_final_np, sigma_final_np, EXT, "final", run_dir, SHAPE_TYPE, true_max_eps_val,
                              true_max_sig_val)

    np.savez(os.path.join(run_dir, "metrics.npz"), **metrics)
    del dataset, net_eps, optimizer
    torch.cuda.empty_cache()


def run_cc_csi(run_dir):
    start_time = time.time()
    TOTAL_EPOCHS = 75000
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

        for epoch in range(epochs):
            chi_f = chi_master_real.unsqueeze(0) + j_complex * chi_master_imag.unsqueeze(0) * (
                    omega_0 / omega_batched).view(N_f, 1, 1)

            # Phase 1: Update J
            Es_pred = torch.bmm(Gs_flat, J.transpose(1, 2).reshape(N_f * N_tx, GRID_SIZE ** 2, 1)).squeeze(2).view(N_f,
                                                                                                                   N_tx,
                                                                                                                   max_M)
            vrho = Es_meas_batched - Es_pred
            chieTot = chi_f * Etot
            vr = chieTot - J
            Es_cross_pred = torch.bmm(Gs_flat, chieTot.transpose(1, 2).reshape(N_f * N_tx, GRID_SIZE ** 2, 1)).squeeze(
                2).view(N_f, N_tx, max_M)
            vxi = Es_meas_batched - Es_cross_pred

            ctmp_S = torch.sum(torch.abs(Es_meas_batched) ** 2, dim=(1, 2), keepdim=True)
            etaS = norm_weights_batched.view(N_f, 1, 1) * ctmp_S / (ctmp_S ** 2 + 1e-12)
            ctmp_D = torch.sum(torch.abs(chieTot) ** 2, dim=(1, 2), keepdim=True)
            etaD = norm_weights_batched.view(N_f, 1, 1) * ctmp_D / (ctmp_D ** 2 + 1e-12)

            Phixi = torch.bmm(Gs_flat.conj().transpose(1, 2), vxi.view(N_f * N_tx, max_M, 1)).view(N_f, N_tx,
                                                                                                   GRID_SIZE ** 2).transpose(
                1, 2)
            chiA = compute_internal_scattered_fft_batched(torch.conj(chi_f) * (etaD * vr - etaS * Phixi), G_hat_batched,
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
            Fvnu = torch.bmm(Gs_flat, v.transpose(1, 2).reshape(N_f * N_tx, GRID_SIZE ** 2, 1)).squeeze(2).view(N_f,
                                                                                                                N_tx,
                                                                                                                max_M)
            Fchie = torch.bmm(Gs_flat, (chi_f * eNu).transpose(1, 2).reshape(N_f * N_tx, GRID_SIZE ** 2, 1)).squeeze(
                2).view(N_f, N_tx, max_M)

            va = -torch.sum(torch.real(v) * torch.real(g_W) + torch.imag(v) * torch.imag(g_W), dim=(1, 2),
                            keepdim=True) / (etaS * torch.sum(
                torch.real(Fvnu) ** 2 + torch.imag(Fvnu) ** 2 + torch.real(Fchie) ** 2 + torch.imag(Fchie) ** 2,
                dim=(1, 2), keepdim=True) + etaD * torch.sum(torch.real(nuchie) ** 2 + torch.imag(nuchie) ** 2,
                                                             dim=(1, 2), keepdim=True) + 1e-12)
            J, Etot = J + va * v, Etot + va * eNu
            g_prev_batched, v_prev_batched = g_W, v

            # Phase 2: Update Chi
            chi_f_curr = chi_master_real.unsqueeze(0) + j_complex * chi_master_imag.unsqueeze(0) * (
                    omega_0 / omega_batched).view(N_f, 1, 1)
            chieTot_curr, vr_curr = chi_f_curr * Etot, (chi_f_curr * Etot) - J
            vxi_curr = Es_meas_batched - torch.bmm(Gs_flat,
                                                   chieTot_curr.transpose(1, 2).reshape(N_f * N_tx, GRID_SIZE ** 2,
                                                                                        1)).squeeze(2).view(N_f, N_tx,
                                                                                                            max_M)
            Phixi_curr = torch.bmm(Gs_flat.conj().transpose(1, 2), vxi_curr.view(N_f * N_tx, max_M, 1)).view(N_f, N_tx,
                                                                                                             GRID_SIZE ** 2).transpose(
                1, 2)

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
            phienuchi = torch.bmm(Gs_flat, enuchi.transpose(1, 2).reshape(N_f * N_tx, GRID_SIZE ** 2, 1)).squeeze(
                2).view(N_f, N_tx, max_M)

            a0 = torch.sum(torch.abs(vr_curr) ** 2, dim=(1, 2))
            a1 = 2.0 * torch.sum(torch.real(enuchi) * torch.real(vr_curr) + torch.imag(enuchi) * torch.imag(vr_curr),
                                 dim=(1, 2))
            a2 = torch.sum(torch.abs(enuchi) ** 2, dim=(1, 2))

            b1 = 2.0 * torch.sum(
                torch.real(enuchi) * torch.real(chieTot_curr) + torch.imag(enuchi) * torch.imag(chieTot_curr),
                dim=(1, 2))
            b2 = a2
            b0 = torch.sum(torch.abs(chieTot_curr) ** 2, dim=(1, 2))

            etaS_sq = etaS.squeeze(1).squeeze(1)
            # 修正复用处 1: 修正 etaD_curr 冗余计算
            etaD_sq = norm_weights_batched * b0 / (b0 ** 2 + 1e-12)
            etaD_curr = etaD_sq.view(N_f, 1, 1)

            # 修正复用处 2: 提取变量 c0, c1, c2，避免后续重复运算
            c0 = etaS_sq * torch.sum(torch.abs(vxi_curr) ** 2, dim=(1, 2))
            c1 = -2.0 * etaS_sq * torch.sum(
                torch.real(phienuchi) * torch.real(vxi_curr) + torch.imag(phienuchi) * torch.imag(vxi_curr), dim=(1, 2))
            c2 = etaS_sq * torch.sum(torch.abs(phienuchi) ** 2, dim=(1, 2))

            c0_sum = torch.sum(c0).item()
            c1_sum = torch.sum(c1).item()
            c2_sum = torch.sum(c2).item()

            e1 = torch.sum(etaD_sq * a1 + c1).item()
            e2 = torch.sum(etaD_sq * a2 + c2).item()
            sb0 = -e1 / (2 * e2 + 1e-12)

            x_vec = torch.linspace(float(-10.0 * abs(sb0)), 0.0, steps=100000, device=device)
            f_val = c2_sum * (x_vec ** 2) + c1_sum * x_vec + c0_sum
            for i in range(N_f):
                # 修正复用处 4: 剔除错误的多余分母冗余
                num_term = a2[i].item() * (x_vec ** 2) + a1[i].item() * x_vec + a0[i].item()
                den_term = b2[i].item() * (x_vec ** 2) + b1[i].item() * x_vec + b0[i].item()
                f_val += norm_weights_batched[i].item() * (num_term / (den_term + 1e-12))

            min_idx = torch.argmin(f_val)
            sb_opt = x_vec[min_idx].item()

            # 【核心修复】：如果数值网格搜索卡死在边界(>=0)，强制回退采用解析公式求出的步长 sb0
            if sb_opt >= 0:
                sb_opt = min(sb0, 0.0)
            # -----------------------------------------------------------------------------------------------------------------

            chi_master_real = torch.clamp(chi_master_real + sb_opt * torch.real(vnuChiCom), min=0.0)
            chi_master_imag = torch.clamp(chi_master_imag + sb_opt * torch.imag(vnuChiCom), max=0.0)
            vgChi_prev, vnuChiCom_prev = vgChi, vnuChiCom

            global_step += 1
            if global_step % 100 == 0:
                eps_np = (chi_master_real.detach().cpu().numpy() + 1.0).reshape(GRID_SIZE, GRID_SIZE)
                sig_np = (-chi_master_imag.detach().cpu().numpy() * (omega_0 * EPS0)).reshape(GRID_SIZE, GRID_SIZE)
                metrics['step'].append(global_step)
                metrics['psnr_eps'].append(calc_psnr(eps_true, eps_np, DATA_RANGE_EPS))
                metrics['ssim_eps'].append(calc_ssim(eps_true, eps_np, DATA_RANGE_EPS))
                metrics['psnr_sig'].append(calc_psnr(sigma_true, sig_np, DATA_RANGE_SIG))
                metrics['ssim_sig'].append(calc_ssim(sigma_true, sig_np, DATA_RANGE_SIG))

        for idx, f_val in enumerate(active_freqs): W_tensors[f_val] = J[idx].detach().clone()

    # 生成最终的反演结果并保存参数模型
    eps_final_np = (chi_master_real.detach().cpu().numpy() + 1.0).reshape(GRID_SIZE, GRID_SIZE)
    sigma_final_np = (-chi_master_imag.detach().cpu().numpy() * (omega_0 * EPS0)).reshape(GRID_SIZE, GRID_SIZE)

    metrics['run_time'] = time.time() - start_time
    metrics['eps_recon'] = eps_final_np
    metrics['sig_recon'] = sigma_final_np

    # 保存 CC-CSI 迭代得到的直接状态量
    torch.save({'chi_master_real': chi_master_real, 'chi_master_imag': chi_master_imag, 'W_tensors': W_tensors},
               os.path.join(run_dir, "model_cc_csi.pth"))

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
def plot_robustness_results(root_dir):
    algs = [('CC-PINN', 'CC-PINN'), ('Data+State-PINN', 'Data+State-PINN'), ('ES-PINN', 'ES-PINN'), ('CC-CSI', 'CC-CSI')]

    # --- 1. 颜色与线型区分 ---
    colors = ['blue', 'orange', 'green', 'red']
    linestyles = ['-', '--', '-.', ':']  # 分别对应: 实线, 虚线, 点划线, 点线

    # --- 2. 统一定义字号 ---
    FONT_TITLE = 24
    FONT_LABEL = 22
    FONT_TICK = 22
    FONT_LEGEND = 22
    LINE_WIDTH = 2.5

    # 适当放大画布以容纳大字号
    fig_eps_step, ax_eps_step = plt.subplots(figsize=(10, 9))
    ax_eps_step_twin = ax_eps_step.twiny()

    fig_sig_step, ax_sig_step = plt.subplots(figsize=(10, 9))
    ax_sig_step_twin = ax_sig_step.twiny()

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
        if display_name == 'ES-PINN':
            ax_eps_step_twin.plot(steps, mean_eps, label=display_name, color=colors[i], linestyle=linestyles[i],
                                  linewidth=LINE_WIDTH)
            if len(runs_eps) > 1: ax_eps_step_twin.fill_between(steps, mean_eps - std_eps, mean_eps + std_eps,
                                                                color=colors[i], alpha=0.2)

            ax_sig_step_twin.plot(steps, mean_sig, label=display_name, color=colors[i], linestyle=linestyles[i],
                                  linewidth=LINE_WIDTH)
            if len(runs_sig) > 1: ax_sig_step_twin.fill_between(steps, mean_sig - std_sig, mean_sig + std_sig,
                                                                color=colors[i], alpha=0.2)
        else:
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

        final_psnrs_eps.append(runs_eps[:, -1] if len(runs_eps) > 1 else np.repeat(runs_eps[0, -1], NUM_RUNS))
        final_psnrs_sig.append(runs_sig[:, -1] if len(runs_sig) > 1 else np.repeat(runs_sig[0, -1], NUM_RUNS))
        run_times_list.append(runs_time if runs_time else [0.0] * max(1, len(runs_eps)))
        labels.append(display_name)

    # =============== 调整字号与保存图表 ===============

    # -- Epsilon Step 收敛图 --
    ax_eps_step.set_title('Epsilon (Relative Permittivity) PSNR vs Steps', fontsize=FONT_TITLE, pad=20)
    ax_eps_step.set_xlabel('Training Steps (CC-PINN / Data+State-PINN / CC-CSI)', fontsize=FONT_LABEL)
    ax_eps_step_twin.set_xlabel('Training Steps (ES-PINN)', color='green', fontsize=FONT_LABEL)
    ax_eps_step.set_ylabel('PSNR (dB)', fontsize=FONT_LABEL)

    ax_eps_step.tick_params(axis='both', which='major', labelsize=FONT_TICK)
    ax_eps_step_twin.tick_params(axis='x', which='major', colors='green', labelsize=FONT_TICK)
    ax_eps_step.grid(True, linestyle='--', alpha=0.7)

    h1, l1 = ax_eps_step.get_legend_handles_labels()
    h2, l2 = ax_eps_step_twin.get_legend_handles_labels()
    # 修改1: loc 设置为 'best'
    ax_eps_step.legend(h1 + h2, l1 + l2, loc='best', fontsize=FONT_LEGEND)
    fig_eps_step.tight_layout()
    fig_eps_step.savefig(os.path.join(root_dir, 'PSNR_Convergence_eps.pdf'), format='pdf')
    plt.close(fig_eps_step)

    # -- Sigma Step 收敛图 --
    ax_sig_step.set_title('Sigma (Conductivity) PSNR vs Steps', fontsize=FONT_TITLE, pad=20)
    ax_sig_step.set_xlabel('Training Steps (CC-PINN / Data+State-PINN / CC-CSI)', fontsize=FONT_LABEL)
    ax_sig_step_twin.set_xlabel('Training Steps (ES-PINN)', color='green', fontsize=FONT_LABEL)
    ax_sig_step.set_ylabel('PSNR (dB)', fontsize=FONT_LABEL)

    ax_sig_step.tick_params(axis='both', which='major', labelsize=FONT_TICK)
    ax_sig_step_twin.tick_params(axis='x', which='major', colors='green', labelsize=FONT_TICK)
    ax_sig_step.grid(True, linestyle='--', alpha=0.7)

    h1, l1 = ax_sig_step.get_legend_handles_labels()
    h2, l2 = ax_sig_step_twin.get_legend_handles_labels()
    # 修改2: loc 设置为 'best'
    ax_sig_step.legend(h1 + h2, l1 + l2, loc='best', fontsize=FONT_LEGEND)
    fig_sig_step.tight_layout()
    fig_sig_step.savefig(os.path.join(root_dir, 'PSNR_Convergence_sig.pdf'), format='pdf')
    plt.close(fig_sig_step)

    # -- Epsilon Time 收敛图 --
    ax_eps_time.set_title('Epsilon (Relative Permittivity) PSNR vs Running Time', fontsize=FONT_TITLE, pad=15)
    ax_eps_time.set_xlabel('Running Time (Seconds)', fontsize=FONT_LABEL)
    ax_eps_time.set_ylabel('PSNR (dB)', fontsize=FONT_LABEL)
    ax_eps_time.tick_params(axis='both', which='major', labelsize=FONT_TICK)
    # 修改3: 增加 loc='best'
    ax_eps_time.legend(loc='best', fontsize=FONT_LEGEND)
    ax_eps_time.grid(True, linestyle='--', alpha=0.7)
    fig_eps_time.tight_layout()
    fig_eps_time.savefig(os.path.join(root_dir, 'Time_PSNR_Convergence_eps.pdf'), format='pdf')
    plt.close(fig_eps_time)

    # -- Sigma Time 收敛图 --
    ax_sig_time.set_title('Sigma (Conductivity) PSNR vs Running Time', fontsize=FONT_TITLE, pad=15)
    ax_sig_time.set_xlabel('Running Time (Seconds)', fontsize=FONT_LABEL)
    ax_sig_time.set_ylabel('PSNR (dB)', fontsize=FONT_LABEL)
    ax_sig_time.tick_params(axis='both', which='major', labelsize=FONT_TICK)
    # 修改4: 增加 loc='best'
    ax_sig_time.legend(loc='best', fontsize=FONT_LEGEND)
    ax_sig_time.grid(True, linestyle='--', alpha=0.7)
    fig_sig_time.tight_layout()
    fig_sig_time.savefig(os.path.join(root_dir, 'Time_PSNR_Convergence_sig.pdf'), format='pdf')
    plt.close(fig_sig_time)

    # -- Boxplots --
    fig_bp_eps, ax_bp_eps = plt.subplots(figsize=(10, 8))
    # 修改5: 增加 patch_artist=True 与 boxprops 设定为浅灰，并接收返回值
    bplot_eps = ax_bp_eps.boxplot(final_psnrs_eps, tick_labels=labels, patch_artist=True,
                                  boxprops=dict(facecolor='lightgray'))

    # 在红线（中位数）上方标注数值
    for median in bplot_eps['medians']:
        x, y = median.get_xdata(), median.get_ydata()
        # x.mean() 取水平线段的中点，y[0] 是纵坐标。在其正上方添加文本
        ax_bp_eps.text(x.mean(), y[0], f'{y[0]:.2f}',
                       ha='center', va='bottom', color='red', fontsize=22, fontweight='bold')

    ax_bp_eps.set_title('Epsilon Final PSNR Boxplot', fontsize=FONT_TITLE, pad=15)
    ax_bp_eps.set_ylabel('PSNR (dB)', fontsize=FONT_LABEL)
    ax_bp_eps.tick_params(axis='both', which='major', labelsize=FONT_TICK)
    ax_bp_eps.grid(True, linestyle='--', alpha=0.7)
    fig_bp_eps.tight_layout()
    fig_bp_eps.savefig(os.path.join(root_dir, 'PSNR_Boxplots_eps.pdf'), format='pdf')
    plt.close(fig_bp_eps)

    fig_bp_sig, ax_bp_sig = plt.subplots(figsize=(10, 8))
    # 修改6: 增加 patch_artist=True 与 boxprops 设定为浅灰，并接收返回值
    bplot_sig = ax_bp_sig.boxplot(final_psnrs_sig, tick_labels=labels, patch_artist=True,
                                  boxprops=dict(facecolor='lightgray'))

    # 在红线（中位数）上方标注数值
    for median in bplot_sig['medians']:
        x, y = median.get_xdata(), median.get_ydata()
        ax_bp_sig.text(x.mean(), y[0], f'{y[0]:.2f}',
                       ha='center', va='bottom', color='red', fontsize=22, fontweight='bold')

    ax_bp_sig.set_title('Sigma Final PSNR Boxplot', fontsize=FONT_TITLE, pad=15)
    ax_bp_sig.set_ylabel('PSNR (dB)', fontsize=FONT_LABEL)
    ax_bp_sig.tick_params(axis='both', which='major', labelsize=FONT_TICK)
    ax_bp_sig.grid(True, linestyle='--', alpha=0.7)
    fig_bp_sig.tight_layout()
    fig_bp_sig.savefig(os.path.join(root_dir, 'PSNR_Boxplots_sig.pdf'), format='pdf')
    plt.close(fig_bp_sig)

    if any(np.sum(times) > 0 for times in run_times_list):
        fig_time, ax_time = plt.subplots(figsize=(10, 8))
        # 修改7: 增加 patch_artist=True 与 boxprops 设定为浅灰，并接收返回值
        bplot_time = ax_time.boxplot(run_times_list, tick_labels=labels, patch_artist=True,
                                     boxprops=dict(facecolor='lightgray'))

        # 在红线（中位数）上方标注数值
        for median in bplot_time['medians']:
            x, y = median.get_xdata(), median.get_ydata()
            ax_time.text(x.mean(), y[0], f'{y[0]:.2f}',
                         ha='center', va='bottom', color='red', fontsize=22, fontweight='bold')

        ax_time.set_title('Algorithm Running Time Boxplot', fontsize=FONT_TITLE, pad=15)
        ax_time.set_ylabel('Time (Seconds)', fontsize=FONT_LABEL)
        ax_time.tick_params(axis='both', which='major', labelsize=FONT_TICK)
        ax_time.grid(True, linestyle='--', alpha=0.7)
        fig_time.tight_layout()
        fig_time.savefig(os.path.join(root_dir, 'Time_Boxplots.pdf'), format='pdf')
        plt.close(fig_time)


# =============================================================================
# 5. 主调控循环 (自适应完成多组数据集的测试)
# =============================================================================
if __name__ == "__main__":

    # 定义需要自动进行鲁棒性测试的 4 组数据集

    # --------------------Simulation Data---------------------
    # "Austria" Target
    # datasets_to_test = [
    #     ("AustriaDiel_4_3freqs_20dB", "Data_Sim/AustriaDiel_4_20dB.txt", 'Austria', 0.5, [0.3, 0.4, 0.5], [[0], [0, 1], [0, 1, 2]], (4, 4, 4), (0e-3, 0e-3, 0e-3)),
    #     ("AustriaDiel_5_3freqs_20dB", "Data_Sim/AustriaDiel_5_20dB.txt", 'Austria', 0.5, [0.3, 0.4, 0.5], [[0], [0, 1], [0, 1, 2]], (5, 5, 5), (0e-3, 0e-3, 0e-3)),
    #     ("AustriaDiel_6_3freqs_20dB", "Data_Sim/AustriaDiel_6_20dB.txt", 'Austria', 0.5, [0.3, 0.4, 0.5], [[0], [0, 1], [0, 1, 2]], (6, 6, 6), (0e-3, 0e-3, 0e-3)),
    #     ("AustriaDiel_7_3freqs_20dB", "Data_Sim/AustriaDiel_7_20dB.txt", 'Austria', 0.5, [0.3, 0.4, 0.5], [[0], [0, 1], [0, 1, 2]], (7, 7, 7), (0e-3, 0e-3, 0e-3))
    # ]

    # "bowtie-cross" Target
    # datasets_to_test = [
    #     ("BowtieCrossDiel_4_3freqs_20dB", "Data_Sim/BowtieCrossDiel_4_20dB.txt", 'bowtie_cross', 0.5, [0.3, 0.4, 0.5], [[0], [0, 1], [0, 1, 2]], (4, 4, 4), (0e-3, 0e-3, 0e-3)),
    #     ("BowtieCrossDiel_5_3freqs_20dB", "Data_Sim/BowtieCrossDiel_5_20dB.txt", 'bowtie_cross', 0.5, [0.3, 0.4, 0.5], [[0], [0, 1], [0, 1, 2]], (5, 5, 5), (0e-3, 0e-3, 0e-3)),
    #     ("BowtieCrossDiel_6_3freqs_20dB", "Data_Sim/BowtieCrossDiel_6_20dB.txt", 'bowtie_cross', 0.5, [0.3, 0.4, 0.5], [[0], [0, 1], [0, 1, 2]], (6, 6, 6), (0e-3, 0e-3, 0e-3)),
    #     ("BowtieCrossDiel_7_3freqs_20dB", "Data_Sim/BowtieCrossDiel_7_20dB.txt", 'bowtie_cross', 0.5, [0.3, 0.4, 0.5], [[0], [0, 1], [0, 1, 2]], (7, 7, 7), (0e-3, 0e-3, 0e-3))
    # ]

    # lower SNRs
    # datasets_to_test = [
    #     ("AustriaDiel_5_3freqs_10dB", "Data_Sim/AustriaDiel_5_10dB.txt", 'Austria', 0.5, [0.3, 0.4, 0.5], [[0], [0, 1], [0, 1, 2]], (5, 5, 5), (0e-3, 0e-3, 0e-3)),
    #     ("AustriaDiel_5_3freqs_0dB", "Data_Sim/AustriaDiel_5_0dB.txt", 'Austria', 0.5, [0.3, 0.4, 0.5], [[0], [0, 1], [0, 1, 2]], (5, 5, 5), (0e-3, 0e-3, 0e-3)),
    # ]

    datasets_to_test = [
        ("BowtieCrossDiel_5_3freqs_10dB", "Data_Sim/BowtieCrossDiel_5_10dB.txt", 'bowtie_cross', 0.5, [0.3, 0.4, 0.5], [[0], [0, 1], [0, 1, 2]], (5, 5, 5), (0e-3, 0e-3, 0e-3)),
        ("BowtieCrossDiel_5_3freqs_0dB", "Data_Sim/BowtieCrossDiel_5_0dB.txt", 'bowtie_cross', 0.5, [0.3, 0.4, 0.5], [[0], [0, 1], [0, 1, 2]], (5, 5, 5), (0e-3, 0e-3, 0e-3))
    ]

    # Lossy Object Data
    # datasets_to_test = [
    #     ("AustriaLossy_e6s30_3freqs_20dB", "Data_Sim/AustriaLossy_e6s30_20dB.txt", 'Austria', 0.5, [0.3, 0.4, 0.5], [[0], [0, 1], [0, 1, 2]], (6.0, 6.0, 6.0), (30e-3, 30e-3, 30e-3)),
    #     ("BowtieCrossLossy_RectEpsr6Sig30_TriEpsr3Sig60_3freqs_20dB", "Data_Sim/BowtieCrossLossy_RectEpsr6Sig30_TriEpsr3Sig60_20dB.txt", 'bowtie_cross', 0.5, [0.3, 0.4, 0.5], [[0], [0, 1], [0, 1, 2]], (6.0, 3.0, 3.0), (30e-3, 60e-3, 60e-3)),
    # ]

    # --------------------Fresnel Data--------------------
    # datasets_to_test = [
    #     ("FoamTwinDielTM_3freqs789", "Fresnel_Data/The_second_opus/FoamTwinDielTM_Format.txt", 'FoamTwinDiel', 0.1, [7, 8, 9], [[0], [0, 1], [0, 1, 2]], (1.45, 3.0, 3.0), (0e-3, 0e-3, 0e-3)),
    #     ("FoamTwinDielTM_3freqs789", "Fresnel_Data/The_second_opus/FoamTwinDielTM_Format.txt", 'FoamTwinDiel', 0.1, [7, 8, 9], [[0, 1, 2]], (1.45, 3.0, 3.0), (0e-3, 0e-3, 0e-3)),
    #     ("FoamTwinDielTM_3freqs345", "Fresnel_Data/The_second_opus/FoamTwinDielTM_Format.txt", 'FoamTwinDiel', 0.1, [3, 4, 5], [[0], [0, 1], [0, 1, 2]], (1.45, 3.0, 3.0), (0e-3, 0e-3, 0e-3)),
    #     ("FoamTwinDielTM_3freqs345", "Fresnel_Data/The_second_opus/FoamTwinDielTM_Format.txt", 'FoamTwinDiel', 0.1, [3, 4, 5], [[0, 1, 2]], (1.45, 3.0, 3.0), (0e-3, 0e-3, 0e-3)),
    # ]

    for d_name, f_name, SHAPE_TYPE, ROI_RANGE, F_SEL, CONFIG_STAGES, eps_vals, sig_vals in datasets_to_test:
        # 动态更新全局变量，使其能够被其余代码正确地读取和运行
        DATASET_NAME = d_name
        FILE_NAME = f_name

        EXT = [-ROI_RANGE, ROI_RANGE, -ROI_RANGE, ROI_RANGE]

        # 分别解包并赋值给 3 个区域的介电常数和电导率
        RANGE_EPS_1, RANGE_EPS_2, RANGE_EPS_3 = eps_vals
        RANGE_SIG_1, RANGE_SIG_2, RANGE_SIG_3 = sig_vals

        if len(CONFIG_STAGES) == 1:
            TEST_ROOT_DIR = os.path.join("robustness_test_sim", DATASET_NAME)
        else:
            TEST_ROOT_DIR = os.path.join("robustness_test_hop", DATASET_NAME)

        print(f"\n======================================================================")
        print(f"Starting experiments for dataset: {DATASET_NAME}")
        print(f"File: {FILE_NAME}")
        print(f"Target Epsilon: {RANGE_EPS_1}, {RANGE_EPS_2}, {RANGE_EPS_3}")
        print(f"Target Sigma (mS/m): {RANGE_SIG_1 * 1000:g}, {RANGE_SIG_2 * 1000:g}, {RANGE_SIG_3 * 1000:g}")
        print(f"======================================================================")

        os.makedirs(TEST_ROOT_DIR, exist_ok=True)

        # ---------------- 绘制并保存真实值参考图像 ----------------
        dx = dy = (2 * ROI_RANGE) / GRID_SIZE
        X_mat, Y_mat = np.meshgrid(np.linspace(-ROI_RANGE + dx / 2, ROI_RANGE - dx / 2, GRID_SIZE),
                                   np.linspace(-ROI_RANGE + dy / 2, ROI_RANGE - dy / 2, GRID_SIZE), indexing='ij')
        r_grid_main = np.stack((X_mat.flatten(), Y_mat.flatten()), axis=1)
        eps_true_main, sigma_true_main, _, _ = generate_ground_truth(r_grid_main, SHAPE_TYPE)

        true_max_eps_main = np.max(eps_true_main)
        true_max_sig_main = np.max(sigma_true_main) * 1000

        save_reconstruction_plots(eps_true_main, sigma_true_main, EXT, "GroundTruth", TEST_ROOT_DIR, SHAPE_TYPE,
                                  true_max_eps_main, true_max_sig_main)
        print(f"Ground truth plots have been successfully saved to '{TEST_ROOT_DIR}'.")


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
                    print(f"    -> Missing recon data in {alg_dir}, please clear the folder to recalculate.")
            else:
                run_func(alg_dir, *args, **kwargs)


        print("\n--- 1. Executing CC-CSI (Deterministic, 1 Run) ---")
        cccsi_dir = os.path.join(TEST_ROOT_DIR, "CC-CSI", "run01")
        check_and_run(cccsi_dir, run_cc_csi)

        for run_idx in range(1, NUM_RUNS + 1):
            run_str = f"run{run_idx:02d}"
            seed = 1000 + run_idx

            print(f"\n=== Starting Robustness Iteration {run_idx}/{NUM_RUNS} (Seed: {seed}) ===")

            # 强制更新各类随机数生成器以测试网络初始化的鲁棒性
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # 1. PINN (Data + State)
            print(f" -> Executing PINN [Data+State] ({run_str})...")
            pinn_dir = os.path.join(TEST_ROOT_DIR, "data+state-PINN", run_str)
            check_and_run(pinn_dir, run_cc_pinn_variant, classic_mode=True)

            # 2. CC-PINN (Data + State + Cross)
            print(f" -> Executing CC-PINN ({run_str})...")
            ccpinn_dir = os.path.join(TEST_ROOT_DIR, "CC-PINN", run_str)
            check_and_run(ccpinn_dir, run_cc_pinn_variant, classic_mode=False)

            # 3. ES-PINN
            print(f" -> Executing ES-PINN ({run_str})...")
            es_dir = os.path.join(TEST_ROOT_DIR, "ES-PINN", run_str)
            check_and_run(es_dir, run_es_pinn)

            # # 4. CC-CSI
            # print(f" -> Executing CC-CSI ({run_str})...")
            # cccsi_dir = os.path.join(TEST_ROOT_DIR, "CC-CSI", run_str)
            # check_and_run(cccsi_dir, run_cc_csi)

        print(f"\nAll experiments for {DATASET_NAME} checked/completed. Generating comparative Robustness Plots...")
        plot_robustness_results(TEST_ROOT_DIR)
        print(f"Finished testing {DATASET_NAME}! Plots saved as PDF in '{TEST_ROOT_DIR}' directory.\n")
