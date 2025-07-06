import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import yaml  # YAML用のimportを追加
from PIL import Image
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import adjusted_rand_score as ari
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 潜在空間の次元を定義
x_dim = 32  # モジュールレベルでx_dimを定義

class AffordanceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # ディレクトリの設定
        self.rgb_dir = os.path.join(root_dir, 'rgb')
        self.affordance_dir = os.path.join(root_dir, 'affordances')
        self.metadata_dir = os.path.join(root_dir, 'metadata')
        
        # 有効なファイルペアを取得
        self.valid_pairs = []
        rgb_files = os.listdir(self.rgb_dir)
        
        for rgb_file in rgb_files:
            if rgb_file.endswith('.jpg'):
                base_name = rgb_file[:-4]  # .jpgを除去
                affordance_file = f"{base_name}.txt"
                metadata_file = f"{base_name}.txt"
                
                affordance_path = os.path.join(self.affordance_dir, affordance_file)
                metadata_path = os.path.join(self.metadata_dir, metadata_file)
                
                if os.path.exists(affordance_path) and os.path.exists(metadata_path):
                    self.valid_pairs.append((rgb_file, affordance_file, metadata_file))
                else:
                    print(f"Missing files for {base_name}")

    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        rgb_file, affordance_file, metadata_file = self.valid_pairs[idx]
        
        # RGB画像の読み込み
        image_path = os.path.join(self.rgb_dir, rgb_file)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # アフォーダンスマップの読み込み
        affordance_path = os.path.join(self.affordance_dir, affordance_file)
        try:
            # テキストファイルとして読み込み
            affordance_map = np.loadtxt(affordance_path)
            # 64x64にリサイズ
            affordance_map = torch.from_numpy(affordance_map).long()  # long型に変更
            affordance_map = F.interpolate(
                affordance_map.unsqueeze(0).unsqueeze(0).float(),  # [1, 1, H, W]に変換
                size=(64, 64),
                mode='nearest'
            ).squeeze(0).squeeze(0).long()  # 元の次元に戻し、long型に戻す
            affordance_tensor = affordance_map
        except Exception as e:
            print(f"Error loading affordance map from {affordance_path}: {str(e)}")
            raise
        
        # メタデータからカテゴリ情報を取得
        metadata_path = os.path.join(self.metadata_dir, metadata_file)
        try:
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
                category_id = metadata['object_id']
        except Exception as e:
            print(f"Error loading metadata from {metadata_path}: {str(e)}")
            raise
        
        return image, affordance_tensor, category_id

class UnFlatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), 128, 4, 4)

class VAE(nn.Module):
    def __init__(self, x_dim=x_dim, h_dim=1024, image_channels=3, affordance_channels=10):
        super(VAE, self).__init__()
        self.x_dim = x_dim  # 潜在変数の次元をインスタンス変数として保存
        
        # 画像エンコーダー
        self.image_encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(image_channels, 16, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            # 32x32 -> 16x16
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            # 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten()  # 128x4x4 -> 2048
        )
        
        # 特徴量のサイズを計算
        self.feature_size = 128 * 4 * 4
        
        # 平均と分散を出力する全結合層
        self.fc1 = nn.Linear(self.feature_size, self.x_dim)  # 平均
        self.fc2 = nn.Linear(self.feature_size, self.x_dim)  # 対数分散
        self.fc3 = nn.Linear(self.x_dim, self.feature_size)  # デコーダーへの入力
        
        # 画像デコーダー
        self.image_decoder = nn.Sequential(
            UnFlatten(),  # 128x4x4に変形
            # 4x4 -> 8x8
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(16, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        # アフォーダンス予測デコーダー（潜在変数からアフォーダンスマップを予測）
        self.affordance_predictor = nn.Sequential(
            UnFlatten(),  # 128x4x4に変形
            # 4x4 -> 8x8
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(16, affordance_channels, kernel_size=4, stride=2, padding=1)
        )
        
        # 重みの初期化
        self.apply(self._init_weights)
        self.affordance_weight = 1.0  # デフォルト値
        self.kl_weight = 0.0  # 追加: KL損失の重み
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, images):
        # 画像のみをエンコード
        features = self.image_encoder(images)
        
        # 平均と分散を計算
        mu = self.fc1(features)
        logvar = self.fc2(features)
        
        return mu, logvar
    
    def bottleneck(self, mu, logvar):
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        # 潜在表現を特徴量に変換
        features = self.fc3(z)
        
        # 画像とアフォーダンスマップを生成
        image_recon = self.image_decoder(features)
        affordance_pred = self.affordance_predictor(features)
        
        return image_recon, affordance_pred
    
    def forward(self, images):
        mu, logvar = self.encode(images)
        z, mu, logvar = self.bottleneck(mu, logvar)
        image_recon, affordance_pred = self.decode(z)
        return image_recon, affordance_pred, mu, logvar, z

    def loss_function(self, image_recon, images, affordance_pred, affordances, mu, logvar, gmm_mu=None, gmm_var=None, iteration=0):
        # 画像の再構成誤差（MSE）
        image_loss = F.mse_loss(image_recon, images, reduction='sum')
        
        # アフォーダンスマップの予測誤差（CrossEntropy）
        affordance_loss = F.cross_entropy(
            affordance_pred.view(affordance_pred.size(0), affordance_pred.size(1), -1),
            affordances.view(affordances.size(0), -1),
            reduction='sum'
        ) * self.affordance_weight
        
        # KLダイバージェンス
        if iteration > 0 and gmm_mu is not None and gmm_var is not None:
            # GMMを事前分布として使用
            gmm_mu = nn.Parameter(gmm_mu)
            prior_mu = gmm_mu
            prior_mu.requires_grad = False
            prior_mu = prior_mu.expand_as(mu).to(device)
            
            gmm_var = nn.Parameter(gmm_var)
            prior_var = gmm_var
            prior_var.requires_grad = False
            prior_var = prior_var.expand_as(logvar).to(device)
            
            prior_logvar = nn.Parameter(prior_var.log())
            prior_logvar.requires_grad = False
            prior_logvar = prior_logvar.expand_as(logvar).to(device)
            
            var_division = logvar.exp() / prior_var
            diff = mu - prior_mu
            diff_term = diff * diff / prior_var
            logvar_division = prior_logvar - logvar
            
            kl_loss = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.x_dim)
        else:
            # 標準正規分布を事前分布として使用
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 総損失
        total_loss = image_loss + affordance_loss + kl_loss
        
        return total_loss, image_loss, affordance_loss, kl_loss

def calculate_mean_iou(pred, target, num_classes=4):
    """mean-IoUを計算する関数"""
    intersection = torch.zeros(num_classes, device=pred.device)
    union = torch.zeros(num_classes, device=pred.device)
    
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection[cls] = (pred_mask & target_mask).sum().float()
        union[cls] = (pred_mask | target_mask).sum().float()
    
    # 0除算を防ぐため、unionが0の場合はIoUを0とする
    iou = torch.where(union > 0, intersection / union, torch.zeros_like(intersection))
    return iou.mean().item()

def train(iteration, gmm_mu, gmm_var, epoch, train_loader, batch_size, all_loader, model_dir, agent, affordance_weight=1.0):
    """VAEの学習を実行"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.affordance_weight = affordance_weight
    
    # 損失とメトリクスの履歴を保存
    history = {
        'total_loss': np.zeros(epoch),
        'image_loss': np.zeros(epoch),
        'affordance_loss': np.zeros(epoch),
        'kl_loss': np.zeros(epoch),
        'mean_iou': np.zeros(epoch)
    }
    
    for i in range(epoch):
        model.train()
        epoch_total_loss = 0
        epoch_image_loss = 0
        epoch_affordance_loss = 0
        epoch_kl_loss = 0
        epoch_mean_iou = 0
        num_batches = 0
        
        for batch_idx, (images, affordances, categories) in enumerate(train_loader):
            images = images.to(device)
            affordances = affordances.to(device)
            categories = categories.to(device)
            
            optimizer.zero_grad()
            
            # 順伝播
            image_recon, affordance_pred, mu, logvar, z = model(images)
            
            # 損失の計算
            if iteration == 0:
                losses = model.loss_function(
                    image_recon, images, affordance_pred, affordances,
                    mu, logvar, gmm_mu=None, gmm_var=None, iteration=iteration
                )
            else:
                batch_gmm_mu = gmm_mu[batch_idx*batch_size:(batch_idx+1)*batch_size]
                batch_gmm_var = gmm_var[batch_idx*batch_size:(batch_idx+1)*batch_size]
                losses = model.loss_function(
                    image_recon, images, affordance_pred, affordances,
                    mu, logvar, batch_gmm_mu, batch_gmm_var, iteration=iteration
                )
            
            # 損失の分解
            total_loss, image_loss, affordance_loss, kl_loss = losses
            
            # 逆伝播
            total_loss = total_loss.mean()
            image_loss = image_loss.mean()
            affordance_loss = affordance_loss.mean()
            kl_loss = kl_loss.mean()
            total_loss.backward()
            optimizer.step()
            
            # 損失の累積
            epoch_total_loss += total_loss.item()
            epoch_image_loss += image_loss.item()
            epoch_affordance_loss += affordance_loss.item()
            epoch_kl_loss += kl_loss.item()
            
            # mean-IoUの計算
            pred = affordance_pred.argmax(dim=1)  # [B, H, W]
            iou = calculate_mean_iou(pred, affordances)
            epoch_mean_iou += iou
            num_batches += 1
        
        # エポックの平均を計算
        n_samples = len(train_loader.dataset)
        epoch_total_loss /= n_samples
        epoch_image_loss /= n_samples
        epoch_affordance_loss /= n_samples
        epoch_kl_loss /= n_samples
        epoch_mean_iou /= num_batches
        
        # 履歴を保存
        history['total_loss'][i] = epoch_total_loss
        history['image_loss'][i] = epoch_image_loss
        history['affordance_loss'][i] = epoch_affordance_loss
        history['kl_loss'][i] = epoch_kl_loss
        history['mean_iou'][i] = epoch_mean_iou
        
        # エポックの結果を表示
        if i == 0 or (i + 1) % 5 == 0 or i == epoch - 1:
            print(f"====> Epoch: {i+1}")
            print(f"Total Loss: {epoch_total_loss:.4f}")
            print(f"Image Loss: {epoch_image_loss:.4f}")
            print(f"Affordance Loss: {epoch_affordance_loss:.4f}")
            print(f"KL Loss: {epoch_kl_loss:.4f}")
            print(f"Mean-IoU: {epoch_mean_iou:.4f}")
    
    # 学習曲線のプロット
    plt.figure(figsize=(15, 10))
    
    # 損失のプロット
    plt.subplot(2, 1, 1)
    plt.plot(range(epoch), history['total_loss'], label='Total Loss', color='blue')
    if iteration != 0:
        loss_0 = np.load(os.path.join(model_dir, f'npy/loss{agent}_0.npy'))
        plt.plot(range(epoch), loss_0, label='Loss (Iteration 0)', color='red', linestyle='--')
    plt.plot(range(epoch), history['image_loss'], label='Image Loss', color='green')
    plt.plot(range(epoch), history['affordance_loss'], label='Affordance Loss', color='orange')
    plt.plot(range(epoch), history['kl_loss'], label='KL Loss', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss History (Agent {agent}, Iteration {iteration})')
    plt.legend()
    
    # mean-IoUのプロット
    plt.subplot(2, 1, 2)
    plt.plot(range(epoch), history['mean_iou'], label='Mean-IoU', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Mean-IoU')
    plt.title(f'Training Mean-IoU History (Agent {agent}, Iteration {iteration})')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f'graph{agent}/training_history_{iteration}.png'))
    plt.close()
    
    # 損失の履歴を保存
    np.save(os.path.join(model_dir, f'npy/loss{agent}_{iteration}.npy'), history['total_loss'])
    np.save(os.path.join(model_dir, f'npy/mean_iou{agent}_{iteration}.npy'), history['mean_iou'])
    
    # モデルの保存
    torch.save(model.state_dict(), os.path.join(model_dir, f"pth/vae{agent}_{iteration}.pth"))
    
    # 全データに対する潜在変数を取得
    model.eval()
    with torch.no_grad():
        all_z = []
        all_labels = []
        all_preds = []
        all_targets = []
        
        for images, affordances, categories in all_loader:
            images = images.to(device)
            affordances = affordances.to(device)
            categories = categories.to(device)
            
            # 潜在変数と予測を取得
            mu, logvar = model.encode(images)
            z, _, _ = model.bottleneck(mu, logvar)
            _, affordance_pred, _, _, _ = model(images)
            
            all_z.append(z.cpu())
            all_labels.append(categories.cpu())
            all_preds.append(affordance_pred.argmax(dim=1).cpu())
            all_targets.append(affordances.cpu())
        
        # リストをテンソルに変換
        all_z = torch.cat(all_z, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        # 最終的なmean-IoUを計算
        final_mean_iou = calculate_mean_iou(
            torch.from_numpy(all_preds).to(device),
            torch.from_numpy(all_targets).to(device)
        )
        print(f"Final Mean-IoU: {final_mean_iou:.4f}")
        
        # アフォーダンス予測の可視化（最初の5サンプル）
        plt.figure(figsize=(15, 10))
        for i in range(min(5, len(all_preds))):
            plt.subplot(2, 5, i + 1)
            plt.imshow(all_targets[i], cmap='tab10', vmin=0, vmax=3)
            plt.title(f'Ground Truth (Sample {i+1})')
            plt.axis('off')
            
            plt.subplot(2, 5, i + 6)
            plt.imshow(all_preds[i], cmap='tab10', vmin=0, vmax=3)
            plt.title(f'Prediction (Sample {i+1})')
            plt.axis('off')
        
        plt.suptitle(f'Affordance Predictions (Agent {agent}, Iteration {iteration}, Mean-IoU: {final_mean_iou:.4f})')
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f'graph{agent}/affordance_predictions_{iteration}.png'))
        plt.close()
    
    return all_z, all_labels, history

def plot_latent(iteration, all_loader, model_dir, agent):
    # モデルの読み込み
    model = VAE().to(device)  # x_dimを明示的に指定
    
    # 保存されたモデルのパラメータを読み込む
    state_dict = torch.load(os.path.join(model_dir, f"pth/vae{agent}_{iteration}.pth"))
    
    # 新しいパラメータをモデルに読み込む
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 潜在変数とラベルの取得
    all_z = []
    all_labels = []
    
    with torch.no_grad():
        for images, affordances, categories in all_loader:
            images = images.to(device)
            affordances = affordances.to(device)
            categories = categories.to(device)
            
            # 潜在変数を取得
            mu, logvar = model.encode(images)
            z, _, _ = model.bottleneck(mu, logvar)
            
            all_z.append(z.cpu())
            all_labels.append(categories.cpu())
    
    # リストをテンソルに変換
    all_z = torch.cat(all_z, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # データセットから含まれるカテゴリを動的に取得
    unique_labels = np.unique(all_labels)
    print(f"データセットに含まれるカテゴリ: {unique_labels}")
    
    # カテゴリ名のマッピング（データセットに含まれるもののみ）
    category_names = {
        0: "bowl",
        1: "tvm", 
        2: "pan",
        3: "hammer",
        4: "knife",
        5: "cup",
        6: "drill",
        7: "racket",
        8: "spatula",
        9: "bottle"
    }
    
    # 実際に含まれるカテゴリのみをフィルタリング
    available_categories = {label: name for label, name in category_names.items() if label in unique_labels}
    print(f"利用可能なカテゴリ: {available_categories}")
    
    # t-SNEで2次元に削減
    print(f"t-SNEによる可視化を実行中...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_z)-1))
    z_tsne = tsne.fit_transform(all_z)
    
    # PCAで2次元に削減
    print(f"PCAによる可視化を実行中...")
    pca = PCA(n_components=2, random_state=42)
    z_pca = pca.fit_transform(all_z)
    
    # 説明分散比を計算
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # t-SNEプロット
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(z_tsne[:, 0], z_tsne[:, 1], c=all_labels, cmap='tab10', alpha=1.0, s=20)
    ax1.set_title(f't-SNE Visualization (Agent {agent}, Iteration {iteration})')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    # カラーバーを追加
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_ticks(list(available_categories.keys()))
    cbar1.set_ticklabels(list(available_categories.values()))
    ax1.grid(True, alpha=0.3)
    
    # PCAプロット
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(z_pca[:, 0], z_pca[:, 1], c=all_labels, cmap='tab10', alpha=1.0, s=20)
    ax2.set_title(f'PCA Visualization (Agent {agent}, Iteration {iteration})')
    ax2.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)')
    # カラーバーを追加
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_ticks(list(available_categories.keys()))
    cbar2.set_ticklabels(list(available_categories.values()))
    ax2.grid(True, alpha=0.3)
    
    # 累積説明分散比のプロット
    ax3 = axes[1, 0]
    n_components = min(20, all_z.shape[1])  # 最大20次元まで表示
    pca_full = PCA(n_components=n_components, random_state=42)
    pca_full.fit(all_z)
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
    
    ax3.plot(range(1, n_components + 1), cumulative_var, 'bo-', linewidth=2, markersize=6)
    ax3.set_title('Cumulative Explained Variance Ratio')
    ax3.set_xlabel('Number of Components')
    ax3.set_ylabel('Cumulative Explained Variance Ratio')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # 各クラスの分布（PCA空間での密度）
    ax4 = axes[1, 1]
    for label, name in available_categories.items():
        mask = all_labels == label
        if np.sum(mask) > 0:  # データが存在する場合のみ
            # tab10カラーマップから色を取得
            color = plt.cm.tab10(label / 10.0)  # 0-9のラベルを0-1に正規化
            ax4.hist(z_pca[mask, 0], bins=20, alpha=0.5, label=f"{name} ({np.sum(mask)})", color=color)
    ax4.set_title('Distribution of PC1 by Class')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('Frequency')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(model_dir, f'graph{agent}/latent_space_{iteration}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()