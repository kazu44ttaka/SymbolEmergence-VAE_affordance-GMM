import os
import numpy as np
from scipy.stats import wishart, multivariate_normal
import matplotlib.pyplot as plt
from tool import calc_ari, cmx
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.cluster import adjusted_rand_score as ari
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
import argparse
from tool import visualize_gmm
import cnn_vae_module_affordance as vae_module
import multiprocessing
from functools import lru_cache
import yaml
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class CachedAffordanceDataset(Dataset):
    def __init__(self, root_dir, transform=None, cache_size=1000):
        self.root_dir = root_dir
        self.transform = transform
        self.cache_size = cache_size
        
        # ディレクトリの設定
        self.rgb_dir = os.path.join(root_dir, 'rgb')
        self.affordance_dir = os.path.join(root_dir, 'affordances')
        self.metadata_dir = os.path.join(root_dir, 'metadata')
        
        # 有効なファイルペアを取得
        self.valid_pairs = []
        rgb_files = os.listdir(self.rgb_dir)
        
        for rgb_file in rgb_files:
            if rgb_file.endswith('.jpg'):
                base_name = rgb_file[:-4]
                affordance_file = f"{base_name}.txt"
                metadata_file = f"{base_name}.txt"
                
                affordance_path = os.path.join(self.affordance_dir, affordance_file)
                metadata_path = os.path.join(self.metadata_dir, metadata_file)
                
                if os.path.exists(affordance_path) and os.path.exists(metadata_path):
                    self.valid_pairs.append((rgb_file, affordance_file, metadata_file))
        
        # キャッシュの初期化
        self.cache = {}
        self.cache_order = []
    
    def __len__(self):
        return len(self.valid_pairs)
    
    @lru_cache(maxsize=1000)
    def _load_affordance_map(self, affordance_path):
        """アフォーダンスマップの読み込みをキャッシュ"""
        return np.loadtxt(affordance_path)
    
    @lru_cache(maxsize=1000)
    def _load_metadata(self, metadata_path):
        """メタデータの読み込みをキャッシュ"""
        with open(metadata_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _update_cache(self, key, value):
        """キャッシュの更新（LRU方式）"""
        if len(self.cache) >= self.cache_size:
            # 最も古いアイテムを削除
            oldest_key = self.cache_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = value
        self.cache_order.append(key)
    
    def __getitem__(self, idx):
        rgb_file, affordance_file, metadata_file = self.valid_pairs[idx]
        
        # キャッシュキーの生成
        cache_key = f"{rgb_file}_{affordance_file}_{metadata_file}"
        
        # キャッシュをチェック
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # RGB画像の読み込み
        image_path = os.path.join(self.rgb_dir, rgb_file)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # アフォーダンスマップの読み込み（キャッシュを使用）
        affordance_path = os.path.join(self.affordance_dir, affordance_file)
        affordance_map = self._load_affordance_map(affordance_path)
        affordance_map = torch.from_numpy(affordance_map).long()
        affordance_map = F.interpolate(
            affordance_map.unsqueeze(0).unsqueeze(0).float(),
            size=(64, 64),
            mode='nearest'
        ).squeeze(0).squeeze(0).long()
        
        # メタデータの読み込み（キャッシュを使用）
        metadata_path = os.path.join(self.metadata_dir, metadata_file)
        metadata = self._load_metadata(metadata_path)
        category_id = metadata['object_id']
        
        # 結果をキャッシュに保存
        result = (image, affordance_map, category_id)
        self._update_cache(cache_key, result)
        
        return result

def get_experiment_dir(args):
    """ハイパーパラメータに基づいて実験ディレクトリ名を生成"""
    # 主要なハイパーパラメータを含むディレクトリ名を生成
    dir_name = (
        f"affordances_"
        f"bs{args.batch_size}_"
        f"vae{args.vae_iter}_"
        f"mh{args.mh_iter}_"
        f"cat{args.category}_"
        f"mode{args.mode}_"
        f"aw{args.affordance_weight:.1f}_"
        f"{args.data_dir}"
    )
    return dir_name

def main():
    parser = argparse.ArgumentParser(description='Symbol emergence based on VAE+GMM for Affordances Dataset')
    parser.add_argument('--batch-size', type=int, default=64, metavar='B', help='input batch size for training')
    parser.add_argument('--vae-iter', type=int, default=50, metavar='V', help='number of VAE iteration')
    parser.add_argument('--mh-iter', type=int, default=50, metavar='M', help='number of M-H mgmm iteration')
    parser.add_argument('--category', type=int, default=5, metavar='K', help='number of category for GMM module')
    parser.add_argument('--mode', type=int, default=-1, metavar='M', help='0:All reject, 1:ALL accept')
    parser.add_argument('--debug', type=bool, default=False, metavar='D', help='Debug mode')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=2, metavar='S', help='random seed')
    parser.add_argument('--data-dir', type=str, default='simple_dataset', help='path to dataset directory')
    parser.add_argument('--affordance-weight', type=float, default=1.0, help='weight for affordance prediction loss')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    print("CUDA", args.cuda)
    if args.debug is True:
        args.vae_iter = 2
        args.mh_iter = 2

    ############################## Making directory ##############################
    # 実験ディレクトリ名の生成
    experiment_dir = get_experiment_dir(args)
    model_dir = "./model"
    dir_name = os.path.join(model_dir, experiment_dir)
    
    # サブディレクトリの設定
    graphA_dir = os.path.join(dir_name, "graphA")
    graphB_dir = os.path.join(dir_name, "graphB")
    pth_dir = os.path.join(dir_name, "pth")
    npy_dir = os.path.join(dir_name, "npy")
    reconA_dir = os.path.join(dir_name, "reconA")
    reconB_dir = os.path.join(dir_name, "reconB")
    log_dir = os.path.join(dir_name, "log")
    result_dir = os.path.join(dir_name, "result")

    # ディレクトリの作成
    for d in [model_dir, dir_name, pth_dir, graphA_dir, graphB_dir, npy_dir, 
            reconA_dir, reconB_dir, log_dir, result_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    # 実験設定を保存
    config_file = os.path.join(dir_name, "config.txt")
    with open(config_file, 'w') as f:
        f.write("Experiment Configuration:\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"VAE Iterations: {args.vae_iter}\n")
        f.write(f"MH Iterations: {args.mh_iter}\n")
        f.write(f"Categories: {args.category}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Affordance Weight: {args.affordance_weight}\n")
        f.write(f"CUDA Enabled: {args.cuda}\n")
        f.write(f"Random Seed: {args.seed}\n")
        f.write(f"Data Directory: {args.data_dir}\n")

    print(f"Results will be saved in: {dir_name}")
    print("Configuration saved to:", config_file)

    ############################## Preparing Dataset ##############################
    print("Dataset: IIT Affordances")

    # データの前処理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # キャッシュ付きデータセットの使用
    dataset = CachedAffordanceDataset(
        root_dir=f"./{args.data_dir}",
        transform=transform,
        cache_size=2000  # キャッシュサイズを増やす
    )
    n_samples = len(dataset)
    print(f"Total number of samples: {n_samples}")

    # データの分割（80%を学習に使用）
    train_size = int(0.8 * n_samples)
    val_size = n_samples - train_size
    print(f"Train size: {train_size}, Validation size: {val_size}")

    # データの分割
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # ワーカー数の最適化
    num_workers = min(multiprocessing.cpu_count(), 4)  # CPUコア数と4の小さい方

    # エージェントAとBのデータローダー（最適化された設定）
    train_loader1 = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=args.cuda,
        persistent_workers=True
    )
    train_loader2 = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=args.cuda,
        persistent_workers=True
    )

    all_loader1 = DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=args.cuda,
        persistent_workers=True
    )

    all_loader2 = DataLoader(   
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=args.cuda,
        persistent_workers=True
    )

    # マルチプロセッシングの設定
    torch.multiprocessing.set_start_method('spawn', force=True)

    print(f"Category: {args.category}")
    print(f"VAE_iter: {args.vae_iter}, Batch_size: {args.batch_size}")
    print(f"MH_iter: {args.mh_iter}, MH_mode: {args.mode}(-1:Com 0:No-com 1:All accept)")

    mutual_iteration = 10
    mu_d_A = np.zeros((train_size))
    var_d_A = np.zeros((train_size))
    mu_d_B = np.zeros((train_size))
    var_d_B = np.zeros((train_size))

    # クラスIDのマッピングを作成（連続したIDに変換）
    class_id_mapping = {old_id: new_id for new_id, old_id in enumerate(np.unique([dataset[i][2] for i in range(len(dataset))]))}
    reverse_mapping = {new_id: old_id for old_id, new_id in class_id_mapping.items()}
    def remap_class_ids(labels):
        return np.array([class_id_mapping[label] for label in labels])

    def analyze_class_difficulty(y_true, y_pred, agent_name, iteration, save_dir):
        """クラスごとの難易度を分析する関数"""
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        class_metrics = {}
        for i in range(len(cm)):
            if str(i) in report:
                class_metrics[i] = {
                    'precision': report[str(i)]['precision'],
                    'recall': report[str(i)]['recall'],
                    'f1-score': report[str(i)]['f1-score'],
                    'support': report[str(i)]['support']
                }
            else:
                class_metrics[i] = {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1-score': 0.0,
                    'support': 0
                }
        difficulty_ranking = sorted(class_metrics.items(), key=lambda x: x[1]['f1-score'])
        analysis_file = os.path.join(save_dir, f'class_analysis_{agent_name}_{iteration}.txt')
        with open(analysis_file, 'w') as f:
            f.write(f"=== Class Difficulty Analysis for {agent_name} (Iteration {iteration}) ===\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_true, y_pred, zero_division=0))
            f.write("\n\nDifficulty Ranking (Hardest to Easiest):\n")
            f.write("Class | F1-Score | Precision | Recall | Support\n")
            f.write("-" * 50 + "\n")
            for class_id, metrics in difficulty_ranking:
                f.write(f"{class_id:5d} | {metrics['f1-score']:8.3f} | {metrics['precision']:9.3f} | {metrics['recall']:6.3f} | {metrics['support']:7.0f}\n")
            f.write(f"\n\nMost Difficult Classes (F1 < 0.5):\n")
            difficult_classes = [c for c, m in difficulty_ranking if m['f1-score'] < 0.5]
            if difficult_classes:
                f.write(f"Classes: {difficult_classes}\n")
                f.write("These classes show poor clustering performance.\n")
            else:
                f.write("All classes show reasonable clustering performance.\n")
            f.write(f"\n\nClass Distribution:\n")
            unique, counts = np.unique(y_true, return_counts=True)
            for class_id, count in zip(unique, counts):
                f.write(f"Class {class_id}: {count} samples\n")
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {agent_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.subplot(2, 2, 2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
        plt.title(f'Normalized Confusion Matrix - {agent_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.subplot(2, 2, 3)
        classes = list(class_metrics.keys())
        f1_scores = [class_metrics[c]['f1-score'] for c in classes]
        plt.bar(classes, f1_scores)
        plt.title(f'F1-Score by Class - {agent_name}')
        plt.xlabel('Class')
        plt.ylabel('F1-Score')
        plt.ylim(0, 1)
        plt.subplot(2, 2, 4)
        precisions = [class_metrics[c]['precision'] for c in classes]
        recalls = [class_metrics[c]['recall'] for c in classes]
        plt.scatter(precisions, recalls, s=100, alpha=0.7)
        for i, c in enumerate(classes):
            plt.annotate(f'C{c}', (precisions[i], recalls[i]), xytext=(5, 5), textcoords='offset points')
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title(f'Precision vs Recall by Class - {agent_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'detailed_confusion_{agent_name}_{iteration}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10, 8))
        error_matrix = cm.copy()
        np.fill_diagonal(error_matrix, 0)
        sns.heatmap(error_matrix, annot=True, fmt='d', cmap='Reds')
        plt.title(f'Error Analysis - {agent_name} (Non-diagonal elements show misclassifications)')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.savefig(os.path.join(save_dir, f'error_analysis_{agent_name}_{iteration}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        return class_metrics, difficulty_ranking

    for it in range(mutual_iteration):
        print(f"------------------Mutual learning session {it} begins------------------")
        ############################## Training VAE ##############################
        # エージェントAのVAE学習
        c_nd_A, label, loss_list = vae_module.train(
            iteration=it,
            gmm_mu=torch.from_numpy(mu_d_A),
            gmm_var=torch.from_numpy(var_d_A),
            epoch=args.vae_iter,
            train_loader=train_loader1,
            batch_size=args.batch_size,
            all_loader=all_loader1,
            model_dir=dir_name,
            agent="A",
            affordance_weight=args.affordance_weight,
        )
        c_nd_B, label, loss_list = vae_module.train(
            iteration=it,
            gmm_mu=torch.from_numpy(mu_d_B),
            gmm_var=torch.from_numpy(var_d_B),
            epoch=args.vae_iter,
            train_loader=train_loader2,
            batch_size=args.batch_size,
            all_loader=all_loader2,
            model_dir=dir_name,
            agent="B",
            affordance_weight=args.affordance_weight,
        )
        # 潜在空間の可視化
        vae_module.plot_latent(iteration=it, all_loader=all_loader1, model_dir=dir_name, agent="A")
        vae_module.plot_latent(iteration=it, all_loader=all_loader2, model_dir=dir_name, agent="B")
        
        K = args.category  # カテゴリ数
        z_truth_n = label  # 真のラベル（オブジェクトクラス）
        dim = len(c_nd_A[0])  # VAEの潜在空間の次元

        ############################## Initializing parameters ##############################
        # ハイパーパラメータの設定
        beta = 1.0
        m_d_A = np.repeat(0.0, dim)
        m_d_B = np.repeat(0.0, dim)
        w_dd_A = np.identity(dim) * 0.1
        w_dd_B = np.identity(dim) * 0.1
        nu = dim

        # μ, Λの初期化
        mu_kd_A = np.empty((K, dim))
        lambda_kdd_A = np.empty((K, dim, dim))
        mu_kd_B = np.empty((K, dim))
        lambda_kdd_B = np.empty((K, dim, dim))
        
        for k in range(K):
            lambda_kdd_A[k] = wishart.rvs(df=nu, scale=w_dd_A, size=1)
            lambda_kdd_B[k] = wishart.rvs(df=nu, scale=w_dd_B, size=1)
            mu_kd_A[k] = np.random.multivariate_normal(mean=m_d_A, cov=np.linalg.inv(beta * lambda_kdd_A[k])).flatten()
            mu_kd_B[k] = np.random.multivariate_normal(mean=m_d_B, cov=np.linalg.inv(beta * lambda_kdd_B[k])).flatten()

        # wの初期化
        w_dk_A = np.random.multinomial(1, [1/K]*K, size=train_size)
        w_dk_B = np.random.multinomial(1, [1/K]*K, size=train_size)

        # 学習パラメータの初期化
        beta_hat_k_A = np.zeros(K)
        beta_hat_k_B = np.zeros(K)
        m_hat_kd_A = np.zeros((K, dim))
        m_hat_kd_B = np.zeros((K, dim))
        w_hat_kdd_A = np.zeros((K, dim, dim))
        w_hat_kdd_B = np.zeros((K, dim, dim))
        nu_hat_k_A = np.zeros(K)
        nu_hat_k_B = np.zeros(K)
        tmp_eta_nB = np.zeros((K, train_size))
        eta_dkB = np.zeros((train_size, K))
        tmp_eta_nA = np.zeros((K, train_size))
        eta_dkA = np.zeros((train_size, K))
        log_cat_liks_A = np.zeros(train_size)
        log_cat_liks_B = np.zeros(train_size)
        mu_d_A = np.zeros((train_size, dim))
        var_d_A = np.zeros((train_size, dim))
        mu_d_B = np.zeros((train_size, dim))
        var_d_B = np.zeros((train_size, dim))

        iteration = args.mh_iter
        ARI_A = np.zeros((iteration))
        ARI_B = np.zeros((iteration))
        concidence = np.zeros((iteration))
        accept_count_AtoB = np.zeros((iteration))
        accept_count_BtoA = np.zeros((iteration))

        ############################## M-H algorithm ##############################
        print(f"M-H algorithm Start({it}): Epoch:{iteration}")
        for i in range(iteration):
            pred_label_A = []
            pred_label_B = []
            count_AtoB = count_BtoA = 0

            """~~~~~~~~~~~~~~~~~~~~~~~~~~~~Speaker:A -> Listener:B~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
            w_dk = np.random.multinomial(1, [1/K]*K, size=train_size)
            for k in range(K):
                # 数値的な安定性のために、対数空間で計算
                tmp_eta_nA[k] = np.diag(-0.5 * (c_nd_A - mu_kd_A[k]).dot(lambda_kdd_A[k]).dot((c_nd_A - mu_kd_A[k]).T)).copy()
                tmp_eta_nA[k] += 0.5 * np.log(np.linalg.det(lambda_kdd_A[k]) + 1e-7)
                # 最大値を引いて数値的な安定性を確保
                max_log_prob = np.max(tmp_eta_nA[k])
                eta_dkA[:, k] = np.exp(tmp_eta_nA[k] - max_log_prob)
            
            # 正規化（数値的な安定性のために、小さい値を足す）
            eta_dkA = eta_dkA + 1e-10
            eta_dkA /= np.sum(eta_dkA, axis=1, keepdims=True)

            for d in range(train_size):
                # 確率が0にならないように、小さい値を足す
                pvals = eta_dkA[d] + 1e-10
                pvals /= np.sum(pvals)
                w_dk_A[d] = np.random.multinomial(n=1, pvals=pvals, size=1).flatten()

                if args.mode == 0:
                    pred_label_A.append(np.argmax(w_dk_A[d]))
                elif args.mode == 1:
                    w_dk[d] = w_dk_A[d]
                    count_AtoB = count_AtoB + 1
                    pred_label_B.append(np.argmax(w_dk[d]))
                else:
                    # 数値的な安定性のために、対数確率を使用
                    log_cat_liks_A = multivariate_normal.logpdf(c_nd_B[d],
                                    mean=mu_kd_B[np.argmax(w_dk_A[d])],
                                    cov=np.linalg.inv(lambda_kdd_B[np.argmax(w_dk_A[d])]),
                                    allow_singular=True)
                    log_cat_liks_B = multivariate_normal.logpdf(c_nd_B[d],
                                    mean=mu_kd_B[np.argmax(w_dk_B[d])],
                                    cov=np.linalg.inv(lambda_kdd_B[np.argmax(w_dk_B[d])]),
                                    allow_singular=True)
                    
                    # 対数空間での除算（引き算）を行い、その後expを取る
                    log_judge_r = log_cat_liks_A - log_cat_liks_B
                    judge_r = np.exp(min(0, log_judge_r))  # 1を超えないようにする
                    
                    rand_u = np.random.rand()
                    if judge_r >= rand_u:
                        w_dk[d] = w_dk_A[d]
                        count_AtoB = count_AtoB + 1
                    else:
                        w_dk[d] = w_dk_B[d]
                    pred_label_B.append(np.argmax(w_dk[d]))

            if args.mode == -1 or args.mode == 1:
                for k in range(K):
                    beta_hat_k_B[k] = np.sum(w_dk[:, k]) + beta
                    m_hat_kd_B[k] = np.sum(w_dk[:, k] * c_nd_B.T, axis=1)
                    m_hat_kd_B[k] += beta * m_d_B
                    m_hat_kd_B[k] /= beta_hat_k_B[k]
                    tmp_w_dd_B = np.dot((w_dk[:, k] * c_nd_B.T), c_nd_B)
                    tmp_w_dd_B += beta * np.dot(m_d_B.reshape(dim, 1), m_d_B.reshape(1, dim))
                    tmp_w_dd_B -= beta_hat_k_B[k] * np.dot(m_hat_kd_B[k].reshape(dim, 1), m_hat_kd_B[k].reshape(1, dim))
                    tmp_w_dd_B += np.linalg.inv(w_dd_B)
                    w_hat_kdd_B[k] = np.linalg.inv(tmp_w_dd_B)
                    nu_hat_k_B[k] = np.sum(w_dk[:, k]) + nu

                    # λ^Bとμ^Bのサンプリング
                    lambda_kdd_B[k] = wishart.rvs(size=1, df=nu_hat_k_B[k], scale=w_hat_kdd_B[k])
                    mu_kd_B[k] = np.random.multivariate_normal(mean=m_hat_kd_B[k],
                                                            cov=np.linalg.inv(beta_hat_k_B[k] * lambda_kdd_B[k]),
                                                            size=1).flatten()

            if args.mode == 0:  # No com
                for k in range(K):
                    beta_hat_k_A[k] = np.sum(w_dk_A[:, k]) + beta
                    m_hat_kd_A[k] = np.sum(w_dk_A[:, k] * c_nd_A.T, axis=1)
                    m_hat_kd_A[k] += beta * m_d_A
                    m_hat_kd_A[k] /= beta_hat_k_A[k]
                    tmp_w_dd_A = np.dot((w_dk_A[:, k] * c_nd_A.T), c_nd_A)
                    tmp_w_dd_A += beta * np.dot(m_d_A.reshape(dim, 1), m_d_A.reshape(1, dim))
                    tmp_w_dd_A -= beta_hat_k_A[k] * np.dot(m_hat_kd_A[k].reshape(dim, 1), m_hat_kd_A[k].reshape(1, dim))
                    tmp_w_dd_A += np.linalg.inv(w_dd_A)
                    w_hat_kdd_A[k] = np.linalg.inv(tmp_w_dd_A)
                    nu_hat_k_A[k] = np.sum(w_dk_A[:, k]) + nu

                    # λ^Aとμ^Aのサンプリング
                    lambda_kdd_A[k] = wishart.rvs(size=1, df=nu_hat_k_A[k], scale=w_hat_kdd_A[k])
                    mu_kd_A[k] = np.random.multivariate_normal(mean=m_hat_kd_A[k],
                                                            cov=np.linalg.inv(beta_hat_k_A[k] * lambda_kdd_A[k]),
                                                            size=1).flatten()

            """~~~~~~~~~~~~~~~~~~~~~~~~~~~~Speaker:B -> Listener:A~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
            w_dk = np.random.multinomial(1, [1/K]*K, size=train_size)
            for k in range(K):
                # 数値的な安定性のために、対数空間で計算
                tmp_eta_nB[k] = np.diag(-0.5 * (c_nd_B - mu_kd_B[k]).dot(lambda_kdd_B[k]).dot((c_nd_B - mu_kd_B[k]).T)).copy()
                tmp_eta_nB[k] += 0.5 * np.log(np.linalg.det(lambda_kdd_B[k]) + 1e-7)
                # 最大値を引いて数値的な安定性を確保
                max_log_prob = np.max(tmp_eta_nB[k])
                eta_dkB[:, k] = np.exp(tmp_eta_nB[k] - max_log_prob)
            
            # 正規化（数値的な安定性のために、小さい値を足す）
            eta_dkB = eta_dkB + 1e-10
            eta_dkB /= np.sum(eta_dkB, axis=1, keepdims=True)

            for d in range(train_size):
                # 確率が0にならないように、小さい値を足す
                pvals = eta_dkB[d] + 1e-10
                pvals /= np.sum(pvals)
                w_dk_B[d] = np.random.multinomial(n=1, pvals=pvals, size=1).flatten()

                if args.mode == 0:
                    pred_label_B.append(np.argmax(w_dk_B[d]))
                elif args.mode == 1:
                    w_dk[d] = w_dk_B[d]
                    count_BtoA = count_BtoA + 1
                    pred_label_A.append(np.argmax(w_dk[d]))
                else:
                    # 数値的な安定性のために、対数確率を使用
                    log_cat_liks_B = multivariate_normal.logpdf(c_nd_A[d],
                                    mean=mu_kd_A[np.argmax(w_dk_B[d])],
                                    cov=np.linalg.inv(lambda_kdd_A[np.argmax(w_dk_B[d])]),
                                    allow_singular=True)
                    log_cat_liks_A = multivariate_normal.logpdf(c_nd_A[d],
                                    mean=mu_kd_A[np.argmax(w_dk_A[d])],
                                    cov=np.linalg.inv(lambda_kdd_A[np.argmax(w_dk_A[d])]),
                                    allow_singular=True)
                    
                    # 対数空間での除算（引き算）を行い、その後expを取る
                    log_judge_r = log_cat_liks_B - log_cat_liks_A
                    judge_r = np.exp(min(0, log_judge_r))  # 1を超えないようにする
                    
                    rand_u = np.random.rand()
                    if judge_r >= rand_u:
                        w_dk[d] = w_dk_B[d]
                        count_BtoA = count_BtoA + 1
                    else:
                        w_dk[d] = w_dk_A[d]
                    pred_label_A.append(np.argmax(w_dk[d]))

            if args.mode == -1 or args.mode == 1:
                for k in range(K):
                    beta_hat_k_A[k] = np.sum(w_dk[:, k]) + beta
                    m_hat_kd_A[k] = np.sum(w_dk[:, k] * c_nd_A.T, axis=1)
                    m_hat_kd_A[k] += beta * m_d_A
                    m_hat_kd_A[k] /= beta_hat_k_A[k]
                    tmp_w_dd_A = np.dot((w_dk[:, k] * c_nd_A.T), c_nd_A)
                    tmp_w_dd_A += beta * np.dot(m_d_A.reshape(dim, 1), m_d_A.reshape(1, dim))
                    tmp_w_dd_A -= beta_hat_k_A[k] * np.dot(m_hat_kd_A[k].reshape(dim, 1), m_hat_kd_A[k].reshape(1, dim))
                    tmp_w_dd_A += np.linalg.inv(w_dd_A)
                    w_hat_kdd_A[k] = np.linalg.inv(tmp_w_dd_A)
                    nu_hat_k_A[k] = np.sum(w_dk[:, k]) + nu

                    # λ^Aとμ^Aのサンプリング
                    lambda_kdd_A[k] = wishart.rvs(size=1, df=nu_hat_k_A[k], scale=w_hat_kdd_A[k])
                    mu_kd_A[k] = np.random.multivariate_normal(mean=m_hat_kd_A[k],
                                                            cov=np.linalg.inv(beta_hat_k_A[k] * lambda_kdd_A[k]),
                                                            size=1).flatten()

            if args.mode == 0:  # No com
                for k in range(K):
                    beta_hat_k_B[k] = np.sum(w_dk_B[:, k]) + beta
                    m_hat_kd_B[k] = np.sum(w_dk_B[:, k] * c_nd_B.T, axis=1)
                    m_hat_kd_B[k] += beta * m_d_B
                    m_hat_kd_B[k] /= beta_hat_k_B[k]
                    tmp_w_dd_B = np.dot((w_dk_B[:, k] * c_nd_B.T), c_nd_B)
                    tmp_w_dd_B += beta * np.dot(m_d_B.reshape(dim, 1), m_d_B.reshape(1, dim))
                    tmp_w_dd_B -= beta_hat_k_B[k] * np.dot(m_hat_kd_B[k].reshape(dim, 1), m_hat_kd_B[k].reshape(1, dim))
                    tmp_w_dd_B += np.linalg.inv(w_dd_B)
                    w_hat_kdd_B[k] = np.linalg.inv(tmp_w_dd_B)
                    nu_hat_k_B[k] = np.sum(w_dk_B[:, k]) + nu

                    # λ^Bとμ^Bのサンプリング
                    lambda_kdd_B[k] = wishart.rvs(size=1, df=nu_hat_k_B[k], scale=w_hat_kdd_B[k])
                    mu_kd_B[k] = np.random.multivariate_normal(mean=m_hat_kd_B[k],
                                                            cov=np.linalg.inv(beta_hat_k_B[k] * lambda_kdd_B[k]),
                                                            size=1).flatten()

            ############################## Evaluation ##############################
            if z_truth_n is not None:
                # クラスIDを連続したIDに変換
                z_truth_n_remapped = remap_class_ids(z_truth_n)
                _, result_a = calc_ari(pred_label_A, z_truth_n_remapped)
                _, result_b = calc_ari(pred_label_B, z_truth_n_remapped)
                concidence[i] = np.round(cohen_kappa_score(pred_label_A, pred_label_B), 3)
                ARI_A[i] = np.round(ari(z_truth_n_remapped, result_a), 3)
                ARI_B[i] = np.round(ari(z_truth_n_remapped, result_b), 3)
                # 最後のステップ（エポック）に一回だけクラス難易度分析を実行
                if i == (iteration-1):
                    print(f"Final Epoch {i+1}: Analyzing class difficulty (ARI_A: {ARI_A[i]:.3f}, ARI_B: {ARI_B[i]:.3f})...")
                    class_metrics_A, difficulty_ranking_A = analyze_class_difficulty(
                        z_truth_n_remapped, result_a, "A", f"{it}_final", result_dir
                    )
                    print(f"Agent A - Most difficult classes: {[c for c, m in difficulty_ranking_A[:3]]}")
                    class_metrics_B, difficulty_ranking_B = analyze_class_difficulty(
                        z_truth_n_remapped, result_b, "B", f"{it}_final", result_dir
                    )
                    print(f"Agent B - Most difficult classes: {[c for c, m in difficulty_ranking_B[:3]]}")
                    comparison_file = os.path.join(result_dir, f'agent_comparison_{it}_final.txt')
                    with open(comparison_file, 'w') as f:
                        f.write(f"=== Final Agent Comparison Analysis (Iteration {it}) ===\n\n")
                        f.write(f"ARI_A: {ARI_A[i]:.3f}, ARI_B: {ARI_B[i]:.3f}\n\n")
                        f.write("Class ID Mapping:\n")
                        for old_id, new_id in class_id_mapping.items():
                            f.write(f"  Original ID {old_id} -> New ID {new_id}\n")
                        f.write("\n")
                        f.write("Common Difficult Classes:\n")
                        difficult_A = [c for c, m in difficulty_ranking_A if m['f1-score'] < 0.5]
                        difficult_B = [c for c, m in difficulty_ranking_B if m['f1-score'] < 0.5]
                        common_difficult = set(difficult_A) & set(difficult_B)
                        f.write(f"Classes difficult for both agents: {list(common_difficult)}\n\n")
                        f.write("Agent-specific Difficult Classes:\n")
                        only_A = set(difficult_A) - set(difficult_B)
                        only_B = set(difficult_B) - set(difficult_A)
                        f.write(f"Only Agent A struggles with: {list(only_A)}\n")
                        f.write(f"Only Agent B struggles with: {list(only_B)}\n")
                        f.write(f"\nFinal Results Summary:\n")
                        f.write(f"Final Epoch: {i+1}/{iteration}\n")
                        f.write(f"Best ARI_A: {max(ARI_A):.3f}\n")
                        f.write(f"Best ARI_B: {max(ARI_B):.3f}\n")
                        f.write(f"Final Kappa: {concidence[i]:.3f}\n")
                        f.write(f"\nFinal Class Performance Summary:\n")
                        f.write("Class | A_F1 | B_F1 | Difficulty\n")
                        f.write("-" * 35 + "\n")
                        for class_id in range(len(class_metrics_A)):
                            a_f1 = class_metrics_A[class_id]['f1-score']
                            b_f1 = class_metrics_B[class_id]['f1-score']
                            avg_f1 = (a_f1 + b_f1) / 2
                            difficulty = "Easy" if avg_f1 > 0.7 else "Medium" if avg_f1 > 0.5 else "Hard"
                            original_id = reverse_mapping.get(class_id, class_id)
                            f.write(f"{original_id:5d} | {a_f1:4.3f} | {b_f1:4.3f} | {difficulty}\n")
            else:
                # ラベルがない場合は、クラスタリング結果のみを評価
                concidence[i] = np.round(cohen_kappa_score(pred_label_A, pred_label_B), 3)
                ARI_A[i] = 0.0
                ARI_B[i] = 0.0

            # 受容回数
            accept_count_AtoB[i] = count_AtoB
            accept_count_BtoA[i] = count_BtoA

            if i == 0 or (i+1) % 10 == 0 or i == (iteration-1):
                print(f"=> Epoch: {i+1}, ARI_A: {ARI_A[i]}, ARI_B: {ARI_B[i]}, Kappa:{concidence[i]}, "
                    f"A2B:{int(accept_count_AtoB[i])}, B2A:{int(accept_count_BtoA[i])}")

            for d in range(train_size):
                mu_d_A[d] = mu_kd_A[np.argmax(w_dk[d])]
                var_d_A[d] = np.diag(np.linalg.inv(lambda_kdd_A[np.argmax(w_dk[d])]))
                mu_d_B[d] = mu_kd_B[np.argmax(w_dk[d])]
                var_d_B[d] = np.diag(np.linalg.inv(lambda_kdd_B[np.argmax(w_dk[d])]))

        # パラメータの保存
        np.save(os.path.join(npy_dir, f'muA_{it}.npy'), mu_kd_A)
        np.save(os.path.join(npy_dir, f'muB_{it}.npy'), mu_kd_B)
        np.save(os.path.join(npy_dir, f'lambdaA_{it}.npy'), lambda_kdd_A)
        np.save(os.path.join(npy_dir, f'lambdaB_{it}.npy'), lambda_kdd_B)

        # 評価指標の保存
        np.savetxt(os.path.join(log_dir, f"ariA{it}.txt"), ARI_A, fmt='%.3f')
        np.savetxt(os.path.join(log_dir, f"ariB{it}.txt"), ARI_B, fmt='%.3f')
        np.savetxt(os.path.join(log_dir, f"kappa{it}.txt"), concidence, fmt='%.3f')

        ############################## Plot ##############################
        # 受容回数のプロット
        plt.figure()
        plt.plot(range(0, iteration), accept_count_AtoB, marker="None", label="Accept_num:AtoB")
        plt.plot(range(0, iteration), accept_count_BtoA, marker="None", label="Accept_num:BtoA")
        plt.xlabel('iteration')
        plt.ylabel('Number of acceptation')
        plt.ylim(0, train_size)
        plt.legend()
        plt.savefig(os.path.join(result_dir, f'accept{it}.png'))
        plt.close()

        # 一致度のプロット
        plt.figure()
        plt.plot(range(0, iteration), concidence, marker="None")
        plt.xlabel('iteration')
        plt.ylabel('Concidence')
        plt.ylim(0, 1)
        plt.title('Kappa')
        plt.savefig(os.path.join(result_dir, f"conf{it}.png"))
        plt.close()

        # ARIのプロット
        if z_truth_n is not None:
            plt.figure()
            plt.plot(range(0, iteration), ARI_A, marker="None", label="ARI_A")
            plt.plot(range(0, iteration), ARI_B, marker="None", label="ARI_B")
            plt.xlabel('iteration')
            plt.ylabel('ARI')
            plt.ylim(0, 1)
            plt.legend()
            plt.title('ARI')
            plt.savefig(os.path.join(result_dir, f"ari{it}.png"))
            plt.close()

            # 混同行列のプロット
            cmx(iteration=it, y_true=z_truth_n, y_pred=result_a, agent="A", save_dir=result_dir)
            cmx(iteration=it, y_true=z_truth_n, y_pred=result_b, agent="B", save_dir=result_dir)

        print(f"Iteration:{it} Done: max_ARI_A: {max(ARI_A)}, max_ARI_B: {max(ARI_B)}, max_Kappa:{max(concidence)}") 

if __name__ == "__main__":
    main()