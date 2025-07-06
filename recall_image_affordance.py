import os
import numpy as np
from scipy.stats import wishart 
import matplotlib.pyplot as plt
from tool import calc_ari, visualize_gmm
from sklearn.metrics import cohen_kappa_score
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split, Dataset
import argparse
from PIL import Image
import yaml
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# アフォーダンスデータセット用のクラス
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
            affordance_map = np.loadtxt(affordance_path)
            affordance_map = torch.from_numpy(affordance_map).long()
            affordance_map = F.interpolate(
                affordance_map.unsqueeze(0).unsqueeze(0).float(),
                size=(64, 64),
                mode='nearest'
            ).squeeze(0).squeeze(0).long()
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

parser = argparse.ArgumentParser(description='Symbol emergence based on VAE+GMM for Affordances Dataset')
parser.add_argument('--batch-size', type=int, default=64, metavar='B', help='input batch size for training')
parser.add_argument('--vae-iter', type=int, default=50, metavar='V', help='number of VAE iteration')
parser.add_argument('--mh-iter', type=int, default=50, metavar='M', help='number of M-H mgmm iteration')
parser.add_argument('--category', type=int, default=5, metavar='K', help='number of category for GMM module')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--data-dir', type=str, default='simple_dataset', help='path to dataset directory')
parser.add_argument('--load-iteration', type=int, default=3, help='iteration to load for recall')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

############################## Making directory ##############################

# 実験ディレクトリ名の生成（main.pyと同じ形式）
def get_experiment_dir(args):
    dir_name = (
        f"affordances_"
        f"bs{args.batch_size}_"
        f"vae{args.vae_iter}_"
        f"mh{args.mh_iter}_"
        f"cat{args.category}_"
        f"mode-1_"
        f"aw1.0_"
        f"{args.data_dir}"
    )
    return dir_name

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

############################## Preparing Dataset ##############################

print("Dataset: IIT Affordances")

# データの前処理
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# データセットの読み込み
dataset = AffordanceDataset(
    root_dir=f"./{args.data_dir}",
    transform=transform
)
n_samples = len(dataset)
print(f"Total number of samples: {n_samples}")

# データの分割（80%を学習に使用）
train_size = int(0.8 * n_samples)
val_size = n_samples - train_size
print(f"Train size: {train_size}, Validation size: {val_size}")

# データの分割
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# データローダーの作成
train_loader1 = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,  # シンプルにするため0に設定
    pin_memory=args.cuda
)
train_loader2 = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=args.cuda
)

all_loader1 = DataLoader(
    train_dataset,
    batch_size=len(train_dataset),
    shuffle=False,
    num_workers=0,
    pin_memory=args.cuda
)

all_loader2 = DataLoader(   
    train_dataset,
    batch_size=len(train_dataset),
    shuffle=False,
    num_workers=0,
    pin_memory=args.cuda
)

import cnn_vae_module_affordance as vae_module

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

def get_concat_h_multi_resize(dir_name, agent, num_classes=10, resample=Image.BICUBIC):
    """複数の画像を横に並べて結合する関数"""
    im_list = []
    for i in range(num_classes):
        try:
            im = Image.open(f'{dir_name}/recon{agent}/manual_{i}.png')
            im_list.append(im)
        except FileNotFoundError:
            print(f"Warning: {dir_name}/recon{agent}/manual_{i}.png not found")
            continue
    
    if not im_list:
        print(f"No images found for agent {agent}")
        return
    
    min_height = min(im.height for im in im_list)
    im_list_resize = [im.resize((int(im.width * min_height / im.height), min_height), resample=resample)
                      for im in im_list]
    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new('RGB', (total_width, min_height))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    dst.save(f'{dir_name}/recon{agent}/concat.png')

def get_concat_image_affordance(dir_name, agent, num_classes=10, resample=Image.BICUBIC):
    """画像とアフォーダンスを並べて結合する関数"""
    print(f"Creating combined image-affordance visualization for Agent {agent}...")
    
    # 画像とアフォーダンスのリストを取得
    image_list = []
    affordance_list = []
    
    for i in range(num_classes):
        try:
            # 画像の読み込み
            image_path = f'{dir_name}/recon{agent}/manual_{i}.png'
            if os.path.exists(image_path):
                im = Image.open(image_path)
                image_list.append(im)
            else:
                print(f"Warning: {image_path} not found")
                continue
                
            # アフォーダンスの読み込み
            affordance_path = f'{dir_name}/recon{agent}/affordance_manual_{i}.png'
            if os.path.exists(affordance_path):
                aff = Image.open(affordance_path)
                affordance_list.append(aff)
            else:
                print(f"Warning: {affordance_path} not found")
                # アフォーダンスが見つからない場合は黒い画像を作成
                aff = Image.new('RGB', im.size, (0, 0, 0))
                affordance_list.append(aff)
                
        except Exception as e:
            print(f"Error processing category {i}: {str(e)}")
            continue
    
    if not image_list or not affordance_list:
        print(f"No valid images or affordances found for agent {agent}")
        return
    
    # 画像とアフォーダンスの数を合わせる
    num_items = min(len(image_list), len(affordance_list))
    image_list = image_list[:num_items]
    affordance_list = affordance_list[:num_items]
    
    # 高さを統一
    min_height = min(im.height for im in image_list + affordance_list)
    
    # 画像とアフォーダンスをリサイズ
    image_list_resize = [im.resize((int(im.width * min_height / im.height), min_height), resample=resample)
                         for im in image_list]
    affordance_list_resize = [aff.resize((int(aff.width * min_height / aff.height), min_height), resample=resample)
                              for aff in affordance_list]
    
    # 画像とアフォーダンスを交互に配置
    combined_list = []
    for i in range(num_items):
        combined_list.append(image_list_resize[i])
        combined_list.append(affordance_list_resize[i])
    
    # 結合画像を作成
    total_width = sum(im.width for im in combined_list)
    dst = Image.new('RGB', (total_width, min_height))
    
    pos_x = 0
    for i, im in enumerate(combined_list):
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    
    # 保存
    output_path = f'{dir_name}/recon{agent}/concat_image_affordance.png'
    dst.save(output_path)
    print(f"Combined image-affordance saved to: {output_path}")

def decode_from_mgmm(load_iteration, sigma, K, decode_k, sample_num, manual, dir_name):
    """GMMからサンプリングして画像とアフォーダンスを生成する関数"""
    print(f"Generating images and affordances from GMM for {K} categories...")
    
    for i in range(K):
        print(f"Processing category {i}...")
        
        # Agent Aの処理
        sample_d = visualize_gmm(iteration=load_iteration,
                                sigma=sigma,
                                K=K, 
                                decode_k=i, 
                                sample_num=sample_num, 
                                manual=manual, 
                                model_dir=dir_name, agent="A")
        decode_affordance(iteration=load_iteration, decode_k=i, sample_num=sample_num, 
                         sample_d=sample_d, manual=manual, model_dir=dir_name, agent="A")
        
        # Agent Bの処理
        sample_d = visualize_gmm(iteration=load_iteration,
                                sigma=sigma,
                                K=K, 
                                decode_k=i, 
                                sample_num=sample_num, 
                                manual=manual, 
                                model_dir=dir_name, agent="B")
        decode_affordance(iteration=load_iteration, decode_k=i, sample_num=sample_num, 
                         sample_d=sample_d, manual=manual, model_dir=dir_name, agent="B")

def decode_affordance(iteration, decode_k, sample_num, sample_d, manual, model_dir, agent):
    """潜在変数から画像とアフォーダンスを生成する関数"""
    print(f"Decoding affordance for Agent: {agent}, category: {decode_k}")
    
    # VAEモデルの読み込み
    model = vae_module.VAE().to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, f"pth/vae{agent}_{iteration}.pth")))
    model.eval()
    
    sample_d = torch.from_numpy(sample_d.astype(np.float32)).clone()
    
    with torch.no_grad():
        sample_d = sample_d.to(device)
        
        # 画像とアフォーダンスの生成
        image_recon, affordance_pred = model.decode(sample_d)
        
        # 画像の保存
        image_filename = f'manual_{decode_k}.png' if manual else f'random_{decode_k}.png'
        save_image(image_recon.view(sample_num, 3, 64, 64), 
                  os.path.join(model_dir, f'recon{agent}/{image_filename}'))
        
        # アフォーダンスマップの保存
        affordance_pred = affordance_pred.argmax(dim=1)  # [B, H, W]
        affordance_filename = f'affordance_manual_{decode_k}.png' if manual else f'affordance_random_{decode_k}.png'

        # サンプル数が1の場合は余白なしで保存
        if sample_num == 1:
            affordance_map = affordance_pred[0].cpu().numpy()
            plt.figure(figsize=(6, 6))
            plt.imshow(affordance_map, cmap='tab10', vmin=0, vmax=3)
            plt.axis('off')
            plt.savefig(os.path.join(model_dir, f'recon{agent}/{affordance_filename}'), bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            # サンプル数が複数の場合は従来通り
            plt.figure(figsize=(8, 6))
            for i in range(min(sample_num, 4)):
                plt.subplot(2, 2, i + 1)
                affordance_map = affordance_pred[i].cpu().numpy()
                plt.imshow(affordance_map, cmap='tab10', vmin=0, vmax=3)
                plt.title(f'Affordance {i+1}')
                plt.axis('off')
            plt.suptitle(f'Affordance Predictions - Agent {agent}, Category {decode_k}')
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, f'recon{agent}/{affordance_filename}'), dpi=300, bbox_inches='tight')
            plt.close()

def evaluate_recall_performance(load_iteration, all_loader, model_dir, agent):
    """想起性能を評価する関数"""
    print(f"Evaluating recall performance for Agent {agent}...")
    
    # VAEモデルの読み込み
    model = vae_module.VAE().to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, f"pth/vae{agent}_{load_iteration}.pth")))
    model.eval()
    
    all_image_losses = []
    all_affordance_losses = []
    all_mean_ious = []
    
    with torch.no_grad():
        for images, affordances, categories in all_loader:
            images = images.to(device)
            affordances = affordances.to(device)
            
            # 順伝播
            image_recon, affordance_pred, mu, logvar, z = model(images)
            
            # 画像の再構成誤差
            image_loss = F.mse_loss(image_recon, images, reduction='mean')
            all_image_losses.append(image_loss.item())
            
            # アフォーダンスの予測誤差
            affordance_loss = F.cross_entropy(
                affordance_pred.view(affordance_pred.size(0), affordance_pred.size(1), -1),
                affordances.view(affordances.size(0), -1),
                reduction='mean'
            )
            all_affordance_losses.append(affordance_loss.item())
            
            # mean-IoUの計算
            pred = affordance_pred.argmax(dim=1)
            mean_iou = calculate_mean_iou(pred, affordances)
            all_mean_ious.append(mean_iou)
    
    # 平均値を計算
    avg_image_loss = np.mean(all_image_losses)
    avg_affordance_loss = np.mean(all_affordance_losses)
    avg_mean_iou = np.mean(all_mean_ious)
    
    print(f"Agent {agent} - Average Image Loss: {avg_image_loss:.4f}")
    print(f"Agent {agent} - Average Affordance Loss: {avg_affordance_loss:.4f}")
    print(f"Agent {agent} - Average Mean-IoU: {avg_mean_iou:.4f}")
    
    # 結果をファイルに保存
    result_file = os.path.join(result_dir, f'recall_performance_{agent}_{load_iteration}.txt')
    with open(result_file, 'w') as f:
        f.write(f"Recall Performance Evaluation - Agent {agent} (Iteration {load_iteration})\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Average Image Loss: {avg_image_loss:.4f}\n")
        f.write(f"Average Affordance Loss: {avg_affordance_loss:.4f}\n")
        f.write(f"Average Mean-IoU: {avg_mean_iou:.4f}\n")
        f.write(f"Number of samples: {len(all_loader.dataset)}\n")
    
    return avg_image_loss, avg_affordance_loss, avg_mean_iou

def main():
    load_iteration = args.load_iteration
    print(f"Loading model from iteration {load_iteration}")
    
    # GMMから画像とアフォーダンスを生成
    decode_from_mgmm(load_iteration=load_iteration, sigma=0, K=args.category, 
                     decode_k=None, sample_num=1, manual=True, dir_name=dir_name)
    
    # 画像の結合
    get_concat_h_multi_resize(dir_name=dir_name, agent="A", num_classes=args.category)
    get_concat_h_multi_resize(dir_name=dir_name, agent="B", num_classes=args.category)
    
    # 画像とアフォーダンスの結合
    get_concat_image_affordance(dir_name=dir_name, agent="A", num_classes=args.category)
    get_concat_image_affordance(dir_name=dir_name, agent="B", num_classes=args.category)
    
    # 想起性能の評価
    print("\n" + "="*50)
    print("EVALUATING RECALL PERFORMANCE")
    print("="*50)
    
    # Agent Aの性能評価
    image_loss_A, affordance_loss_A, mean_iou_A = evaluate_recall_performance(
        load_iteration, all_loader1, dir_name, "A"
    )
    
    # Agent Bの性能評価
    image_loss_B, affordance_loss_B, mean_iou_B = evaluate_recall_performance(
        load_iteration, all_loader2, dir_name, "B"
    )
    
    # 比較結果の保存
    comparison_file = os.path.join(result_dir, f'recall_comparison_{load_iteration}.txt')
    with open(comparison_file, 'w') as f:
        f.write("Recall Performance Comparison\n")
        f.write("=" * 40 + "\n\n")
        f.write("Metric          | Agent A | Agent B | Difference\n")
        f.write("-" * 40 + "\n")
        f.write(f"Image Loss      | {image_loss_A:.4f} | {image_loss_B:.4f} | {abs(image_loss_A - image_loss_B):.4f}\n")
        f.write(f"Affordance Loss | {affordance_loss_A:.4f} | {affordance_loss_B:.4f} | {abs(affordance_loss_A - affordance_loss_B):.4f}\n")
        f.write(f"Mean-IoU        | {mean_iou_A:.4f} | {mean_iou_B:.4f} | {abs(mean_iou_A - mean_iou_B):.4f}\n")
    
    print(f"\nResults saved to: {dir_name}")
    print(f"Performance comparison saved to: {comparison_file}")
    
    # 生成されたファイルの一覧を表示
    print(f"\nGenerated visualization files:")
    for agent in ["A", "B"]:
        print(f"\nAgent {agent}:")
        print(f"  - concat.png: Combined images only")
        print(f"  - concat_image_affordance.png: Images and affordances alternating")

if __name__=="__main__":
    main()
