import os
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from collections import defaultdict

# データセットのパス設定
INPUT_ROOT = 'single_object_dataset'
OUTPUT_ROOT = 'simple_dataset'

# 使用するクラスを指定（空のリストの場合はすべてのクラスを使用）
TARGET_CLASSES = [
    'bowl', 'cup', 'pan', 'hammer', 'knife', 
    'drill', 'racket', 'spatula', 'bottle', 'tvm'
]
# 特定のクラスのみを使用したい場合は以下のように指定
TARGET_CLASSES = ['tvm', 'pan', 'cup', 'bottle', 'bowl']

# 出力ディレクトリの作成
os.makedirs(os.path.join(OUTPUT_ROOT, 'rgb'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, 'affordances'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, 'metadata'), exist_ok=True)

# 画像変換の設定
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_feature_extractor():
    """CNNモデルを読み込み、特徴抽出器として設定"""
    model = models.resnet152(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # 最後の全結合層を除去
    model.eval()
    return model

def extract_features(model, image_path):
    """画像から特徴量を抽出"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        features = model(img_tensor)
    
    return features.squeeze().numpy()

def read_metadata(metadata_path):
    """メタデータファイルを読み込む"""
    metadata = {}
    with open(metadata_path, 'r') as f:
        for line in f:
            if line.strip():
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == 'object_class':
                        metadata[key] = value
                    elif key == 'bbox':
                        metadata[key] = eval(value)
    return metadata

def process_class_images(class_name, image_paths, model, max_images=500):
    """クラスごとの画像を処理し、類似度の高い画像を抽出"""
    # 特徴量の抽出
    features = []
    valid_paths = []
    
    for img_path in tqdm(image_paths, desc=f"Processing {class_name}"):
        try:
            feature = extract_features(model, img_path)
            features.append(feature)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    if not features:
        return []
    
    features = np.array(features)
    
    # 画像数がmax_images以下の場合はすべて使用
    if len(features) <= max_images:
        print(f"{class_name}: {len(features)}枚（すべて使用）")
        return valid_paths
    
    # 画像数がmax_imagesより多い場合はK-meansクラスタリングで選択
    print(f"{class_name}: {len(features)}枚から{max_images}枚を選択")
    
    # K-meansクラスタリング
    kmeans = KMeans(n_clusters=max_images, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    # 各クラスターの中心に最も近い画像を選択
    selected_indices = []
    for cluster_id in range(kmeans.n_clusters):
        cluster_features = features[clusters == cluster_id]
        cluster_paths = [valid_paths[i] for i in range(len(valid_paths)) if clusters[i] == cluster_id]
        
        # クラスター中心との距離を計算
        distances = np.linalg.norm(cluster_features - kmeans.cluster_centers_[cluster_id], axis=1)
        closest_idx = np.argmin(distances)
        selected_indices.append(np.where(clusters == cluster_id)[0][closest_idx])
    
    return [valid_paths[i] for i in selected_indices]

def copy_selected_images(selected_paths):
    """選択された画像と関連ファイルを新しいデータセットにコピー"""
    for img_path in tqdm(selected_paths, desc="Copying selected images"):
        # ファイル名の取得
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        
        # 関連ファイルのパス
        rgb_path = img_path
        affordance_path = os.path.join(INPUT_ROOT, 'affordances', f"{base_name}.txt")
        metadata_path = os.path.join(INPUT_ROOT, 'metadata', f"{base_name}.txt")
        
        # 出力パス
        output_rgb_path = os.path.join(OUTPUT_ROOT, 'rgb', img_name)
        output_affordance_path = os.path.join(OUTPUT_ROOT, 'affordances', f"{base_name}.txt")
        output_metadata_path = os.path.join(OUTPUT_ROOT, 'metadata', f"{base_name}.txt")
        
        # ファイルのコピー
        shutil.copy2(rgb_path, output_rgb_path)
        shutil.copy2(affordance_path, output_affordance_path)
        shutil.copy2(metadata_path, output_metadata_path)

def main():
    # 特徴抽出器の読み込み
    print("特徴抽出器を読み込んでいます...")
    model = load_feature_extractor()
    
    # クラスごとに画像をグループ化
    class_images = defaultdict(list)
    metadata_dir = os.path.join(INPUT_ROOT, 'metadata')
    
    print("メタデータを読み込んでいます...")
    for metadata_file in os.listdir(metadata_dir):
        if metadata_file.endswith('.txt'):
            metadata_path = os.path.join(metadata_dir, metadata_file)
            metadata = read_metadata(metadata_path)
            
            if 'object_class' in metadata:
                class_name = metadata['object_class']
                # TARGET_CLASSESが指定されている場合は、そのクラスのみを処理
                if TARGET_CLASSES and class_name not in TARGET_CLASSES:
                    continue
                    
                img_name = metadata_file.replace('.txt', '.jpg')
                img_path = os.path.join(INPUT_ROOT, 'rgb', img_name)
                if os.path.exists(img_path):
                    class_images[class_name].append(img_path)
    
    # 処理対象クラスの確認
    if TARGET_CLASSES:
        print(f"\n処理対象クラス: {TARGET_CLASSES}")
        missing_classes = set(TARGET_CLASSES) - set(class_images.keys())
        if missing_classes:
            print(f"警告: 以下のクラスが見つかりませんでした: {missing_classes}")
    else:
        print(f"\nすべてのクラスを処理します")
    
    # クラスごとの画像数を表示
    print("\nクラスごとの画像数:")
    for class_name, image_paths in class_images.items():
        print(f"  {class_name}: {len(image_paths)}枚")
    
    # 各クラスごとに類似度の高い画像を抽出
    selected_images = []
    class_stats = {}
    
    for class_name, image_paths in class_images.items():
        print(f"\n{class_name}クラスの処理を開始します...")
        selected = process_class_images(class_name, image_paths, model, max_images=500)
        selected_images.extend(selected)
        class_stats[class_name] = {
            'original': len(image_paths),
            'selected': len(selected)
        }
        print(f"{class_name}: {len(selected)}枚の画像を選択しました")
    
    # 選択された画像を新しいデータセットにコピー
    print("\n選択された画像を新しいデータセットにコピーしています...")
    copy_selected_images(selected_images)
    
    # 最終統計を表示
    print(f"\n=== 最終統計 ===")
    print(f"合計選択画像数: {len(selected_images)}枚")
    print("\nクラスごとの詳細:")
    for class_name, stats in class_stats.items():
        reduction_rate = (1 - stats['selected'] / stats['original']) * 100
        print(f"  {class_name}: {stats['original']}枚 → {stats['selected']}枚 (削減率: {reduction_rate:.1f}%)")
    
    print(f"\n処理完了: 合計{len(selected_images)}枚の画像を抽出しました")

if __name__ == '__main__':
    main() 