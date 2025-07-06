import os
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import argparse

# データセットのパス設定
DATASET_ROOT = 'IIT_Affordances_2017'
OUTPUT_ROOT = 'single_object_dataset'

# オブジェクトクラスの定義
OBJECT_CLASSES = {
    0: 'bowl',
    1: 'tvm', 
    2: 'pan',
    3: 'hammer',
    4: 'knife',
    5: 'cup',
    6: 'drill',
    7: 'racket',
    8: 'spatula',
    9: 'bottle'
}

# アフォーダンスクラスの定義
AFFORDANCE_CLASSES = {
    0: 'background',
    1: 'contain',
    2: 'cut',
    3: 'display',
    4: 'engine',
    5: 'grasp',
    6: 'hit',
    7: 'pound',
    8: 'support',
    9: 'w-grasp'
}

def read_object_labels(label_path):
    """オブジェクトラベルファイルを読み込む"""
    objects = []
    with open(label_path, 'r') as f:
        for line in f:
            obj_id, xmin, ymin, xmax, ymax = map(int, line.strip().split())
            objects.append({
                'obj_id': obj_id,
                'bbox': (xmin, ymin, xmax, ymax)
            })
    return objects

def read_affordance_labels(label_path):
    """アフォーダンスラベルファイルを読み込む"""
    with open(label_path, 'r') as f:
        # 各行を読み込み、空白で分割して整数に変換
        labels = [list(map(int, line.strip().split())) for line in f]
    return np.array(labels)

def process_image(image_name, target_classes=None):
    """単一オブジェクトの画像を処理"""
    # パスの設定
    rgb_path = os.path.join(DATASET_ROOT, 'rgb', image_name)
    affordance_path = os.path.join(DATASET_ROOT, 'affordances_labels', image_name.replace('.jpg', '.txt'))
    label_path = os.path.join(DATASET_ROOT, 'object_labels', image_name.replace('.jpg', '.txt'))
    
    # オブジェクトラベルの読み込み
    objects = read_object_labels(label_path)
    
    # 単一オブジェクトの場合のみ処理
    if len(objects) != 1:
        return False
    
    obj = objects[0]
    obj_id = obj['obj_id']
    
    # クラスフィルタリング
    if target_classes is not None and obj_id not in target_classes:
        return False
    
    xmin, ymin, xmax, ymax = obj['bbox']
    
    # 画像の読み込み（ここで先に定義）
    rgb_img = Image.open(rgb_path)
    affordance_labels = read_affordance_labels(affordance_path)
    
    # 正方形クロップ領域の計算
    box_w = xmax - xmin
    box_h = ymax - ymin
    
    # bbox中心
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    
    # パディングの計算
    padding_w = max(10, int(box_w * 0.1))
    padding_h = max(10, int(box_h * 0.1))
    side_w = box_w + 2 * padding_w
    side_h = box_h + 2 * padding_h
    side_ = max(side_w, side_h)
    
    # クロップ領域の計算
    sq_xmin = max(0, cx - side_ // 2)
    sq_ymin = max(0, cy - side_ // 2)
    sq_xmax = min(rgb_img.width, cx + side_ // 2)
    sq_ymax = min(rgb_img.height, cy + side_ // 2)
    
    crop_w = sq_xmax - sq_xmin
    crop_h = sq_ymax - sq_ymin
    
    side = max(crop_w, crop_h)
    
    # クロップ画像の取得
    cropped_img = rgb_img.crop((sq_xmin, sq_ymin, sq_xmax, sq_ymax))
    cropped_arr = np.array(cropped_img)

    # 必要な余白サイズを計算
    pad_h = (side - crop_h) // 2
    pad_w = (side - crop_w) // 2

    # エッジ拡張でパディング
    padded_arr = np.pad(
        cropped_arr,
        ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='edge'
    )
    rgb_square = Image.fromarray(padded_arr)

    # アフォーダンスマップも同様に
    aff_crop = affordance_labels[sq_ymin:sq_ymax, sq_xmin:sq_xmax]
    aff_padded = np.pad(
        aff_crop,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode='edge'
    )

    # 保存
    output_rgb_path = os.path.join(OUTPUT_ROOT, 'rgb', image_name)
    output_affordance_path = os.path.join(OUTPUT_ROOT, 'affordances', image_name.replace('.jpg', '.txt'))

    rgb_square.save(output_rgb_path)
    np.savetxt(output_affordance_path, aff_padded, fmt='%d')
    
    # メタデータの保存
    metadata = {
        'original_image': image_name,
        'object_id': obj_id,
        'object_class': OBJECT_CLASSES[obj_id],
        'bbox': [xmin, ymin, xmax, ymax],
        'affordance_classes': AFFORDANCE_CLASSES
    }
    
    with open(os.path.join(OUTPUT_ROOT, 'metadata', image_name.replace('.jpg', '.txt')), 'w') as f:
        for key, value in metadata.items():
            if key == 'affordance_classes':
                f.write(f"{key}:\n")
                for class_id, class_name in value.items():
                    f.write(f"  {class_id}: {class_name}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Create single object dataset with class filtering')
    parser.add_argument('--classes', nargs='+', type=str, 
                       default=['cup', 'bottle', 'bowl', 'tvm', 'pan', 'hammer', 'knife', 'drill', 'racket', 'spatula'],
                       help='Target object classes to include (e.g., cup bottle bowl)')
    parser.add_argument('--output-dir', type=str, default="single_object_dataset",
                       help='Output directory for the dataset')
    args = parser.parse_args()
    
    # 出力ディレクトリの設定
    global OUTPUT_ROOT
    OUTPUT_ROOT = args.output_dir
    
    # 出力ディレクトリの作成
    os.makedirs(os.path.join(OUTPUT_ROOT, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, 'affordances'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, 'metadata'), exist_ok=True)
    
    # ターゲットクラスの設定
    target_classes = None
    if args.classes:
        # クラス名からIDに変換
        class_name_to_id = {name: id for id, name in OBJECT_CLASSES.items()}
        target_classes = [class_name_to_id[name] for name in args.classes]
        print(f"対象クラス: {args.classes}")
        print(f"対象クラスID: {target_classes}")
    else:
        print("すべてのクラスを含めます")
    
    # object_labelsディレクトリから画像リストを取得
    image_list = []
    for label_file in os.listdir(os.path.join(DATASET_ROOT, 'object_labels')):
        if label_file.endswith('.txt'):
            # .txtを.jpgに変換して画像名を取得
            image_name = label_file.replace('.txt', '.jpg')
            image_list.append(image_name)
    
    # 処理の実行
    processed_count = 0
    class_counts = {class_id: 0 for class_id in range(len(OBJECT_CLASSES))}
    
    for image_name in tqdm(image_list, desc="Processing images"):
        if process_image(image_name, target_classes):
            processed_count += 1
            # クラス別カウント
            label_path = os.path.join(DATASET_ROOT, 'object_labels', image_name.replace('.jpg', '.txt'))
            objects = read_object_labels(label_path)
            if objects:
                obj_id = objects[0]['obj_id']
                class_counts[obj_id] += 1
    
    print(f"\n処理完了: {processed_count}個の単一オブジェクト画像を抽出しました")
    
    # クラス別統計の表示
    print("\nクラス別統計:")
    for class_id, count in class_counts.items():
        if count > 0:
            print(f"  {OBJECT_CLASSES[class_id]}: {count}個")

if __name__ == '__main__':
    main() 