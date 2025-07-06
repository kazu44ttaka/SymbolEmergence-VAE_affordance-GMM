# アフォーダンス特徴予測を併用した記号創発
VAEにアフォーダンス特徴を予測するヘッドを追加することで潜在変数にアフォーダンス特徴を反映し、それに基づいたメトロポリスヘイスティングス名づけゲームによる記号創発を検証する。

リポジトリのプログラムについて:
- `main.py`: メインプログラム．エージェント間でネーミングゲームを行います．
- `cnn_vae_module_affordance.py`: main.py内でVAEの学習を行わせるプログラム．
- `create_single_object_dataset.py`, `create_simple_dataset.py`: データセットを作成するプログラム.
- `recall_image_affordance.py`: 学習後のエージェントに画像の想起とアフォーダンスの予測を行わせるプログラム．
- `tool.py`: 様々な関数が格納されたプログラム.

本レポジトリは以下のレポジトリを改変して作成されました。

https://github.com/is0383kk/SymbolEmergence-VAE-GMM

# Requirements
- uv

# データセット
データセットは[IIT-AFF Dataset](https://sites.google.com/site/iitaffdataset/)を使用しています。リンクからzipファイルをダウンロードし、レポジトリ上に解凍してください。

その後`create_single_object_dataset.py`を実行すると、データセットに含まれる画像のうち単一オブジェクトのみを含む画像を抽出することができます。

特定のクラスのみに対して実験を行いたい場合は`create_simple_dataset.py`でそのクラスのみを含むデータセットを作成することができます。また、各クラスに属する画像の最大枚数を決めることもできます。その際にはResNet152を用いた特徴量抽出によって類似度が高い上位N枚の画像が抽出されます。

# 実行方法
`main.py`を実行することで確率的推論によるネーミングゲームを行います．

# 出力例
```
 $ python3 main.py 

CUDA True
Results will be saved in: ./model\affordances_bs64_vae50_mh50_cat5_mode-1_aw1.0_simple_dataset
Configuration saved to: ./model\affordances_bs64_vae50_mh50_cat5_mode-1_aw1.0_simple_dataset\config.txt
Dataset: IIT Affordances
Total number of samples: 2384
Train size: 1907, Validation size: 477
Category: 5
VAE_iter: 50, Batch_size: 64
MH_iter: 50, MH_mode: -1(-1:Com 0:No-com 1:All accept)
------------------Mutual learning session 0 begins------------------
====> Epoch: 1
Total Loss: 8007.9282     
Image Loss: 1110.0923     
Affordance Loss: 6862.5572
KL Loss: 35.2787
Mean-IoU: 0.1305
====> Epoch: 25
Total Loss: 1371.5478
Image Loss: 373.2783
Affordance Loss: 912.5022
KL Loss: 85.7673
Mean-IoU: 0.6207
====> Epoch: 50
Total Loss: 934.6113
Image Loss: 299.8477
Affordance Loss: 548.9367
KL Loss: 85.8268
Mean-IoU: 0.6726
Final Mean-IoU: 0.6353
====> Epoch: 1
Total Loss: 9222.4276
Image Loss: 1104.3040
Affordance Loss: 8101.9288
KL Loss: 16.1948
Mean-IoU: 0.1113
====> Epoch: 25
Total Loss: 1296.2485
Image Loss: 374.4188
Affordance Loss: 838.4851
KL Loss: 83.3446
Mean-IoU: 0.6336
====> Epoch: 50
Total Loss: 1034.3539
Image Loss: 341.6952
Affordance Loss: 607.7255
KL Loss: 84.9332
Mean-IoU: 0.6645
Final Mean-IoU: 0.6519
データセットに含まれるカテゴリ: [0 1 2 5 9]
利用可能なカテゴリ: {0: 'bowl', 1: 'tvm', 2: 'pan', 5: 'cup', 9: 'bottle'}
t-SNEによる可視化を実行中...
PCAによる可視化を実行中...
データセットに含まれるカテゴリ: [0 1 2 5 9]
利用可能なカテゴリ: {0: 'bowl', 1: 'tvm', 2: 'pan', 5: 'cup', 9: 'bottle'}
t-SNEによる可視化を実行中...
PCAによる可視化を実行中...
M-H algorithm Start(0): Epoch:50
=> Epoch: 1, ARI_A: 0.021, ARI_B: 0.009, Kappa:0.425, A2B:1200, B2A:972
=> Epoch: 10, ARI_A: 0.26, ARI_B: 0.275, Kappa:0.823, A2B:1520, B2A:1576
=> Epoch: 20, ARI_A: 0.27, ARI_B: 0.277, Kappa:0.874, A2B:1621, B2A:1681
=> Epoch: 30, ARI_A: 0.244, ARI_B: 0.254, Kappa:0.86, A2B:1595, B2A:1681
=> Epoch: 40, ARI_A: 0.258, ARI_B: 0.263, Kappa:0.868, A2B:1623, B2A:1669
Final Epoch 50: Analyzing class difficulty (ARI_A: 0.251, ARI_B: 0.249)...
Agent A - Most difficult classes: [0, 1, 3]
Agent B - Most difficult classes: [0, 1, 3]
=> Epoch: 50, ARI_A: 0.251, ARI_B: 0.249, Kappa:0.867, A2B:1649, B2A:1676
Iteration:0 Done: max_ARI_A: 0.274, max_ARI_B: 0.281, max_Kappa:0.892
```
評価値について:
- `ARI_A`: エージェントAのARI：エージェントAのサイン変数 w^A と真のMNISTラベルとの一致度合いを表す．
- `ARI_B`: エージェントBのARI：エージェントBのサイン変数 w^B と真のMNISTラベルとの一致度合いを表す．
- `Kappa`: カッパ係数：エージェント間のサイン変数 w^A と w^B の一致度合いを表す．
- `A2B`: 発話者A・聞き手Bのとき，Aが提案したサインをBが受容した回数.
- `B2A`: 発話者B・聞き手Aのとき，Bが提案したサインをAが受容した回数.

# エージェントによる画像の想起とアフォーダンス予測
ネーミングゲームの終了後にエージェントは画像の想起とそのアフォーダンスの予測を行うことができます.  
エージェントによる画像の想起では，エージェント内のVAEが推論した潜在変数に対してGMMが推定した平均パラメータを用います.  
この平均パラメータをVAEデコーダに入力することで画像を再構成しエージェントに画像を想起させ、さらにそのアフォーダンスを予測させることができます。
  
`main.py`によるネーミングゲーム終了後，`recall_image_affordance.py`を実行してください．

Recall image and affordance：
<div>
<img src='image/concat_image_affordance.png' width="400px">
</div>
