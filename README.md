# YCB SynthForge

BlenderProcによる合成データ生成とYOLO26によるYCB物体検出パイプライン

## 概要

YCB SynthForgeは、103種類のYCBオブジェクトを検出するためのEnd-to-Endパイプラインです。

- **合成データ生成**: BlenderProcによるフォトリアリスティックなレンダリング
- **ドメインランダム化**: Sim-to-Real転移のための多様なデータ生成
- **YOLO26学習**: COCO事前学習モデルのファインチューニング

## プロジェクト構成

```
ycb_synthforge/
├── docker/
│   ├── Dockerfile.blenderproc    # BlenderProc環境
│   └── Dockerfile.yolo26         # YOLO26学習環境
├── docker-compose.yml
├── models/
│   └── ycb/                      # YCB 3Dモデル (103クラス)
├── resources/
│   └── cctextures/               # CC0テクスチャ (2022枚)
├── weights/
│   ├── yolo26n.pt                # YOLO26 Nano (2.6M params)
│   └── yolo26s.pt                # YOLO26 Small
├── scripts/
│   ├── download_weights.py       # YOLO26重みダウンロード
│   ├── download_ycb_models.py    # YCB 3Dモデルダウンロード
│   ├── download_cctextures.py    # CC0テクスチャダウンロード
│   ├── blenderproc/              # データ生成スクリプト
│   │   ├── generate_dataset.py   # メイン生成スクリプト
│   │   ├── config.yaml           # 生成設定
│   │   ├── scene_setup.py        # シーン構築
│   │   ├── lighting.py           # 照明ランダム化
│   │   ├── camera.py             # カメラ効果
│   │   ├── materials.py          # マテリアルランダム化
│   │   └── ycb_classes.py        # 103クラス定義
│   ├── data_processing/
│   │   ├── coco_to_yolo.py       # COCO→YOLO変換 (train/val/test分割)
│   │   └── merge_datasets.py     # データセット結合
│   └── training/
│       ├── train_yolo26.py       # 学習スクリプト
│       ├── train_config.yaml     # 学習設定
│       ├── evaluate.py           # 評価スクリプト
│       └── inference.py          # 推論スクリプト
├── data/
│   └── synthetic/
│       ├── coco/                 # 生成データ (COCO形式)
│       │   ├── images/           # レンダリング画像
│       │   └── annotations.json  # アノテーション
│       └── yolo/                 # 変換データ (YOLO形式)
│           ├── images/{train,val,test}/
│           ├── labels/{train,val,test}/
│           └── dataset.yaml
├── runs/                         # 学習結果
│   └── */weights/{best,last}.pt
└── logs/                         # ログファイル
```

## セットアップ

### 必要環境

- Docker & Docker Compose v2+ (`docker compose` コマンドを使用)
  - 注意: 旧版の `docker-compose` (v1.x) は非対応
- NVIDIA GPU (CUDA対応)
- NVIDIA Container Toolkit
- 推奨: RTX 3090/4090 (VRAM 24GB)

### Dockerイメージのビルド

```bash
# 全イメージをビルド
docker compose build

# 個別ビルド
docker compose build blenderproc
docker compose build yolo26_train
```

### YOLO26重みのダウンロード

```bash
# 利用可能なモデル一覧を表示
python scripts/download_weights.py --list

# デフォルト (nano + small) をダウンロード
python scripts/download_weights.py

# 特定のモデルをダウンロード
python scripts/download_weights.py --models yolo26n yolo26s yolo26m

# 全モデルをダウンロード
python scripts/download_weights.py --all

# 強制的に再ダウンロード
python scripts/download_weights.py --models yolo26m --force
```

| モデル | パラメータ数 | サイズ | 用途 |
|--------|------------|--------|------|
| yolo26n | 2.6M | ~5 MB | 最速・エッジデバイス向け |
| yolo26s | 9.4M | ~19 MB | バランス型 |
| yolo26m | 20.1M | ~40 MB | 推奨・汎用 |
| yolo26l | 25.3M | ~49 MB | 高精度 |
| yolo26x | 56.9M | ~109 MB | 最高精度 |

### YCB 3Dモデルのダウンロード

```bash
# オブジェクト一覧を表示
python scripts/download_ycb_models.py --list

# カテゴリ一覧を表示
python scripts/download_ycb_models.py --list-categories

# 全103オブジェクトをダウンロード (~3GB)
python scripts/download_ycb_models.py --all

# カテゴリ指定でダウンロード
python scripts/download_ycb_models.py --category food fruit kitchen

# 特定オブジェクトのみダウンロード
python scripts/download_ycb_models.py --objects 001_chips_can 002_master_chef_can

# 強制的に再ダウンロード
python scripts/download_ycb_models.py --all --force
```

| カテゴリ | オブジェクト数 | 内容 |
|---------|--------------|------|
| food | 10 | 缶詰、箱入り食品 |
| fruit | 8 | バナナ、りんご、レモン等 |
| kitchen | 14 | ボウル、マグ、フォーク等 |
| tool | 17 | ドリル、ハンマー、ドライバー等 |
| sport | 6 | ボール類 |
| toy | 34 | レゴ、飛行機、サイコロ等 |
| misc | 14 | カップ、タイマー等 |

### CC0テクスチャのダウンロード

[ambientCG](https://ambientcg.com/)からPBRテクスチャをダウンロード（CC0ライセンス）。

```bash
# カテゴリ一覧を表示
python scripts/download_cctextures.py --list-categories

# デフォルト100テクスチャをダウンロード
python scripts/download_cctextures.py

# カテゴリ指定でダウンロード
python scripts/download_cctextures.py --category floor wall table

# プレフィックス指定 (Wood*, Metal* 各20枚)
python scripts/download_cctextures.py --prefix Wood Metal --limit 20

# 特定テクスチャをダウンロード
python scripts/download_cctextures.py --textures Wood001 Metal002 Tiles005

# 高解像度でダウンロード (1K/2K/4K/8K)
python scripts/download_cctextures.py --resolution 4K

# オンラインで検索
python scripts/download_cctextures.py --search Marble --limit 30
```

| カテゴリ | 用途 | プレフィックス |
|---------|------|---------------|
| floor | 床 | Wood, WoodFloor, Tiles, Marble, Concrete |
| wall | 壁 | Bricks, PaintedPlaster, Wallpaper, Facade |
| table | テーブル | Wood, Metal, Plastic, Marble, Granite |
| metal | 金属 | Metal, MetalPlates, DiamondPlate, Rust |
| fabric | 布 | Fabric, Leather, Carpet, Wicker |
| natural | 自然 | Ground, Grass, Rock, Gravel, Sand |
| industrial | 工業 | Asphalt, Concrete, CorrugatedSteel |

## パイプライン実行

### 1. 合成データ生成
scripts/blenderproc/config.yaml
```bash
# バックグラウンドで生成（config.yamlのnum_images設定に従う）
docker compose run -d blenderproc

# コンテナIDを確認
docker ps | grep blenderproc

# 進捗確認（生成された画像数）
ls data/synthetic/coco/images/ | wc -l

# リアルタイムログ確認
docker logs -f <container_id>

# または最新ログのみ
docker logs --tail 30 <container_id>
```

生成枚数は `scripts/blenderproc/config.yaml` の `scene.num_images` で設定（デフォルト: 12,000枚）。

### 2. COCO→YOLO形式変換

```bash
docker compose run --rm yolo26_train python \
  scripts/data_processing/coco_to_yolo.py \
  --coco_json data/synthetic/coco/annotations.json \
  --output_dir data/synthetic/yolo \
  --train_ratio 0.833 \
  --val_ratio 0.083 \
  --test_ratio 0.083
```

### 3. YOLO26学習

```bash
docker compose run --rm yolo26_train python \
  scripts/training/train_yolo26.py \
  --config scripts/training/train_config.yaml
```

### 4. 評価

```bash
docker compose run --rm yolo26_train python \
  scripts/training/evaluate.py \
  --weights runs/ycb_yolo26/weights/best.pt \
  --data data/synthetic/yolo/dataset.yaml
```

### 5. 推論

```bash
docker compose run --rm yolo26_inference python \
  scripts/training/inference.py \
  --weights runs/ycb_yolo26/weights/best.pt \
  --source /path/to/images
```

## ドメインランダム化

Sim-to-Realギャップを軽減するため、以下の要素をランダム化:

| カテゴリ | ランダム化項目 | 範囲 |
|---------|---------------|------|
| **背景** | 床テクスチャ | Wood, Concrete, Tiles, Marble, Metal, Fabric |
| | 壁テクスチャ | Concrete, Plaster, Brick, Paint, Wallpaper |
| | テーブル材質 | Wood, Metal, Plastic |
| **照明** | 光源数 | 1-4個 |
| | 色温度 | 2700K-6500K |
| | 強度 | 100-1000W相当 |
| | 影の柔らかさ | 0.3-0.9 |
| **カメラ** | 距離 | 0.4-2.0m |
| | 仰角 | 10-70° |
| | 方位角 | 0-360° |
| | 露出 | EV -1.5〜+1.5 |
| | ISO | 100-3200 |
| | 被写界深度 | f/1.8-11.0 |
| **マテリアル** | 金属度 | 0.8-1.0 (金属物体) |
| | 粗さ | 0.05-0.6 |
| | 色相シフト | ±10° |
| **オブジェクト** | 位置 | X,Y: ±0.3m |
| | 回転 | 0-360° (各軸) |
| | スケール | ±5% |

## データセット構成

| Split | 枚数 | 割合 | 用途 |
|-------|------|------|------|
| Train | 10,000 | 83.3% | モデル学習 |
| Val | 1,000 | 8.3% | ハイパーパラメータ調整 |
| Test | 1,000 | 8.3% | 最終評価 |

## YCBオブジェクトクラス (103種)

<details>
<summary>クラス一覧を表示</summary>

### 食品・飲料 (ID: 0-9)
| ID | 名前 | ID | 名前 |
|----|------|----|------|
| 0 | 001_chips_can | 5 | 006_mustard_bottle |
| 1 | 002_master_chef_can | 6 | 007_tuna_fish_can |
| 2 | 003_cracker_box | 7 | 008_pudding_box |
| 3 | 004_sugar_box | 8 | 009_gelatin_box |
| 4 | 005_tomato_soup_can | 9 | 010_potted_meat_can |

### 果物 (ID: 10-17)
| ID | 名前 | ID | 名前 |
|----|------|----|------|
| 10 | 011_banana | 14 | 015_peach |
| 11 | 012_strawberry | 15 | 016_pear |
| 12 | 013_apple | 16 | 017_orange |
| 13 | 014_lemon | 17 | 018_plum |

### キッチン用品 (ID: 18-31)
019_pitcher_base, 021_bleach_cleanser, 022_windex_bottle, 023_wine_glass, 024_bowl, 025_mug, 026_sponge, 027-skillet, 028_skillet_lid, 029_plate, 030_fork, 031_spoon, 032_knife, 033_spatula

### 工具 (ID: 32-48)
035_power_drill, 036_wood_block, 037_scissors, 038_padlock, 039_key, 040_large_marker, 041_small_marker, 042_adjustable_wrench, 043_phillips_screwdriver, 044_flat_screwdriver, 046_plastic_bolt, 047_plastic_nut, 048_hammer, 049-052_clamps

### スポーツ・おもちゃ (ID: 49-102)
ボール類、チェーン、フォームブロック、サイコロ、ビー玉、カップ、木製ブロック、おもちゃの飛行機、レゴデュプロ、タイマー、ルービックキューブ

</details>

## 設定ファイル

### データ生成設定 (`scripts/blenderproc/config.yaml`)

```yaml
scene:
  num_images: 12000
  objects_per_scene: [2, 8]    # シーンあたりのYCBオブジェクト数

rendering:
  engine: "CYCLES"
  samples: 32                   # 32=高速(推奨), 128=高品質
  use_denoising: true
  use_gpu: true

camera:
  distance: [0.4, 2.0]
  elevation: [10, 70]
  azimuth: [0, 360]
```

### 学習設定 (`scripts/training/train_config.yaml`)

```yaml
model:
  architecture: yolo26n         # nano / small / medium
  weights: /workspace/weights/yolo26n.pt
  num_classes: 103

training:
  epochs: 100
  batch_size: 16
  imgsz: 640
  optimizer: auto
  lr0: 0.01
  patience: 20

augmentation:
  mosaic: 1.0
  mixup: 0.1
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
```

## 出力ファイル

### 学習結果 (`runs/`)

```
runs/ycb_yolo26/
├── weights/
│   ├── best.pt              # ベストモデル (mAP基準)
│   └── last.pt              # 最終エポックモデル
├── results.csv              # エポックごとのメトリクス
├── results.png              # 学習曲線グラフ
├── confusion_matrix.png     # 混同行列
├── labels.jpg               # ラベル分布
├── train_batch*.jpg         # 訓練バッチサンプル
└── val_batch*_pred.jpg      # 検証予測結果
```

## 生成時間の目安

| 設定 | samples | 速度 | 12,000枚の所要時間 |
|-----|---------|------|-------------------|
| 高速 | 32 | ~45枚/分 | ~4.5時間 |
| 高品質 | 128 | ~10枚/分 | ~20時間 |

## トラブルシューティング

### GPUが認識されない

```bash
# ホストでNVIDIA確認
nvidia-smi

# コンテナ内で確認
docker compose run --rm yolo26_train nvidia-smi
```

### メモリ不足エラー

`docker-compose.yml`で`shm_size`を増加:

```yaml
services:
  blenderproc:
    shm_size: '16gb'
  yolo26_train:
    shm_size: '16gb'
```

### NumPy互換性警告

`Dockerfile.yolo26`で`numpy<2`を指定済み。警告が出る場合は再ビルド:

```bash
docker compose build yolo26_train --no-cache
```

### OBJファイルの警告

`Invalid normal index`警告は無害。YCBモデルのメッシュ問題で、レンダリングに影響なし。

### YCBモデルが見つからない

モデルが以下のいずれかの構造であることを確認:

```
models/ycb/{object_name}/poisson/textured.obj     # 標準YCB構造（推奨）
models/ycb/{object_name}/{object_name}/poisson/textured.obj
models/ycb/{object_name}/textured.obj
```

### docker-composeエラー

旧版 `docker-compose` (v1.x) ではCompose file形式が非対応:

```bash
# エラー例
The Compose file is invalid because: Unsupported config option for services

# 解決: docker compose (v2+) を使用
docker compose run -d blenderproc  # ○ 正しい
docker-compose run -d blenderproc  # × 旧版は非対応
```

## ライセンス

- YCBモデル: [YCB Object and Model Set License](https://www.ycbbenchmarks.com/)
- CC0テクスチャ: [CC0 1.0 Universal](https://ambientcg.com/)
- BlenderProc: MIT License
- Ultralytics YOLO: AGPL-3.0

## 参考文献

- [BlenderProc](https://github.com/DLR-RM/BlenderProc)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [YCB Object Dataset](https://www.ycbbenchmarks.com/)
- [ambientCG Textures](https://ambientcg.com/)
