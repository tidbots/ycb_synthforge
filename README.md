# YCB SynthForge

BlenderProcによる合成データ生成とYOLO26によるYCB物体検出パイプライン

## 概要

YCB SynthForgeは、85種類のYCBオブジェクトを検出するためのEnd-to-Endパイプラインです。

- **合成データ生成**: BlenderProcによるフォトリアリスティックなレンダリング
- **ドメインランダム化**: Sim-to-Real転移のための多様なデータ生成
- **YOLO26学習**: COCO事前学習モデルのファインチューニング
- **google_16k + tsdf形式**: オブジェクトごとに最適な形式を自動選択

## プロジェクト構成

```
ycb_synthforge/
├── docker/
│   ├── Dockerfile.blenderproc    # BlenderProc環境
│   └── Dockerfile.yolo26         # YOLO26学習環境
├── docker-compose.yml
├── models/
│   ├── ycb/                      # YCB 3Dモデル (85クラス, google_16k/tsdf形式)
│   └── tidbots/                  # カスタム3Dモデル (独自オブジェクト用)
├── resources/
│   └── cctextures/               # CC0テクスチャ (2022枚)
├── weights/
│   ├── yolo26n.pt                # YOLO26 Nano (2.6M params)
│   └── yolo26s.pt                # YOLO26 Small
├── scripts/
│   ├── download_weights.py       # YOLO26重みダウンロード
│   ├── download_ycb_models.py    # YCB 3Dモデルダウンロード
│   ├── download_cctextures.py    # CC0テクスチャダウンロード
│   ├── fix_tsdf_materials.py     # tsdf形式のマテリアル修正
│   ├── validate_meshes.py        # メッシュ品質検証
│   ├── generate_thumbnails.py    # サムネイル生成 (google_16k/tsdf比較)
│   ├── generate_thumbnails_all_formats.py  # 全形式サムネイル比較
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
│   │   ├── merge_datasets.py     # データセット結合
│   │   ├── create_subset.py      # サブセット作成（追加学習用）
│   │   └── merge_for_incremental.py  # データセット統合（追加学習用）
│   ├── training/
│   │   ├── train_yolo26.py       # 学習スクリプト
│   │   ├── train_incremental.py  # 追加学習スクリプト
│   │   ├── train_config.yaml     # 学習設定
│   │   └── evaluate.py           # 評価スクリプト
│   └── inference/
│       ├── inference.py          # 推論スクリプト
│       ├── ensemble_inference.py # アンサンブル推論
│       └── ensemble_example.py   # アンサンブル使用例
├── data/
│   └── synthetic/
│       ├── coco/                 # 生成データ (COCO形式)
│       │   ├── images/           # レンダリング画像
│       │   └── annotations.json  # アノテーション
│       └── yolo/                 # 変換データ (YOLO形式)
│           ├── images/{train,val,test}/
│           ├── labels/{train,val,test}/
│           └── dataset.yaml
├── outputs/                      # 出力ディレクトリ
│   ├── trained_models/           # 学習済みモデル
│   │   └── ycb_yolo26_run/
│   │       ├── weights/          # best.pt, last.pt
│   │       └── results.csv       # 学習メトリクス
│   └── inference_results/        # 推論結果
│       └── predictions/          # 認識結果画像
├── yolo_dataset/                 # YOLO形式データセット
│   ├── images/{train,val}/
│   ├── labels/{train,val}/
│   └── dataset.yaml
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

**重要**: 本プロジェクトでは`google_16k`形式を基本とし、一部オブジェクトで`tsdf`形式を使用します。

```bash
# オブジェクト一覧を表示
python scripts/download_ycb_models.py --list

# カテゴリ一覧を表示
python scripts/download_ycb_models.py --list-categories

# 全オブジェクトをgoogle_16k形式でダウンロード
python scripts/download_ycb_models.py --all --format google_16k

# tsdf形式もダウンロード（一部オブジェクトで必要）
python scripts/download_ycb_models.py --all --format berkeley

# カテゴリ指定でダウンロード
python scripts/download_ycb_models.py --category food fruit kitchen --format google_16k

# 特定オブジェクトのみダウンロード
python scripts/download_ycb_models.py --objects 003_cracker_box 005_tomato_soup_can --format google_16k

# 強制的に再ダウンロード
python scripts/download_ycb_models.py --all --format google_16k --force
```

| 形式 | 説明 | 用途 |
|------|------|------|
| google_16k | 16kポリゴン、高品質テクスチャ | ✅ 基本形式（72オブジェクト） |
| tsdf | TSDF再構成メッシュ | ✅ 一部オブジェクトで使用（13オブジェクト） |
| google_64k | 64kポリゴン、より高解像度 | |
| google_512k | 512kポリゴン、最高解像度 | |
| poisson | poisson再構成（非推奨） | ❌ テクスチャ破損あり |

### tsdf形式のマテリアル修正

tsdf形式のOBJファイルはマテリアル参照が欠落しているため、初回ダウンロード後に修正が必要です:

```bash
# tsdf形式のOBJファイルを修正（usemtl行を追加）
docker compose run --rm fix_tsdf_materials
```

**注意**: 修正前のファイルは `.obj.backup` として自動保存されます。

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

生成枚数は `scripts/blenderproc/config.yaml` の `scene.num_images` で設定（デフォルト: 30,000枚）。


![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/scene_000009.png)
![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/scene_000013.png)
![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/scene_000029.png)
![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/scene_000455.png)

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
# YOLO26m (Medium) モデルで学習
docker compose run --rm \
  -v $(pwd)/yolo_dataset:/workspace/yolo_dataset:ro \
  yolo26_train python3 /workspace/scripts/training/train_yolo26.py \
  --data /workspace/yolo_dataset/dataset.yaml \
  --weights /workspace/weights/yolo26m.pt \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --project /workspace/outputs/trained_models \
  --name ycb_yolo26_run \
  --device 0 \
  --workers 8
```

#### 学習結果の例 (YOLO26m, 50エポック)

| メトリクス | 値 |
|-----------|-----|
| **mAP50** | 97.52% |
| **mAP50-95** | 95.30% |
| **Precision** | 97.32% |
| **Recall** | 94.43% |
| 学習時間 | 約59分 (RTX 4090) |

学習済み重みは `outputs/trained_models/ycb_yolo26_run/weights/` に保存されます:
- `best.pt` - 最高精度のモデル（推論用に推奨）
- `last.pt` - 最終エポックのモデル

### 4. 評価

```bash
docker compose run --rm yolo26_train python \
  scripts/training/evaluate.py \
  --weights outputs/trained_models/ycb_yolo26_run/weights/best.pt \
  --data yolo_dataset/dataset.yaml
```

### 5. 推論

```bash
# 検証画像に対して推論を実行
docker compose run --rm \
  -v $(pwd)/yolo_dataset:/workspace/yolo_dataset:ro \
  yolo26_inference python3 /workspace/scripts/evaluation/inference.py \
  --model /workspace/outputs/trained_models/ycb_yolo26_run/weights/best.pt \
  --source /workspace/yolo_dataset/images/val \
  --output /workspace/outputs/inference_results \
  --conf 0.5 \
  --device 0
```

推論結果は `outputs/inference_results/predictions/` に保存されます:
- `*.jpg` - バウンディングボックス付きの認識結果画像
- `labels/` - YOLO形式のラベルファイル
- `results.json` - 全検出結果のJSON

#### 認識結果のサンプル

![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/inference_sample1.jpg)
![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/inference_sample2.jpg)
![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/inference_sample3.jpg)

### 6. リアルタイム検出 (Webカメラ)

USB Webカメラを使用したリアルタイム物体検出:

```bash
# Docker経由で実行
./run_realtime_detection.sh

# オプション指定
./run_realtime_detection.sh --camera 0 --conf 0.5

# ホストで直接実行（要: pip install ultralytics opencv-python）
./run_realtime_detection_host.sh
```

**操作方法:**
| キー | 動作 |
|------|------|
| `q` | 終了 |
| `s` | スクリーンショット保存 |
| `c` | 信頼度表示の切り替え |

## ユーティリティ

### メッシュ検証

YCBオブジェクトのメッシュ品質を自動検証:

```bash
docker compose run --rm mesh_validator
```

結果は `data/mesh_validation_results.json` に保存されます。

### サムネイル生成

全オブジェクトのgoogle_16k/tsdf形式を比較するサムネイルを生成:

```bash
docker compose run --rm thumbnail_generator
```

結果:
- `data/thumbnails/*.png` - 個別サムネイル
- `data/thumbnails/comparison_grid.png` - 比較グリッド

### 全形式サムネイル比較

全オブジェクトの4形式（clouds/google_16k/poisson/tsdf）を比較:

```bash
docker compose run --rm thumbnail_all_formats
```

結果:
- `data/thumbnails_all_formats/*.png` - 個別サムネイル
- `data/thumbnails_all_formats/comparison_grid_all.png` - 全形式比較グリッド

**注意**: clouds形式（点群）とpoisson形式はテクスチャをサポートしていないため、グレーで表示されます。

### tsdfマテリアル修正

tsdf形式のOBJファイルにマテリアル参照を追加:

```bash
docker compose run --rm fix_tsdf_materials
```

## ドメインランダム化

Sim-to-Realギャップを軽減するため、以下の要素をランダム化:

| カテゴリ | ランダム化項目 | 範囲 |
|---------|---------------|------|
| **背景** | 床テクスチャ | Wood, Concrete, Tiles, Marble, Metal, Fabric |
| | 壁テクスチャ | Concrete, Plaster, Brick, Paint, Wallpaper |
| | テーブル材質 | Wood, Metal, Plastic |
| **照明** | 光源数 | 3-5個 |
| | 色温度 | 2700K-6500K |
| | 強度 | 800-2000W相当 |
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
| Train | 25,000 | 83.3% | モデル学習 |
| Val | 2,500 | 8.3% | ハイパーパラメータ調整 |
| Test | 2,500 | 8.3% | 最終評価 |

## カスタムモデルソースの追加

YCB以外の独自3Dモデルを追加して学習データに含めることができます。

### ディレクトリ構造

```
models/
├── ycb/                          # YCBオブジェクト (クラスID: 0-102)
│   └── 002_master_chef_can/
│       └── google_16k/
│           └── textured.obj
└── tidbots/                      # カスタムオブジェクト (クラスID: 103-)
    ├── my_bottle/
    │   └── google_16k/
    │       ├── textured.obj
    │       └── textured.png
    └── my_gripper/
        └── google_16k/
            ├── textured.obj
            └── textured.png
```

### 設定ファイル

`scripts/blenderproc/config.yaml` でモデルソースを設定:

```yaml
model_sources:
  # YCBモデル
  ycb:
    path: "/workspace/models/ycb"
    include:                      # 特定オブジェクトのみ使用
      - "002_master_chef_can"
      - "005_tomato_soup_can"
      - "006_mustard_bottle"

  # カスタムモデル
  tidbots:
    path: "/workspace/models/tidbots"
    include: []                   # 空=全オブジェクト使用
```

### クラスIDの割り当て

| ソース | クラスID範囲 | 説明 |
|--------|-------------|------|
| ycb | 0-102 | 既存のYCB IDを維持 |
| tidbots | 103- | 自動で連番割り当て |
| (追加ソース) | 続きから連番 | ソース順に割り当て |

### 対応モデル形式

以下の形式を自動検出（優先順）:

1. `object_name/google_16k/textured.obj` (YCB形式)
2. `object_name/tsdf/textured.obj`
3. `object_name/textured.obj` (シンプル形式)
4. `object_name/*.obj` (任意のOBJ)

### 3Dモデルの変換

ダウンロードした3DモデルをOBJ形式に変換するスクリプトを用意しています。

**Blender形式 (.blend) → OBJ:**
```bash
docker compose run --rm -v /tmp/mymodel:/tmp/mymodel blenderproc \
  blenderproc run /workspace/scripts/convert_blend_to_obj.py \
  /tmp/mymodel/model.blend \
  /tmp/mymodel/output \
  /tmp/mymodel/textures  # テクスチャディレクトリ（オプション）
```

**COLLADA形式 (.dae) → OBJ:**
```bash
docker compose run --rm -v /tmp/mymodel:/tmp/mymodel blenderproc \
  blenderproc run /workspace/scripts/convert_dae_to_obj.py \
  /tmp/mymodel/model.dae \
  /tmp/mymodel/output
```

**FBX形式 (.fbx) → OBJ:**
```bash
docker compose run --rm -v /tmp/mymodel:/tmp/mymodel blenderproc \
  blenderproc run /workspace/scripts/convert_fbx_to_obj.py \
  /tmp/mymodel/model.fbx \
  /tmp/mymodel/output
```

変換後、出力ファイルをmodelsディレクトリにコピー:
```bash
mkdir -p models/tidbots/my_object/google_16k
cp /tmp/mymodel/output/* models/tidbots/my_object/google_16k/
```

### モデルのスケール確認・修正

ダウンロードした3Dモデルはスケールがバラバラなことが多いです。生成画像にオブジェクトが表示されない場合、スケールを確認してください。

**スケール確認:**
```bash
python3 << 'EOF'
from pathlib import Path

def get_obj_size(obj_path):
    min_c, max_c = [float('inf')]*3, [float('-inf')]*3
    with open(obj_path) as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                for i in range(3):
                    v = float(parts[i+1])
                    min_c[i], max_c[i] = min(min_c[i], v), max(max_c[i], v)
    return [max_c[i] - min_c[i] for i in range(3)]

for obj in Path('models/tidbots').glob('*/google_16k/textured.obj'):
    size = get_obj_size(obj)
    print(f"{obj.parent.parent.name}: {size[0]*100:.1f} x {size[1]*100:.1f} x {size[2]*100:.1f} cm")
EOF
```

**スケール修正（例: 0.03倍に縮小）:**
```bash
python3 << 'EOF'
from pathlib import Path
import shutil

def scale_obj(obj_path, scale):
    shutil.copy(obj_path, str(obj_path) + '.backup')
    lines = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                lines.append(f"v {float(p[1])*scale:.6f} {float(p[2])*scale:.6f} {float(p[3])*scale:.6f}\n")
            else:
                lines.append(line)
    with open(obj_path, 'w') as f:
        f.writelines(lines)

# 例: coke_zeroを0.03倍に縮小
scale_obj(Path('models/tidbots/coke_zero/google_16k/textured.obj'), 0.03)
EOF
```

**適切なサイズの目安:**
| オブジェクト | 実際のサイズ |
|-------------|-------------|
| 缶（350ml） | 6-7 × 12-13 cm |
| ペットボトル | 6-8 × 20-25 cm |
| りんご | 7-8 × 7-8 cm |

### カスタムモデルのみで学習

YCBを使わず、独自モデルのみで学習する場合:

```yaml
# scripts/blenderproc/config.yaml
model_sources:
  # YCB disabled
  # ycb:
  #   path: "/workspace/models/ycb"
  #   include: []

  # カスタムモデルのみ使用
  tidbots:
    path: "/workspace/models/tidbots"
    include: []  # 空=全オブジェクト使用

scene:
  num_images: 2000          # 少数クラスなら2000枚程度で十分
  objects_per_scene: [1, 5]  # クラス数に合わせて調整
```

**推奨データ量の目安:**
| クラス数 | 推奨枚数 | 1クラスあたり |
|---------|---------|--------------|
| 5 | 2,000 | 400枚 |
| 10 | 3,000 | 300枚 |
| 20 | 5,000 | 250枚 |
| 50+ | 10,000+ | 200枚+ |

### 特定オブジェクトのみ使用

全オブジェクトではなく、特定のオブジェクトのみを使用する場合:

```yaml
model_sources:
  ycb:
    path: "/workspace/models/ycb"
    include:
      - "002_master_chef_can"     # 缶
      - "003_cracker_box"         # 箱
      - "006_mustard_bottle"      # ボトル
      - "024_bowl"                # 食器
      - "025_mug"                 # マグカップ
```

これにより、学習対象を絞り込んで効率的にモデルを作成できます。

## YCBオブジェクトクラス (85種)

<details>
<summary>利用可能なクラス一覧を表示</summary>

### 食品・飲料 (10個)
001_chips_can*, 002_master_chef_can, 003_cracker_box, 004_sugar_box, 005_tomato_soup_can, 006_mustard_bottle, 007_tuna_fish_can, 008_pudding_box, 009_gelatin_box, 010_potted_meat_can

### 果物 (8個)
011_banana, 012_strawberry, 013_apple, 014_lemon, 015_peach, 016_pear, 017_orange, 018_plum

### キッチン用品 (11個)
019_pitcher_base, 021_bleach_cleanser, 022_windex_bottle, 023_wine_glass*, 024_bowl, 025_mug, 026_sponge, 028_skillet_lid, 029_plate, 030_fork, 031_spoon, 032_knife, 033_spatula

### 工具 (14個)
035_power_drill, 036_wood_block, 037_scissors, 038_padlock, 040_large_marker, 041_small_marker*, 042_adjustable_wrench, 043_phillips_screwdriver, 044_flat_screwdriver, 048_hammer, 049_small_clamp*, 050_medium_clamp, 051_large_clamp, 052_extra_large_clamp

### スポーツ (6個)
053_mini_soccer_ball, 054_softball, 055_baseball, 056_tennis_ball, 057_racquetball, 058_golf_ball*

### その他 (36個)
059_chain, 061_foam_brick, 062_dice*, 063-a_marbles, 063-b_marbles, 065-a〜j_cups, 070-a/b_colored_wood_blocks, 071_nine_hole_peg_test, 072-a_toy_airplane, 073-a〜m_lego_duplo*, 076_timer*, 077_rubiks_cube

**\*** tsdf形式を使用

</details>

### メッシュ形式の選択

| 形式 | 使用オブジェクト数 | 説明 |
|------|------------------|------|
| google_16k | 72個 | 高品質テクスチャ、基本形式 |
| tsdf | 13個 | google_16kで品質問題があるオブジェクト |

#### tsdf形式を使用するオブジェクト (13個)
```
001_chips_can, 041_small_marker, 049_small_clamp, 058_golf_ball, 062_dice,
073-g〜m_lego_duplo (7個), 076_timer
```

### 除外オブジェクト (6個)

以下のオブジェクトはgoogle_16k/tsdf両形式でメッシュ品質に問題があるため除外:
```
072-b_toy_airplane, 072-c_toy_airplane, 072-d_toy_airplane,
072-e_toy_airplane, 072-h_toy_airplane, 072-k_toy_airplane
```

## 設定ファイル

### データ生成設定 (`scripts/blenderproc/config.yaml`)

```yaml
# モデルソース設定（複数ソース対応）
model_sources:
  ycb:
    path: "/workspace/models/ycb"
    include: []                   # 空=全オブジェクト使用
    # include:                    # 特定オブジェクトのみ使用する場合
    #   - "002_master_chef_can"
    #   - "005_tomato_soup_can"

  tidbots:                        # カスタムモデルソース
    path: "/workspace/models/tidbots"
    include: []

scene:
  num_images: 30000
  objects_per_scene: [2, 8]    # シーンあたりのオブジェクト数

rendering:
  engine: "CYCLES"
  samples: 32                   # 32=高速(推奨), 128=高品質
  use_denoising: true
  use_gpu: true

camera:
  distance: [0.4, 0.9]          # オブジェクトが大きく写る距離
  elevation: [35, 65]           # テーブルを見下ろす角度
  azimuth: [0, 360]

lighting:
  num_lights: [3, 5]            # 最低3つのライトで十分な照明を確保
  intensity: [800, 2000]        # 明るめの照明（暗すぎるシーンを防止）
  ambient: [0.4, 0.7]           # 環境光も強めに設定

placement:
  position:
    x_range: [-0.25, 0.25]      # グリッド配置の範囲
    y_range: [-0.25, 0.25]
  use_physics: false            # 物理シミュレーション無効（グリッド配置を使用）
```

### 学習設定 (`scripts/training/train_config.yaml`)

```yaml
model:
  architecture: yolo26n         # nano / small / medium
  weights: /workspace/weights/yolo26n.pt
  num_classes: 85               # 利用可能なクラス数（除外オブジェクトを除く）

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

### 学習結果 (`outputs/trained_models/`)

```
outputs/trained_models/ycb_yolo26_run/
├── weights/
│   ├── best.pt              # ベストモデル (mAP基準) - 推論用に推奨
│   ├── last.pt              # 最終エポックモデル
│   └── epoch*.pt            # チェックポイント (10エポックごと)
├── args.yaml                # 学習パラメータ
├── results.csv              # エポックごとのメトリクス
├── labels.jpg               # ラベル分布
└── train_batch*.jpg         # 訓練バッチサンプル
```

### 推論結果 (`outputs/inference_results/`)

```
outputs/inference_results/predictions/
├── *.jpg                    # バウンディングボックス付き認識結果画像
├── labels/                  # YOLO形式のラベルファイル
└── results.json             # 全検出結果 (JSON)
```

## 生成時間の目安

| 設定 | samples | 速度 | 30,000枚の所要時間 |
|-----|---------|------|-------------------|
| 高速 | 32 | ~45枚/分 | ~11時間 |
| 高品質 | 128 | ~10枚/分 | ~50時間 |

## 追加学習（Incremental Learning）

学習済みモデルに新しいオブジェクトを追加する方法です。全データを再学習せずに効率的に拡張できます。

### 手法の比較

| 手法 | 学習時間 | 精度維持 | 実装難易度 |
|------|---------|---------|-----------|
| 全データ再学習 | 長い | 高い | 簡単 |
| リプレイ (サブセット) | 短い | やや低下 | 簡単 |
| Backbone凍結 | 短い | やや低下 | 簡単 |
| 知識蒸留 | 中程度 | 高い | やや複雑 |

### ステップ1: サブセット作成

元データから代表的なサンプルを抽出（クラス均等サンプリング）:

```bash
docker compose run --rm yolo26_train python \
  scripts/data_processing/create_subset.py \
  --source /workspace/yolo_dataset \
  --output /workspace/data/ycb_subset \
  --num_samples 5000 \
  --val_samples 500
```

### ステップ2: データ統合

サブセットと新しいオブジェクトのデータを統合:

```bash
docker compose run --rm yolo26_train python \
  scripts/data_processing/merge_for_incremental.py \
  --base /workspace/data/ycb_subset \
  --new /workspace/data/new_objects \
  --output /workspace/data/merged_dataset
```

### ステップ3: Backbone凍結で追加学習

```bash
docker compose run --rm yolo26_train python \
  scripts/training/train_incremental.py \
  --weights /workspace/outputs/trained_models/ycb_yolo26_run/weights/best.pt \
  --data /workspace/data/merged_dataset/dataset.yaml \
  --freeze 10 \
  --epochs 30 \
  --lr0 0.001
```

### パラメータ比較

| パラメータ | 通常学習 | 追加学習 |
|-----------|---------|---------|
| データ量 | 30,000枚 | 5,500枚 |
| freeze | 0 | 10 |
| lr0 | 0.01 | 0.001 |
| epochs | 50-100 | 30-50 |
| **推定時間** | ~1時間 | ~15分 |

## アンサンブル推論

複数のモデルを組み合わせて推論する方法です。モデルを追加・削除する際に再学習が不要です。

### 手法の比較

| 項目 | 追加学習 | アンサンブル推論 |
|------|---------|-----------------|
| 推論速度 | 速い（1モデル） | 遅い（N回推論） |
| メモリ | 少ない | 多い（N倍） |
| 精度維持 | 忘却リスクあり | 各モデル維持 |
| 柔軟性 | 再学習が必要 | モデル追加/削除が容易 |

### 画像推論

```bash
docker compose run --rm ensemble_inference python \
  scripts/inference/ensemble_inference.py \
  --models weights/yolo26n.pt outputs/ycb_best.pt \
  --source data/test_images/ \
  --output outputs/ensemble_results \
  --show-model
```

### リアルタイム推論（Webカメラ）

```bash
docker compose run --rm ensemble_inference python \
  scripts/inference/ensemble_inference.py \
  --models weights/yolo26n.pt outputs/ycb_best.pt \
  --source 0 \
  --realtime
```

### Pythonコードでの使用

```python
from ensemble_inference import EnsembleDetector

# 複数モデルを初期化
detector = EnsembleDetector(
    model_paths=[
        'yolo26n.pt',      # COCO 80クラス (ID: 0-79)
        'ycb_best.pt',     # YCB 85クラス  (ID: 80-164)
        'custom.pt',       # カスタム     (ID: 165+)
    ],
    conf_threshold=0.3,
    iou_threshold=0.5,
)

# 推論
detections = detector.predict(image)

# 結果を描画
result = detector.draw_detections(image, detections, show_model=True)
```

### 推奨ケース

| ケース | 推奨手法 |
|--------|---------|
| リアルタイム検出が必要 | 追加学習 |
| 精度が最優先 | アンサンブル |
| モデルを頻繁に更新 | アンサンブル |
| エッジデバイス | 追加学習 |
| GPU複数台あり | アンサンブル（並列実行可） |

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

### YCBモデルが見つからない

google_16kまたはtsdf形式のモデルが必要です:

```
models/ycb/{object_name}/google_16k/textured.obj
models/ycb/{object_name}/tsdf/textured.obj
```

ダウンロード:
```bash
# google_16k形式
python scripts/download_ycb_models.py --all --format google_16k

# tsdf形式（一部オブジェクトで必要）
python scripts/download_ycb_models.py --all --format berkeley
```

### tsdf形式でテクスチャが表示されない

tsdf形式のOBJファイルはマテリアル参照（usemtl）が欠落しています。修正スクリプトを実行してください:

```bash
docker compose run --rm fix_tsdf_materials
```

### テクスチャが壊れて表示される

`poisson`形式を使用している可能性があります。`google_16k`または`tsdf`形式を使用してください。

### 特定のオブジェクトのメッシュが歪む

メッシュ品質を確認するには、サムネイル生成スクリプトを使用:

```bash
# サムネイル生成（全オブジェクトのgoogle_16k/tsdf比較）
docker compose run --rm thumbnail_generator

# 結果を確認
xdg-open data/thumbnails/comparison_grid.png
```

問題のあるオブジェクトを発見した場合、`generate_dataset.py`で設定:

```python
# scripts/blenderproc/generate_dataset.py

# 完全に除外するオブジェクト
EXCLUDED_OBJECTS = {
    "072-b_toy_airplane",
    "問題のあるオブジェクト名",  # 追加
}

# tsdf形式を使用するオブジェクト（google_16kに問題がある場合）
USE_TSDF_FORMAT = {
    "001_chips_can",
    "問題のあるオブジェクト名",  # 追加
}
```

### メッシュ品質の自動検証

```bash
# メッシュの自動検証（Non-manifold edges等をチェック）
docker compose run --rm mesh_validator

# 結果を確認
cat data/mesh_validation_results.json
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
