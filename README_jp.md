# limbs3d

3Dスキャンした四肢データ（現状は脚を想定）から、周径や体積を計測するためのPythonパッケージです。

## できること（ざっくり）

- 点群データの前処理（床合わせ、床除去、ノイズ除去など）
- 左右脚の分離
- 点群からメッシュ生成、穴埋め
- メッシュ体積の計算
- 周径計測に使う補助処理

## 必要環境

- Python 3.11以上

主な依存ライブラリ:

- open3d
- pyvista
- pymeshfix
- scikit-learn
- lsq-ellipse

## インストール

### 1) 依存関係を入れる

このリポジトリ直下で:

```bash
uv sync
```

または:

```bash
pip install -r requirements.txt
```

### 2) パッケージとして使う

```bash
pip install -e .
```

## クイックスタート

```python
import open3d as o3d
from limbs3d.leg import separate2legs, align2Floor, deleteFloor
from limbs3d.volume import get_surface, get_fix, get_vol

# 点群を読み込み
pcd = o3d.io.read_point_cloud("your_leg_pointcloud.ply")

# 前処理
pcd = align2Floor(pcd)
pcd = deleteFloor(pcd)

# 左右脚に分離
left_leg, right_leg = separate2legs(pcd)

# 左脚の体積を計算（例）
mesh = get_surface(left_leg, depth=10)
mesh_fixed = get_fix(mesh)
volume = get_vol(mesh_fixed)

print("volume:", volume)
```

## サンプル

- examples フォルダにノートブック例があります。
- まずは perimeter/volume 系のノートブックを見るのがおすすめです。

## 注意

- 入力データの座標系やスケール（mm, cmなど）で結果が変わります。
- 床検出やクラスタ分離の閾値は、スキャン条件に合わせて調整が必要です。
- 現時点では汎用の四肢対応というより、脚データ向けの実装です。

## ライセンス

LICENSE を参照してください。

