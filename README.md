# HDBSCAN_treetool
クラスタリングに自作HDBSCANを用いた際のプログラム
##  仮想環境のセットアップ (Conda)

このプロジェクトを実行するには、`environment.yml`ファイルを使用して、必要な依存関係を含む仮想環境を作成する．
足りない場合には各自インストールお願いします．

### 1. 仮想環境の作成

プロジェクトのルートディレクトリで以下のコマンドを実行し、`treetool`という名前の仮想環境を作成します。

```bash
conda env create --file environment.yml
```

### 2. 仮想環境のアクティベート

環境が作成されたら以下のコマンドでアクティベートします．
```bash
conda activate treetool
```

