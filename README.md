# Adaptive Group Encoding
[Protecting Adaptive Sampling from Information Leakage on
Low-Power Sensors](https://dl.acm.org/doi/pdf/10.1145/3503222.3507775)
の再現実験手順を記す。
なお、再現対象は論文内のFigure 6、Table 4の二つである。

## 再現の手順
### 1. 環境構築
[git](https://github.com/tejaskannan/adaptive-group-encoding)からコードをcloneする。

仮想環境を用いる。下記のコマンドで仮想環境を作り、仮想環境内に入る。
```
python3 -m venv adaptiveleak-env
. adaptiveleak-env/bin/activate
```
下記のコマンドを用いてパッケージをインストールする。このコマンドはrootディレクトリで行う。
```
pip3 install --upgrade pip
pip3 install -e .
```

### 2. データセットの入手
[Google Drive](https://drive.google.com/drive/folders/1BrXn-Spc3GwbSmZu-xI5mLefBqNQ8vMa?usp=sharing)からダウンロードする。zipファイルを解凍し、以下のディレクトリにそれぞれ配置する。

1. `datasets/datasets` -> `adaptiveleak/datasets`
2. `saved_models/saved_models` -> `adaptiveleak/saved_models`
3. `traces/traces` -> `adaptiveleak/traces`
<!-- 4. `msp_results.zip` -> `adaptiveleak/device/results` -->

### 3. シミュレーション
#### サンプリング
```
cd adaptiveleak
./run_simulator.sh <dataset-name>
```
結果は`saved_models/<dataset-name>/<date>`に保存される。
`<date>`は次の攻撃シミュレーションにおいて使用する。なお、我々の実験データは`saved_models/<dataset-name>/2023-11-15`、筆者らのデータは`saved_models/<dataset-name>/results`に保存されている。

#### 攻撃シミュレーション
`adaptiveleak/attack`ディレクトリに移動して行う。

```
python train.py --policy <policy-name> --encoding <encoding-name> --dataset <dataset-name> --folder <date> --window-siz/home/mtanaka/adaptive-group-encoding/adaptiveleak/saved_models/eoge <window-size> --num-samples <num-samples>
```

論文内の数値を再現するために、以下の組み合わせを全て実行する。

`--policy`：adaptive_deviation, adaptive_heuristic  
`--encoding`：group, standard, padded  
`--dataset`：uci_har, trajectories, eog, haptics, mnist, pavement, tiselac, strawberry, epilepsy  
`--folder`：`<date>`  
`--window-size`：10  
`--num-samples`：10000  


### 4. シミュレーション結果のプロット
`adaptiveleak/analysis`ディレクトリに移動して行う。

論文中のFigure 6を再現するためには、以下を実行する。
```
python plot_all_attacks.py --folder <experiment-name> --datasets uci_har trajectories eog haptics mnist pavement tiselac strawberry epilepsy --output-file [<output-path>]
```

上記を実行した際に
`OSError: 'seaborn-ticks' is not a valid package style, path of style file, URL of style file, or library style name (library styles are listed in 'style.available')`
が出た場合の対応は注意事項に記載。

論文中のTable 4を再現するためには、以下を実行する。
```
python plot_error.py --folder <experiment-name> --dataset <dataset-name> --metric mae --output-file [<output-path>]
```
出力された数字を表にまとめることでTable 4と同じ表を得ることができる。

## 評価環境
Ubuntu(WSL 2)


## 注意事項
### 命名規則
論文と異なる命名規則がある。
論文内の用語は、コードにおいて以下のように書き換えられている。
`AGE` -> `group`  
`Linear policy` -> `adaptive heuristic policy`  
`Activity`データセット -> `uci_har`  
`Characters`データセット -> `trajectories`  
`Password`データセット -> `haptics`  

### `'seaborn-ticks' is not a valid package style` errorへの対処
`analysis/plot_utils.py`の`PLOT_STYLE = seaborn-ticks`をコメントアウトし`PLOT_STYLE = 'seaborn-v0_8'`に書き換える。
```
# PLOT_STYLE = 'seaborn-ticks'
PLOT_STYLE = 'seaborn-v0_8'
```

