# zernike_batch

Spatial wafer profile modelling and reconstruction pipeline supporting multiple basis decompositions (Zernike, Legendre, RBF, Wavelet) and Gaussian-process regressors with uncertainty analytics.

---

## 1. Environment Setup

```bash
# create and activate a virtual environment
python3 -m venv wafer_env
source wafer_env/bin/activate

# upgrade pip
pip install --upgrade pip

# install runtime dependencies
pip install \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    PyWavelets \
    PyYAML \
    joblib
```

### Training & Inference

```bash
# training (writes outputs to results/)
python train.py --config config.yaml

# inference (requires trained models in results/)
python predict.py --config_pred config_pred.yaml
```

---

## 2. Project Layout

```
zernike_batch/
├── config.yaml              # training configuration
├── config_pred.yaml         # inference configuration
├── data/                    # conditions.csv + per-wafer distributions
├── results/                 # training artefacts, metrics, plots
├── results_test/            # optional sandbox for experiments
├── train.py                 # end-to-end training pipeline
├── predict.py               # inference script
├── wafer_ml/
│   ├── config.py            # YAML loader into Config dataclass
│   ├── data_loader.py       # CSV ingestion helpers
│   ├── evaluation.py        # metrics (R², RMSE, MAE)
│   ├── features/            # basis extraction modules
│   │   ├── zernike.py       # Zernike basis fitting & reconstruction
│   │   ├── legendre.py      # Legendre polynomial features
│   │   ├── rbf.py           # shared-centre RBF features with ridge solver
│   │   └── wavelet.py       # 2-D wavelet decomposition utilities
│   ├── models/              # regression backends
│   │   ├── linear_regression.py
│   │   ├── gpr.py           # multi-output Gaussian process wrapper
│   │   └── zernike_gpr.py   # hybrid Zernike + GPR model
│   ├── preprocessing.py     # tabular preprocessors + spline resampling
│   ├── utils.py             # file I/O, residual helpers
│   └── visualization.py     # plotting utilities (heatmaps, uncertainty, etc.)
└── README.md
```

### High-Level Flow

```mermaid
flowchart TD
    A[Load config.yaml] --> B[Read conditions.csv]
    B --> C[Load wafer distributions]
    C --> D[Spline resampling (polar/cartesian)]
    D --> E[Train/test split <br/> + preprocess conditions]
    E --> F{Enabled methods?}
    F -->|Zernike Linear| G1[Compute Zernike coeffs<br/>Train LinearRegression]
    F -->|Legendre| G2[Legendre coeffs<br/>Train LinearRegression]
    F -->|RBF| G3[Shared centres, ridge fit<br/>Train Linear/GPR]
    F -->|Wavelet| G4[Wavelet coeffs<br/>Train Linear/GPR]
    F -->|Direct GPR| G5[Flattened distributions<br/>Train multi-output GPR]
    F -->|Zernike GPR| G6[Hybrid coeff GPR]
    G1 --> H[Reconstruct test wafers]
    G2 --> H
    G3 --> H
    G4 --> H
    G5 --> H
    G6 --> H
    H --> I[Metrics & scatter plots]
    H --> J[Sample diagnostics (heatmaps, coeff histograms)]
    G3 --> K[Uncertainty analytics (if GPR)]
    G4 --> K
    G6 --> K
    I --> L[results/<method>/metrics.csv]
    J --> M[results/<method>/samples/]
    K --> N[results/<method>/uncertainty/]
    K --> O[results/uncertainty_summary/]
```

---

## 3. Model Overview

| 手法 | 概要 | ゼルニケ + 線形回帰との比較: メリット | デメリット | 解釈性・不確実性評価 | 推奨ユースケース |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Zernike Linear | 既知の円盤領域に対して低次Zernike基底＋線形回帰 | 既存ベースライン | 高次成分に弱い | 係数ヒストグラムと真値比較のみ。不確実性評価なし | 迅速なベースライン、球面/円盤対称性が強い場合 |
| Legendre | 直交Legendre基底で矩形領域を展開、線形回帰 | 非円形パターンや境界条件の違いに強い | 基底数が増えやすく、円盤境界に弱い | 係数プロット、係数残差のみ | 任意矩形エリアの粗い近似、比較研究 |
| RBF + GPR | 正規化座標・共有センター RBF＋リッジ解＋GPR | ローカル変動に強く、不確実性（σ）取得 | センター数により計算コスト増、ハイパーパラ調整が必要 | 可視化: 係数分散、特徴量重要度、2D不確実性ヒートマップ | 中局所構造が強いデータ、特徴量感度解析が必要なケース |
| Wavelet + GPR | 2D Wavelet 分解（PyWavelets）＋GPR | マルチスケール表現による高周波取り込み、不確実性付き | 境界条件調整が難しい、係数整理が複雑 | 同上の無い線形と比較しσ可視化あり | エッジ/高周波成分が重要な分布再構成 |
| Direct GPR | 条件→全分布を直接予測 | モデル間で最も柔軟に任意パターンを表現 | 次元数が高く計算量が大きい | 分散取得可（散布図のみ利用） | 少数サンプル＋連続分布の直接予測 |
| Zernike GPR | Zernike係数をGPRで回帰後、分布再構成 | ベースラインの物理解釈＋不確実性評価の両立 | 高次基底が多いとGPRコスト増 | 係数σプロット、特徴量重要度、ヒートマップあり | 円盤領域＋不確実性定量化が必要な解析 |

---

## 4. `config.yaml` 主な設定

| セクション | キー | 役割 / バリエーション |
| -------- | --- | ---------------------- |
| `data` | `conditions_file`, `distribution_dir`, `id_column` | 入力CSV・IDカラム指定 |
| `preprocessing` | `standardize`, `normalize`, `impute`, `distribution_spline` | Tabular処理と分布スプライン（`mode: polar/cartesian`, `n_radii`, `n_angles`, `max_radius`, `radius_padding`, `max_points`） |
| `methods` | 各手法の `enabled`, ハイパーパラ | 例: `zernike_linear.max_order`, `rbf.n_centers`, `wavelet.n_coeffs`, `*_gpr.length_scale`, `optimize_hyperparams` 等 |
| `training` | `test_size`, `random_seed` | データ分割 |
| `output` | `directory` | 出力ディレクトリ |
| `visualization` | `sample_ids` (重点サンプル出力), `heatmap` (グリッド密度・境界色), `uncertainty` (`enabled`, `features`=不確実性ペアに使う特徴量) |

同様に `config_pred.yaml` では予測対象の条件CSVと既存モデルディレクトリを指示します。

---

## 5. 出力ファイルと意味

| パス | 内容 |
| ---- | ---- |
| `results/<method>/metrics.csv` | 分布/係数の R²・RMSE・MAE |
| `results/<method>/distribution_scatter.png` | 予測 vs 真値の散布図 |
| `results/<method>/coefficients_scatter.png` | 係数真値 vs 予測散布図（該当手法のみ） |
| `results/<method>/samples/<id>/comparison_heatmap.png` | オリジナル/再構成/残差ヒートマップ (jet, ウェーハ境界マスク) |
| `results/<method>/samples/<id>/true_vs_pred_scatter.png` | サンプル別真値 vs 予測散布 |
| `results/<method>/samples/<id>/coeff_hist.png` | 真値・予測係数の棒グラフ比較 |
| `results/<method>/samples/<id>/coefficients.csv` | 係数真値/予測表 |
| `results/<method>/uncertainty/coeff_variance_bar.png` | 係数ごとの平均分散棒グラフ (GPR系) |
| `results/<method>/uncertainty/coeff_relative_uncertainty.png` | 相対不確実性 (σ/|μ|) |
| `results/<method>/uncertainty/coeff_histograms.png` | 全サンプル係数ヒストグラムグリッド |
| `results/<method>/uncertainty/feature_importance.png` + `importance_description.txt` | 特徴量重要度バー & 算出説明 |
| `results/<method>/uncertainty/uncertainty_heatmap_*_vs_*.png` | 指定特徴量ペアに対する平均σカラーマップ |
| `results/<method>/uncertainty/coefficient_uncertainty_stats.csv` | 係数別統計 (平均σ, 分散等) |
| `results/uncertainty_summary/coefficient_uncertainty_summary.csv` | 全手法の不確実性統計まとめ |
| `results/uncertainty_summary/uncertainty_overview.png` | 手法別平均σ/分散の比較図 |
| `results/<method>/model.pkl` | 学習済みモデル（ピクル保存） |

これらを参照することで、モデル性能の定量評価だけでなく、特徴量感度や係数レベルのロバスト性、不確実性分布を包括的に把握できます。

---

以下に、report.md の内容全体をそのまま示します。Markdown ファイルにコピペしてご利用ください。

ウェハ面内エッチアマウント分布の特徴量化と予測に役立つ複数手法の比較

以下の表では、半導体バッチ装置で各高さ（ウェハロット位置）におけるウェハ面内エッチアマウント分布を特徴量化・予測するための候補手法をまとめます。各手法について技術概要、必要な入力データ、学習データ構造、出力データ、想定される応用分野および機械学習上の利点や課題を示します。参考文献の内容を必要に応じて引用しています。

手法	技術概要	入力データ	学習データ	出力データ	応用分野	ML効果（利点／課題）
Zernikeポリノミアル展開＋線形回帰	ウェハ分布を半径方向に対して正規化し、正規直交基底であるZernikeポリノミアルでモード分解する。得られた各係数をエッチ量の特徴量として用い、プロセス条件や高さ（ロット内位置）を入力として係数を予測するための回帰モデルを構築する。Zernike多項式は単位円上で直交するため項間の共線性を低減し、オーバーレイ補正の安定性が向上する ￼。	ウェハ座標（半径および角度）とエッチ量の2Dマップ、プロセス条件変数、ロット内高さ	各ウェハから計算したZernike係数ベクトルと対応する条件／高さ。2〜3ロット分の繰返しデータを使用して平均・分散を算出	予測されたZernike係数と、それを用いた再構成分布。係数のばらつきや異常検知も可能	ウェハ表面形状のモデリング、フォトリソグラフィのオーバーレイ補正	直交基底により特徴量が少数で済み、線形回帰が安定する。実際のHVM評価ではZernikeによる補正でオーバーレイばらつきが7 %低減し、線形モデルのばらつきが22 %減少し歩留りが0.1 %向上した ￼。ただし低次のZernikeでは局所的な非対称パターンを捉えにくく、高次になると過学習やノイズ増幅の課題がある。
Legendreポリノミアルなど直交多項式による分解	ウェハ面を直交領域（x,y∈[–1,1]）へ写像し、LegendreやChebyshev多項式など直交基底で二次元展開する。Zernikeが円環対称成分を捉えるのに対し、Legendre系は矩形領域の対称・非対称パターンを表現しやすい。基底係数をプロセス条件と高さから回帰予測する。	正規化したx・y座標とエッチ量マップ、プロセス条件、高さ	各ウェハから計算したLegendre係数（高次の多項式まで含める）、条件／高さ	予測した係数ベクトルと再構成マップ。必要に応じて局所勾配や粗さ指標も出力	オーバーレイ補正のフィールドレベル成分モデリング、表面形状評価	Zernikeと同様に直交性があり項間の干渉が少ない。矩形領域での非対称パターンを表現でき、基底の次数を増やしても計算は容易。課題は高次数になるとノイズに敏感になり、パラメータ数が増加すること。
RBF (Radial Basis Function) 補間／RBFネットワーク	Radial basis functionを用いて多変量関数を補間する手法。分布を基底関数 φ(∥x−ξ∥) の線形和として表現し、各中心ξは測定点に対応する。RBFは高次元でも適用でき、グリッド外の散在データ補間にも適する ￼。RBFネットワークとして学習すれば分布パターンを非線形に近似可能。	各ウェハの測定点座標（散在点でも可）とエッチ量、プロセス条件	各ウェハに対するRBF係数ベクトル／ネットワーク重みとプロセス条件、高さ	予測したRBF係数、再構成したエッチ量分布。新条件・新高さでの連続的な分布予測	散在データ補間、表面形状モデリング、ベイズ/カーネル法のサロゲートモデル	RBFは高次元でもテンサープロダクト法より少量のデータで精度良く近似でき ￼、非線形パターンを表現可能。小規模データでも安定だが、カーネル幅や正則化の選択が難しく、データ数が増えると計算コストが高くなる。
勾配ベース特徴量（ヒストグラム・オブ・オリエンテッド・グラディエント）	分布の局所的な勾配方向と大きさを計算し、小領域ごとに勾配の方向ヒストグラムを作成するHOG特徴量を用いる。HOGは物体検出に広く使われ、局所勾配方向の発生頻度を数え、対照正規化により精度を向上させる ￼。これにより端部や斜めの勾配パターンを数値化でき、非対称な不均一分布の識別に有効。	ウェハエッチ量画像（グリッド状にリサンプリング）、プロセス条件、高さ	各ウェハについて計算したHOGベクトルと条件／高さ	予測したHOG特徴やパターンクラス。特徴の主成分分析やクラスタラベル	パターン分類、異常検知、局所勾配解析	HOGは局所勾配やエッジ方向の情報を捕捉し、照度変化や回転に対して比較的頑健である ￼。機械学習により代表パターンへのクラスタリングや条件との相関解析が可能。一方で高次元になるため次元削減が必要で、ノイズや解像度に敏感。
ウェーブレット変換によるマルチスケール特徴	離散ウェーブレット変換で分布を周波数帯に分解し、各サブバンドの平均・標準偏差・エネルギー等の統計量を特徴量とする。ウェーブレットは時系列や画像の時間‐周波数解析に利用され、画像信号を周波数サブバンドに分解して異常を検出する ￼。境界処理とマルチスケールエッジ検出に優れ、指紋や汚れ検出に効果があるが、尖鋭なマイクロクラックには不向きという報告もある ￼。	ウェハ分布の2D画像、プロセス条件、高さ	各ウェハのウェーブレット係数や統計量と条件／高さ	予測したウェーブレット特徴および逆変換による分布再構成	欠陥検出、多重解像度分析、異常パターン分類	マルチスケールで局所と全体のパターンを捉え、ノイズに頑健。小データでもエネルギー分布等の低次統計を学習できる。鋭いエッジや複雑な形状には別途特徴を併用する必要がある。
PCA／SVDとクラスタリング	各ウェハ分布をベクトル化して主成分分析 (PCA) や特異値分解 (SVD) により空間変動の主要モードを抽出し、ノイズを除去する。得られた低次元特徴に対して階層型クラスタリングやk‑meansを適用してパターン毎にグループ化する。SVD→クラスタリング→辞書学習で系統的な欠陥パターンを同定する方法が提案されており ￼、ランダム欠陥が混入したデータでも再現性の高いパターンを抽出できる。	ベクトル化したウェハ分布、プロセス条件、高さ	各ウェハの主成分スコアや上位特異ベクトル、クラスタラベル	主要成分係数、クラスタ分類、パターン毎の平均分布	欠陥パターン同定、再現性評価、異常検知	データ主導の基底により分散説明率の高いモードを抽出し、小データでも効率的に特徴量が減らせる。階層クラスタリングにより異常ウェハを外れ値として除外できる ￼。線形性の仮定により非線形パターンが捉えにくいことや、適切な成分数の選択が必要。
オートエンコーダ／変分オートエンコーダ (VAE)	自己符号化器は入力データを圧縮し再構成するニューラルネットワークで、データ圧縮と再構成を通して意味のある潜在表現を学習する。エンコーダが入力の高次元マップから低次元潜在空間へ写像し、デコーダが元の分布を再構成する。潜在ベクトルをプロセス条件や高さと結合して回帰することで分布を予測したり、VAEやGANと組み合わせてデータ拡張を行うことができる。	ウェハ分布の画像、プロセス条件、高さ（教師ありの場合は条件を付加）。データ増量のため回転・反転などの拡張。	無ラベルのウェハ分布（自己教師あり）。条件を付加して条件付きVAEやConditional GANで学習する場合は条件ラベルも含む。	潜在ベクトル、再構成分布、あるいは生成された新規分布。	異常検知、特徴抽出、データ生成、少量データの補完	オートエンコーダはラベル無しデータでも低次元表現を学習でき、特徴抽出や次元削減に有効。デノイジングオートエンコーダはノイズを除去し、VAEは確率モデルとして新しい分布の生成や不確実性評価が可能。データ量が少ないと過学習しやすく、ネットワークのハイパーパラメータ調整が重要。
CNN・深層学習＋データ拡張	ウェハ分布を画像とみなし、畳み込みニューラルネットワーク (CNN) で自動的に特徴抽出を行う。条件や高さを入力とし、出力層に回帰や分類を置いて分布パターンや係数を予測する。データ不足を補うため、回転・対称性利用やGenerative Adversarial Network (GAN) によるデータ拡張を用いる。例えばウェハ欠陥分類では、グローバルとローカル特徴を別々に抽出してGANでデータを生成するG2LGAN法が提案され、データ不均衡下でも高精度を達成した ￼。	ウェハ分布画像（グリッド化した2Dデータ）、プロセス条件、高さ	条件付きでラベル付けした分布パターン。GANの場合は生成器・識別器の訓練に使用。	予測された分布マップまたはパターン分類結果。GANでは合成ウェハ分布。	不良パターン分類、条件からの分布予測、データ拡張	CNNは畳み込みにより階層的特徴を自動抽出し、人間より高い認識精度を達成する ￼。GANを用いたウェハマップ分類ではAccuracy 98.39%、F1スコア93.01%を達成した ￼。しかしモデルのパラメータ数が多く、学習には大量データが必要。少数データの場合、データ拡張や事前学習、正則化が不可欠。
ガウス過程回帰 (GPR) /クリギング	関数値の集合に対し共分散（カーネル）関数を定義し、観測データに基づいて未知の関数の事後分布を得るベイズ的非パラメトリック回帰手法。ガウス過程はパラメータ数を固定せず、観測されたデータに整合的な関数の分布を推定する ￼。エッチ量分布をプロセス条件と高さの関数として扱い、空間座標を含むカーネルにより空間的な相関を表現することで少数データから連続的な予測が可能。	プロセス条件変数、高さインデックス、場合によっては位置座標; 観測されたエッチ量	観測データ（条件・高さ・位置とエッチ量）、カーネル関数のハイパーパラメータ	新たな条件・高さ・位置におけるエッチ量の予測分布（平均と分散）	サロゲートモデリング、少量データの関数近似、予測と不確実性評価	ガウス過程は非パラメトリックで、小規模データでも柔軟に関数形状を捉えられる。カーネルにより「近い入力は近い出力を持つ」という仮定を組み込める ￼。予測と同時に信頼区間が得られるため、高さや条件による再現性の評価に有用。データ数が増えると計算量がO(N^3)に増大するので近似や低ランク化が必要。

考察と活用指針
	•	学習データが少ないことへの対策: ZernikeやLegendre展開、PCA／SVDなど基底展開に基づく手法は、少ないデータでも安定して特徴量を抽出できる。基底の次数を上げすぎると過学習しやすいので、再構成誤差や交差検証により次数を決める。また、勾配やウェーブレット統計量など手作業で設計する特徴はデータ量を必要としない。
	•	パターンのクラスタリングと再現性評価: PCA＋クラスタリングやSVD＋辞書学習では、各条件・高さにおける分布をクラスタリングして高再現性パターンと低再現性パターンを識別できる ￼。ランダム欠陥やノイズがある場合でも主成分に投影することで除去できる。
	•	非線形パターンや複雑な形状: 勾配やウェーブレット、RBF、GPR、オートエンコーダ／CNNなど非線形手法を組み合わせると局所的かつ複雑な分布を捉えやすい。CNNやGANは高精度だがデータ量が多く必要なので、シミュレーションデータや物理モデルによる合成データを生成し事前学習させると良い。
	•	不均一性要因の説明: 特徴量をプロセス条件や高さと回帰・分類すると、不均一分布の要因を推定できる。たとえばZernike係数に対する線形回帰モデルから特定係数と条件の相関を抽出し、フィードバック制御に活用する。GPRやベイズ回帰を用いると条件ごとの不確実性を評価でき、再現性の低いパターンを特定できる。

以上の手法を組み合わせることで、少ない学習データからウェハ面内エッチ量分布の特徴付けと予測を行い、プロセス条件やロット内高さの影響を分析できる。実際のデータ特性や目的に合わせて複数の手法を試行し、再現性評価や特徴の物理的解釈を行うことが望ましい。

ゼルニケ展開に対するガウス過程回帰の応用

バッチ装置のロット内高さごとに計測される面内エッチ量分布を特徴量化する際、Zernike係数などの目的変数が互いに相関を持つことが多い。こうした相関を明示的に扱う手法として、ガウス過程回帰 (Gaussian Process Regression, GPR) を応用することが考えられる。GPRはカーネル関数を通じて入力空間における近接点の出力相関を表現し、予測値とともに信頼区間（予測分散）を提供する非パラメトリック手法である ￼ ￼。以下では、Zernike展開や面内分布の仮説共分散モデルに対するGPRの応用事例とメリット・デメリットをまとめる。

GPR活用事例
	•	プラズマエッチ・バーチャルメトロロジー – LynnらはプラズマエッチのバーチャルメトロロジーにGPRを適用し、プロセス中に得られる温度やガス流量などからエッチレートを予測した ￼。訓練データとテストデータを時系列・インターリーブで分けた実験では、モデル性能を平均絶対百分率誤差 (MAPE) と決定係数R²で評価し、ステップワイズ選択した入力と二乗指数カーネルを用いたGPRが最良のMAPEを達成し、ニューラルネットワークよりも精度が高かった ￼。またGPRは予測値と共に信頼区間を出力でき、外乱による高周波変動も区間内に含められるため、エンジニアにとって有用である ￼。
	•	バーチャルメトロロジー付きランツーラン制御 – WanとMcLooneは、GPRを用いてバーチャルメトロロジーを行い、その平均値と分散をRun‑to‑Run (R2R) 制御のEWMA係数調整に利用する手法を提案した ￼。GPRの予測平均をVM値とし、予測分散から導く「Gaussian Reliance Index」を用いて制御ゲインを動的に調整することで、予測信頼度に応じてプロセス入力を変更し、従来の線形モデルより優れた制御性能を示した ￼。
	•	前眼部トポグラフィと屈折値予測 – Espinosaらは、角膜表面のZernike係数や眼球長を特徴量として主観的屈折度を予測する複数の機械学習アルゴリズムを比較し、Gaussian Process Regressionが最も低い平均絶対誤差を達成した ￼。高次のZernike分解によっても予測精度は大きく向上せず、GPRは小さなデータセットでも安定して学習できると報告している ￼。
	•	レーザ駆動プロトンビームの最適化 – Glennらはレーザ駆動プロトンビーム生成において、焦点波面の歪みを表す六つのZernike係数を調整するためBayesian Optimizationを採用し、そのサロゲートモデルとしてGPRを使用した ￼。実験の各バーストで得たデータからGPRの事後分布を更新し、期待改良を最大化するZernike係数を次の試行に提示することで、プロトンエネルギーを11 %向上させた ￼。

ゼルニケ展開＋GPRの比較表

以下の表は、先述の各手法に加えて、Zernike展開から得た係数や面内パターンの共分散を考慮したGPRモデルを追加し、技術概要や利点・課題をまとめたものである。Zernike係数など目的変数の共分散構造を明示的に扱うため、単一の出力を独立に予測する線形回帰に比べて複数出力を同時にモデリングできる点が特徴である。

手法	技術概要	入力データ	学習データ	出力データ	応用分野	ML効果（利点／課題）
Zernike展開＋ガウス過程回帰	面内エッチ量分布をZernike多項式で展開し、得られた複数の係数をベクトルとして扱う。プロセス条件やロット内高さを入力とし、カーネル関数で入力空間の距離を定義したGPRを用いて係数ベクトルを同時に回帰する。共分散行列を通じて係数間の相関もモデル化できる（多変量GPあるいは共クラギング）。	プロセス条件変数、ロット高さ、Zernike係数や位置座標（複数出力の場合）。	数回のロットから得たZernike係数ベクトルと条件／高さ。カーネルのハイパーパラメータは対数周辺尤度最大化で推定 ￼。	新条件や新高さにおけるZernike係数の予測平均と分散；逆展開による面内分布。信頼区間から再現性の高低を評価できる ￼。	バーチャルメトロロジー、Run‑to‑Run制御、光学系波面補正、レーザ最適化。	利点: 非線形関係を少量データで学習し、予測と同時に信頼区間を提供できる ￼。複数のZernike係数を同時に扱うことで目的変数の共分散を考慮し、線形回帰より精度が高いことが報告されている ￼。Confidence interval を用いた制御では予測信頼度に応じた重み付けが可能となり、Run‑to‑Run制御性能が向上した ￼。課題: 計算量がO(N³)と大きくデータ数が増えると学習が困難 ￼。カーネル選択や多次元出力の共分散構造の設計が必要で、モデル解釈性も低い。高次元入力では効率が低下するため、先に特徴量削減（PCA／Zernike低次成分）を行うとよい。

線形回帰との比較

線形回帰は入力変数と出力係数の線形関係を仮定し、パラメータ数が少なく計算が軽量であるため実装が容易である。Zernike係数ごとに独立な線形モデルを作る場合、ロット数が少ない状況でも安定した推定が可能で、係数とプロセス条件の相関を直接解釈しやすい。しかし非線形な関係や係数間の相関を捉えられないため、複雑な面内パターンや外れ値に対する柔軟性が低い。これに対してGPRはカーネルにより非線形性と相関を表現し、観測が少ない領域での推定不確実性を可視化できる点が大きなメリットである ￼ ￼。一方で計算コストやハイパーパラメータ選定が課題となり、大規模データにはスパース近似が必要になる ￼。

まとめ

Zernike展開による特徴量化に加え、ガウス過程回帰を用いて係数の相関と非線形性をモデル化することは、少ない学習データでの予測精度向上と不確実性評価に有効である。プラズマエッチやCMPプロセスのバーチャルメトロロジーでは、GPRがニューラルネットワークより精度が高く信頼区間も提供できることが報告されている ￼。Run‑to‑Run制御においては、予測の信頼度に応じて制御ゲインを調整する仕組みが提案され、制御性能向上が示された ￼。一方でGPRは計算負荷やカーネル選定の課題があり、多次元出力を扱う場合には共分散構造の設計が重要である。線形回帰と組み合わせて、低次のZernike係数には線形モデル、高次の非線形パターンにはGPRやRBFなどを併用するといったハイブリッドモデルも検討できる。 ￼ ￼

