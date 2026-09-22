# MNIST 训练与独立推理夹具

使用 datasets、Lightning、Transformers 和 tqdm，覆盖训练、断点恢复、预训练／微调、模型导出及独立 HTTP 推理。默认使用本地数据，不自动下载或安装依赖。

## 环境

Python 3.10–3.12，推荐 Python 3.10；Python 命令通过 Poetry 执行。网页使用 React、TypeScript、Vite、pnpm，构建后由 FastAPI 托管。

```bash
poetry env use python3.10
poetry install

# Torch 及训练生态由 pip 安装，不进入 Poetry 主依赖和锁文件。
# 已安装时无需重复执行；本次验证版本如下。
poetry run pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cpu
poetry run pip install datasets==5.0.1 lightning==2.6.6

pnpm --dir src/web install --frozen-lockfile
pnpm --dir src/web run build
```

CUDA 环境使用匹配驱动的 wheel，例如将 Torch 安装源改为
`https://download.pytorch.org/whl/cu128`。CPU 和 CUDA 构建建议使用独立环境。
Transformers、tqdm、jsonargparse[signatures] 等普通依赖由 Poetry 管理。业务代码和测试不会自动安装包。

## 本地数据与训练

默认数据目录为 `data-bin/MigoXV/mnist-4`：

```text
data-bin/MigoXV/mnist-4/
  data/train-*.parquet
  data/test-*.parquet
```

每条样本包含 `image`（28×28 uint8 灰度图）和 `label`（0–9 整数）。
`id`、`label_name` 可保留，不传入模型。通过 datasets 本地 Parquet loader 加载，
预处理结果默认缓存到数据集目录下的 `.mnist-cache`；可用 `data.cache_dir` 指定缓存目录。

训练前校验字段、标签、类别数量和选定样本的图像格式。默认从 60,000 条训练数据中，
按类别和 seed 划分 5,000 条验证数据，其余 55,000 条再划分互斥 A/B 子集。
官方 10,000 条测试数据只用于训练结束后的评估，不参与最佳模型选择。
记录源文件内容指纹和实际划分索引，恢复时验证一致。

```bash
poetry run python -m mnist_stub.commands.train fit --config examples/train.yaml

# 分组配置覆盖；显式 CLI 选项优先级最高
poetry run python -m mnist_stub.commands.train fit --config examples/train.yaml \
  --data.batch_size=128 --optimizer.lr=0.0005 --trainer.max_epochs=5

# 小样本真实数据验证
poetry run python -m mnist_stub.commands.train fit --trainer.max_epochs=1 --data.train_limit=256 \
  --data.validation_limit=64 --data.test_limit=64

# 完全离线的合成数据验证
poetry run python -m mnist_stub.commands.train fit --data.synthetic=true --trainer.max_epochs=1
```

默认 CPU、FP32、单设备、AdamW（lr=0.001、weight_decay=0.01）、batch size 64、
seed 42、2 个计算线程、0 个 DataLoader worker。CUDA 必须显式指定
`--trainer.accelerator=gpu --trainer.devices='[0]'`，不可用时报错，不静默回退。当前不提供随机增强、混合精度或调度器。

训练专用入口为 `src/mnist_stub/commands/train.py`，使用 LightningCLI，
可通过 `python -m mnist_stub.commands.train` 或直接执行该 Python 文件启动。
Typer 的 `mnist` 命令仅保留导出、评估、推理、服务和 smoke，不再提供 train 子命令。

配置由 LightningCLI/jsonargparse 解析，使用原生 `fit --config ... --group.field=value`，
配置文件放在覆盖参数之前。Trainer 使用 `max_epochs`、`accelerator`、`devices`、
`default_root_dir` 等原生字段。模型、数据、optimizer、checkpoint、logging、
evaluation 分组分别配置；seed 使用顶层 `seed_everything`，CPU 线程数使用 `threads`。
不再使用旧 dotlist 裸参数或训练 YAML 的 `${oc.env:...}` 语法。

使用 `fit --help` 查看参数，`fit --print_config` 输出配置而不启动训练。
启用 LightningCLI 原生环境变量，例如 `MNIST_FIT__DATA__PATH`、
`MNIST_FIT__TRAINER__MAX_EPOCHS`；未知字段、类型错误和非有限数值会失败。
每次运行保存可再次作为 `--config` 输入的完整 `config.yaml`。

## tqdm 与日志

交互终端默认显示数据预处理、训练、验证和测试进度条，输出重定向时默认关闭。
使用 `--logging.progress=true/false` 显式控制；
`logging.refresh_rate` 控制训练进度刷新间隔。

进度条和终端日志均写 stderr，logging 通过 tqdm 的日志重定向协调刷新；
文件日志独立写入，不含进度条控制字符。逐步指标由 Lightning CSVLogger 写入，
不会逐 batch 打印普通日志。stdout 仅输出最终 JSON，便于平台直接解析。
`logging.enabled=false` 关闭实验指标文件，仍保留运行审计日志。

## 预训练、微调与恢复

```bash
poetry run python -m mnist_stub.commands.train fit --config examples/pretrain.yaml
poetry run python -m mnist_stub.commands.train fit --config examples/finetune.yaml \
  --init_model outputs/runs/PRETRAIN_RUN/model

# epochs 是目标总轮数；不是额外训练轮数
poetry run python -m mnist_stub.commands.train fit \
  --config outputs/runs/RUN_ID/config.yaml \
  --ckpt_path outputs/runs/RUN_ID/checkpoints/last.ckpt --trainer.max_epochs=5
```

预训练使用 A 子集；微调使用 B 子集，默认冻结卷积层一轮后解冻。
冻结期间优化器只包含分类头；解冻时追加卷积参数组，保留分类头的优化器状态。

`--ckpt_path` 恢复 Lightning 原生训练状态、随机状态、DataLoader 状态及最佳模型快照。
仅支持完整 epoch 边界，不保存中途 batch 游标。每次恢复新建运行目录并记录父运行，
不覆盖父目录。允许改变目标总轮数、输出目录、数据／缓存路径和日志／最终评估设置；
数据内容、划分与训练策略必须一致。相同环境下测试连续训练与恢复训练的状态逐项对齐，
不承诺跨软件版本或设备逐位一致。

`--init_model` 从新模型目录或新的 `.ckpt` 初始化权重；`model.model_path` 用于本地
Transformers 模型目录。初始化会重新建立优化器和计数，不能与恢复同时使用。
恢复微调时若保存的配置含 `init_model`，显式传 `--init_model=null`。
旧手写 `.pt` checkpoint、旧资产和旧配置均不兼容。

`last.ckpt` 每个完整 epoch 更新，即使验证指标没有提升。
默认按 `val_accuracy` 保存最佳 checkpoint，也可配置 `val_loss`；
`checkpoint.save_top_k` 控制保留数量。最后 checkpoint 内保存历史最佳权重，
单文件迁移后仍可继续选择或导出最佳模型。SIGINT/SIGTERM 记录中断状态，
最近完整 epoch 的 checkpoint 可继续使用。

## 产物与模型导出

```text
outputs/runs/<run-id>/
  config.yaml          # LightningCLI 原生配置，可用于重跑或恢复
  config.json          # 内部训练审计配置
  environment.json
  data.json
  train.log
  metrics/hparams.yaml
  metrics/metrics.csv
  status.json
  checkpoints/last.ckpt
  checkpoints/epoch=...-step=...-val_accuracy=....ckpt
  evaluation.json
  model/
    config.json
    preprocessor_config.json
    model.safetensors
    manifest.json
```

训练结束默认用最佳权重评估测试集并自动导出模型。
`--evaluation.test_after_fit=false` 可关闭最终测试；验证与最佳模型选择仍会执行。

```bash
poetry run mnist export outputs/runs/RUN_ID/checkpoints/last.ckpt model-bin/mnist
poetry run mnist evaluate --model model-bin/mnist
poetry run mnist serve --model model-bin/mnist
```

export 默认导出 checkpoint 内的最佳权重，`--current` 导出该 checkpoint 的当前权重。
目标目录已存在时拒绝覆盖。模型目录采用 Transformers 本地格式：
导入项目模型注册后可通过 AutoConfig、AutoModel、AutoModelForImageClassification、
AutoProcessor 加载，均可离线运行。推理无需 Lightning、datasets 或任务模块。

部署 manifest 格式版本为 2，记录十类标签、预处理、特征形状、来源和文件哈希。
加载时校验架构、文件 SHA-256、权重 key／shape／dtype／有限数值。
导出先写临时目录，重新加载并对齐输出后发布。CNN 保持两组 Conv/ReLU/MaxPool，
通道为 16、32，分类前特征为 64 维，输出 10 类。训练和推理共用 Processor，
像素先除以 255，再按均值 0.1307、标准差 0.3081 归一化。

## 网页与 HTTP 推理

默认访问 <http://127.0.0.1:8000>，支持画布输入、上传图片、十类概率、
实际 28×28 输入及卷积／池化／分类前特征。界面采用 MANAS 工作表面，
设计约定见 [界面实施约定](docs/ui-design.md)。

```bash
poetry run mnist predict digit.png --model model-bin/mnist --features
poetry run mnist serve --model model-bin/mnist --api-only

curl -X POST 'http://127.0.0.1:8000/api/predict?features=true&polarity=auto' \
  -H 'Content-Type: image/png' --data-binary @digit.png
```

| 接口 | 用途 |
| --- | --- |
| GET /healthz | 进程存活 |
| GET /readyz | 模型与执行 worker 就绪 |
| GET /api/model | 模型资产、device、dtype、runner |
| POST /api/predict | 原始图片字节预测，可选 features、polarity |
| GET /api/metrics | 请求计数、队列水位及延迟统计 |
| GET /docs | OpenAPI 文档 |

上传限 PNG/JPEG、2 MiB、400 万像素；支持 EXIF、透明背景与极性处理，
前景缩放至 20×20 后放入 28×28。标准数据集评估直接使用原始 MNIST 像素，
不执行上传图片适配。默认 CPU、FP32、eager、单进程单模型，
由专用线程执行，有界队列最多等待 8 个请求，默认超时 10 秒。
错误状态包括 413、415、422、503、504。取消等待不会强行中止已开始的设备计算。

预测返回 prediction、probabilities、input、model_id、inference_ms、queue_ms、features；
可选特征为 conv1（16×28×28）、pool1（16×14×14）、conv2（32×14×14）、
pool2（32×7×7）、embedding（64）。队列、生命周期和网页接口保持原有行为。

## 自动验证与开发

```bash
poetry run ruff check src/mnist_stub tests
poetry run ruff format --check src/mnist_stub tests
poetry run pytest -q
poetry run mnist smoke
poetry run mnist smoke --real
poetry run mnist smoke --device cuda:0
```

smoke 使用独立进程覆盖训练、恢复、A/B 微调、导出、HTTP 推理与优雅关闭；
真实数据模式要求测试准确率 ≥97%。每次运行记录独立 report.json 和阶段日志。
普通测试使用合成数据或临时 Parquet，不联网、不依赖远程模型。
进度条测试使用伪终端，并检查 stdout JSON、日志重复、异常清理及重定向行为。

```bash
pnpm --dir src/web run lint
pnpm --dir src/web run build
# 服务启动后执行浏览器测试
pnpm --dir src/web run test:e2e
```

目录职责：commands 装配入口，configs 定义 schema，models 定义与注册模型及 Processor，
tasks 实现 DataModule／LightningModule 与运行编排，criterions 定义 loss，
checkpointing 保存训练状态，exporting 导出交付模型，inference 和 web 提供独立服务。
旧 training 目录已移除。

推理环境变量见 `.env.example`；训练使用上文的 LightningCLI 原生环境变量。CLI 不隐式读取 .env；VS Code 配置使用 envFile，
训练调试直接运行 CLI，后端调试先构建网页。data-bin、model-bin、outputs、
tmp-workspace 存放本地产物且被 Git 忽略。

平台提供工作目录、运行环境和命令参数，并采集 stdout JSON、退出码与文件。
本项目不主动调用平台 API；设备分配、运行终止、产物持久化与部署由平台管理。
