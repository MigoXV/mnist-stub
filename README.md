# MNIST 训练与推理基准夹具

这个项目用一个小型 CNN 验证深度学习平台的完整链路：数据准备、监督训练、Checkpoint、恢复训练、预训练与微调、模型导出、独立 HTTP 推理，以及手写交互和中间层展示。它是功能回归夹具，不是追求极限准确率或吞吐量的模型。

## 架构边界

单仓库、共享 Python 依赖，通过不同命令启动功能：

```text
训练入口 → training（数据、优化、Checkpoint） → 导出资产
                      ↓                         ↓
              models / assets / common ← inference → HTTP API → React
```

训练和推理只共享 CNN 定义、资产契约与基础工具，二者业务模块不互相导入。推理不需要训练目录、Checkpoint、MNIST 数据集或 TorchVision 导入；独立进程测试会拦截这些导入。模型资产仍需要此仓库的共享模型实现，并非任意运行时可执行的 ONNX 文件。

目录职责：

- `src/mnist_stub/models`、`assets.py`：固定 CNN、资产读写与校验。
- `src/mnist_stub/training`：数据、训练、评估和恢复；`configs` 定义类型化配置。
- `src/mnist_stub/inference`：图片适配、Runtime、单模型有界执行队列。
- `src/mnist_stub/web`：FastAPI；`src/web`：React + TypeScript + Vite。
- `examples`：标准训练、预训练、微调配置；`tests`：离线自动测试。
- `data-bin` 缓存数据；`model-bin` 存放部署资产；`outputs` 存放运行产物；`tmp-workspace` 用于临时工作。以上目录按需创建并被 Git 忽略。

不引入 Lightning、Transformers、数据库、实验管理服务、容器编排或模型仓库。

## 环境安装

需要 Python 3.10（支持 3.10–3.12）、Poetry、Node.js 22.12+ 或 24、pnpm。Python 命令均通过 Poetry 执行；前端标准入口是后端托管的构建产物。

```bash
poetry env use python3.10
poetry install

# CPU 基线：固定兼容的 PyTorch / TorchVision 组合
poetry run pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cpu

pnpm --dir src/web install --frozen-lockfile
pnpm --dir src/web run build
```

CUDA 环境将 Torch 安装命令替换为：

```bash
poetry run pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
poetry run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CUDA 12.8 wheel 需要兼容的 NVIDIA 驱动。CPU、CUDA 请使用各自的独立虚拟环境；已经安装另一种构建时，pip 不一定会自动切换同版本 wheel。Torch 生态不进入 Poetry 锁文件，不在业务代码或测试中自动安装。其余 Python 依赖由 `poetry.lock` 锁定，前端由 `pnpm-lock.yaml` 锁定。

## 快速跑通

```bash
# 唯一会下载数据的入口；后续可离线运行
poetry run mnist prepare-data

# 训练 3 个 Epoch；末行 stdout 是含 run_dir/model/checkpoint 的 JSON
poetry run mnist train --config examples/train.yaml

# 将下面的 RUN_ID 替换为上一步返回的真实运行目录名
poetry run mnist export outputs/runs/RUN_ID/checkpoints/last.pt model-bin/mnist

# 同时提供 HTTP API 和网页
poetry run mnist serve --model model-bin/mnist
```

浏览器访问 <http://127.0.0.1:8000>。在画布写一个数字或上传图片，即可查看十类概率、28×28 实际输入、卷积/池化特征图和 64 维分类前特征。支持触摸、通道翻页及中间结果开关。

部署不带网页时使用 `--api-only`。默认 CPU、FP32、eager、单进程单模型；CUDA 显式传 `--device cuda:0`，不可用时直接报错，不静默回退。默认监听 `127.0.0.1`；平台可显式传 `--host 0.0.0.0 --port 8000`。

训练完成时已在运行目录的 `model/` 下自动导出最佳验证模型，也可直接用于 `serve --model`。单独 `export` 用于将资产放入平台约定的部署路径。

## 训练场景与恢复语义

模型固定为两组 Conv/ReLU/MaxPool，卷积通道 16、32，全连接 64→10。默认 Adam、学习率 0.001、batch size 64、seed 42、2 个 CPU 计算线程、0 个 DataLoader worker；无随机增强、Dropout、混合精度或学习率调度器。

官方训练集按类别固定划分为 55,000 个训练样本与 5,000 个验证样本。训练集再分为不重叠的 A/B 子集，各约一半且均包含十类标签。官方 10,000 个测试样本只用于每次训练执行结束后的评估，不用于选取最佳模型。

```bash
# A 子集预训练
poetry run mnist train --config examples/pretrain.yaml

# B 子集微调：加载上一步资产；先冻结卷积骨干 1 个 Epoch，再解冻 1 个 Epoch
poetry run mnist train --config examples/finetune.yaml \
  --init-model outputs/runs/PRETRAIN_RUN/model

# 恢复至总共 5 个 Epoch，而不是再训练 5 个
poetry run mnist train --resume outputs/runs/RUN_ID/checkpoints/last.pt --epochs 5

# 官方测试集评估导出资产
poetry run mnist evaluate --model model-bin/mnist
```

`--resume` 与 `--init-model` 互斥：

| 路径 | 权重 | 优化器、计数、随机状态 | 数据/训练策略 |
| --- | --- | --- | --- |
| resume | 完整恢复 | 完整恢复 | 必须与原 Checkpoint 一致 |
| init-model | 从资产或 Checkpoint 加载 | 重新初始化 | 允许新划分、学习率和冻结策略 |

恢复支持 Epoch 边界，不保存未完成 Epoch 的 batch 游标。SIGINT/SIGTERM 会标记运行中断；硬杀无法更新状态文件，但原子保存的最近 Checkpoint 仍可使用。

恢复默认沿用 Checkpoint 配置，仅允许改变目标 Epoch 总数、输出目录和数据路径；数据内容及划分必须一致。每次恢复产生新的运行目录，记录父运行，不覆盖原目录。解冻时保留分类头 Adam 状态，并将卷积参数加入新的参数组。最佳 Checkpoint 与最后 Checkpoint 均保存完整训练状态，最后 Checkpoint 内嵌最佳快照，单文件迁移后仍能保留最佳结果。

精确恢复验证针对相同软件、设备及配置；不承诺跨 PyTorch 版本或 CPU/GPU 的逐位一致。跨设备部署资产支持独立加载，并用数值容差验证输出。

配置优先级是默认值 → YAML → 显式 CLI 参数；未知字段、类型错误和非法数值会失败。常用环境变量为 `MNIST_DATA_DIR`、`MNIST_OUTPUT_DIR`、`MNIST_MODEL`、`MNIST_DEVICE`、`MNIST_HOST`、`MNIST_PORT`。CLI 不隐式读取 `.env`；VS Code 使用 `.env.example` 对应配置及 `envFile` 加载。需要改变 batch size 等训练参数时编辑 YAML。

## 产物和资产契约

```text
outputs/runs/<run-id>/
  config.json          # 解析后的配置
  environment.json     # Python / Torch / NumPy / CUDA / 系统 / Git revision
  data.json            # 原始数据指纹和实际划分索引
  train.log            # 配置、冻结状态、Epoch 指标
  metrics.jsonl        # step Loss/LR，Epoch Loss/Accuracy/参数数量
  status.json          # running/completed/interrupted/failed，父运行与输出
  checkpoints/
    last.pt            # 最近完整 Epoch，可恢复
    best.pt            # 最佳验证 Epoch，可恢复
  evaluation.json      # 最佳模型的测试结果
  model/
    manifest.json
    weights.pt
```

部署资产格式版本为 1：

- `manifest.json`：架构 `mnist-cnn-v1`、标签顺序 0–9、输入形状、预处理版本、均值 0.1307 / 标准差 0.3081、中间层形状、来源信息与权重 SHA-256。
- `weights.pt`：CPU FP32 `state_dict`，不包含优化器或可执行模型对象。
- `model_id` 是权重文件的 SHA-256，用于核对加载的精确资产。
- 加载使用 `weights_only=True`，验证格式、架构、预处理契约、权重哈希、key、shape、dtype 和有限数值。
- 导出先写临时目录并重新加载对齐，再发布；目标已存在则拒绝覆盖。结构或预处理变化需要提升版本，不自动猜测兼容性。

`export` 默认从 Checkpoint 的最佳权重导出，`--current` 导出该 Checkpoint 当时权重。推理服务只接受上述资产目录。

## HTTP 与独立推理

```bash
# 离线预测；可选 --features / --polarity dark|light|auto
poetry run mnist predict digit.png --model model-bin/mnist --features

# HTTP 请求体是图片原始字节，不是 multipart/form-data
curl -X POST 'http://127.0.0.1:8000/api/predict?features=true&polarity=auto' \
  -H 'Content-Type: image/png' --data-binary @digit.png
```

接口：

| 路径 | 用途 |
| --- | --- |
| GET /healthz | 进程存活 |
| GET /readyz | 模型加载、预热及执行 worker 就绪 |
| GET /api/model | 资产信息、device、dtype、runner |
| POST /api/predict | 预测和可选中间层结果 |
| GET /api/metrics | 请求计数、排队水位、最近 1,000 次服务请求耗时分位数 |
| GET /docs | OpenAPI 交互文档 |

预测响应包含 `prediction`、`probabilities[10]`、`input[28][28]`、`model_id`、`inference_ms`、`queue_ms` 和 `features`。不开启中间结果时 `features={}`；开启时各层为 `{shape, values}`：

| 层 | 形状（不含 batch） |
| --- | --- |
| conv1 | 16 × 28 × 28 |
| pool1 | 16 × 14 × 14 |
| conv2 | 32 × 14 × 14 |
| pool2 | 32 × 7 × 7 |
| embedding | 64 |

卷积层值为 ReLU 后的激活。网页按每个通道的 min/max 缩放颜色并标注范围；全零通道显示为底色。特征是中间结果，不代表因果解释；概率也不是识别正确性的保证。

上传限制为 PNG/JPEG、2 MiB、400 万像素。服务端处理 EXIF 方向、透明背景、笔迹极性、前景裁剪、20×20 内等比例缩放和 28×28 居中。标准 MNIST 评估直接使用原始像素，只共享归一化，不应用上传图片适配。网页预览的是模型实际接收的像素。

推理仅一个专用执行线程拥有模型；事件循环处理请求，队列最多等待 8 个请求，默认超时 10 秒。错误状态：413 图片过大、415 媒体类型不支持、422 图片/参数错误、503 未就绪或队列满、504 超时。排队请求可取消；已开始的计算不会强行中断，其迟到结果会丢弃。服务正常退出会停止接收并清理排队任务；硬件/原生算子挂死由平台进程监督负责终止。

`inference_ms` 是预处理至输出回传 CPU 的总耗时，不是 CUDA kernel 计时；`queue_ms` 单独记录。指标不等同于专用性能基准，不对不同硬件设固定延迟门槛。

## 自动验收

```bash
poetry run ruff check src/mnist_stub tests
poetry run ruff format --check src/mnist_stub tests
poetry run pytest -q
pnpm --dir src/web run lint
pnpm --dir src/web run build

# 默认合成数据，不联网、不下载数据；通过独立子进程覆盖所有场景
poetry run mnist smoke

# 完整真实 MNIST；需先 prepare-data，准确率门槛 >= 97%
poetry run mnist smoke --real

# GPU：显式设备，CUDA 不可用时报错
poetry run mnist smoke --device cuda:0
```

每次 smoke 在 `outputs/smoke/<id>/` 保存阶段日志与 `report.json`。其中独立服务进程禁止导入训练模块及 TorchVision，从不同工作目录加载资产，验证 HTTP 概率/特征和 SIGTERM 优雅退出、端口释放。成功返回 0，失败返回非零；训练命令日志发往 stderr，最终结果 JSON 发往 stdout，便于平台采集。

自动测试覆盖连续训练与恢复训练的参数/优化器/随机状态逐项对齐、冻结与解冻、数据划分、资产损坏、图片输入、HTTP 路由、队列过载、超时、并发隔离和 CUDA 输出对齐。GPU 测试在无 CUDA 环境明确跳过。

浏览器测试在服务启动后运行：

```bash
# 首次安装浏览器（也可通过 PLAYWRIGHT_CHROMIUM_EXECUTABLE 指定本机 Chromium）
pnpm --dir src/web exec playwright install chromium

# 另一个终端保持 serve 运行
pnpm --dir src/web run test:e2e
# 非默认地址：
MNIST_WEB_URL=http://127.0.0.1:8765 pnpm --dir src/web run test:e2e
```

浏览器测试验证手写、上传、空白错误、概率、特征切换、清空和移动端布局，截图输出至 `src/web/test-results/`。Playwright 仅为开发依赖，不影响服务运行。

VS Code 的 backend 调试会先执行 `web: build`；train 调试直接运行 CLI，无前端构建依赖。开发时可使用 `pnpm --dir src/web run dev` 获取 HMR；标准运行仍先构建、再由后端提供页面。部署时应保留 `src/web/dist`，缺少构建时默认服务启动会明确失败；API-only 模式不需要该目录。

## 平台接入约定

平台只需要提供工作目录、环境、数据缓存和命令参数，并采集退出码、日志与文件。Session、GPU 分配、运行终止、Checkpoint 持久化、模型版本和部署副本由平台管理。本项目不主动调用平台 API。

先用合成 smoke 验证文件和进程链路，再用真实 MNIST smoke 验证训练质量。运行目录和模型资产不要复用为可覆盖的单一全局路径；使用每次运行返回的明确路径，避免误测旧模型。
