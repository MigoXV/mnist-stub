import { useEffect, useRef, useState } from "react";
import type { Prediction } from "../types";
import { layers } from "../types";
import { Heatmap, Icon } from "./Primitives";

type Selection = { name: string; channel: number; values: number[][] };
function Inspector({
  selection,
  close,
}: {
  selection: Selection;
  close: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const node = dialog.current!;
    node.showModal();
    return () => {
      node.close();
      previous?.focus();
    };
  }, []);
  const flat = selection.values.flat();
  return (
    <dialog
      className="inspector"
      ref={dialog}
      aria-labelledby="inspector-title"
      onCancel={(event) => {
        event.preventDefault();
        close();
      }}
    >
      <div className="inspector-heading">
        <div>
          <span className="kicker">通道检查器</span>
          <h2 id="inspector-title">
            {selection.name} · 通道 {selection.channel + 1}
          </h2>
        </div>
        <button className="quiet-button" onClick={close} autoFocus>
          <Icon name="close" />
          关闭
        </button>
      </div>
      <div className="inspector-body">
        <Heatmap
          values={selection.values}
          label={selection.name + "通道放大图"}
        />
        <dl className="channel-statistics">
          <div>
            <dt>形状</dt>
            <dd>
              {selection.values.length} × {selection.values[0].length}
            </dd>
          </div>
          <div>
            <dt>最小值</dt>
            <dd>{Math.min(...flat).toFixed(4)}</dd>
          </div>
          <div>
            <dt>最大值</dt>
            <dd>{Math.max(...flat).toFixed(4)}</dd>
          </div>
          <div>
            <dt>均值</dt>
            <dd>
              {(flat.reduce((a, b) => a + b, 0) / flat.length).toFixed(4)}
            </dd>
          </div>
        </dl>
        <div className="scale">
          <span>低</span>
          <i />
          <span>高</span>
        </div>
        <p className="field-help">
          按本通道的实际最小值与最大值缩放颜色。不同通道的颜色不可直接比较绝对大小。
        </p>
        <details>
          <summary>查看数值矩阵</summary>
          <div
            className="matrix-scroll"
            tabIndex={0}
            role="region"
            aria-label="通道数值矩阵"
          >
            <table className="matrix">
              <caption className="sr-only">激活值，按行和列排列</caption>
              <thead>
                <tr>
                  <th scope="col">行 / 列</th>
                  {selection.values[0].map((_, i) => (
                    <th key={i} scope="col">
                      {i + 1}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {selection.values.map((row, i) => (
                  <tr key={i}>
                    <th scope="row">{i + 1}</th>
                    {row.map((value, j) => (
                      <td key={j}>{value.toFixed(3)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </div>
    </dialog>
  );
}

export function Analysis({
  active,
  result,
  busy,
  enableFeatures,
}: {
  active: boolean;
  result: Prediction | null;
  busy: boolean;
  enableFeatures: () => void;
}) {
  const [tab, setTab] = useState("probabilities");
  const [layer, setLayer] = useState<string>("conv1");
  const [page, setPage] = useState(0);
  const [selection, setSelection] = useState<Selection | null>(null);
  useEffect(() => {
    setSelection(null);
    setPage(0);
  }, [result]);
  useEffect(() => {
    if (!active) setSelection(null);
  }, [active]);
  const maps = result?.features[layer]?.values as number[][][] | undefined;
  const embedding = result?.features.embedding?.values as number[] | undefined;

  return (
    <section
      className="analysis-workspace"
      aria-labelledby="analysis-title"
      aria-busy={busy}
    >
      <div className="analysis-heading">
        <h2 id="analysis-title">结果分析</h2>
        <span className="result-label">
          {busy ? "等待本次响应" : result ? "本次识别结果" : "尚未识别"}
        </span>
      </div>
      <div className="result-summary">
        <div className="digit-result">
          <span className="small-label">预测数字</span>
          <strong>{result?.prediction ?? "—"}</strong>
          <span>
            {result
              ? "模型输出 · " +
                (result.probabilities[result.prediction] * 100).toFixed(2) +
                "%"
              : "输入一个数字后开始"}
          </span>
        </div>
        <dl className="run-statistics">
          <div>
            <dt>推理耗时</dt>
            <dd>
              {result ? result.inference_ms.toFixed(1) : "—"}
              <small>ms</small>
            </dd>
          </div>
          <div>
            <dt>排队耗时</dt>
            <dd>
              {result ? result.queue_ms.toFixed(1) : "—"}
              <small>ms</small>
            </dd>
          </div>
        </dl>
        <figure className="actual-input">
          {result ? (
            <Heatmap input values={result.input} label="模型输入预览" />
          ) : (
            <div className="input-placeholder">28 × 28</div>
          )}
          <figcaption>实际模型输入</figcaption>
        </figure>
      </div>
      <div className="analysis-tabs" role="group" aria-label="分析内容">
        <button
          aria-pressed={tab === "probabilities"}
          onClick={() => setTab("probabilities")}
        >
          分类概率<span>10</span>
        </button>
        <button
          aria-pressed={tab === "features"}
          onClick={() => setTab("features")}
        >
          网络特征<span>5 层</span>
        </button>
      </div>
      {tab === "probabilities" ? (
        <div className="probability-analysis">
          <div className="analysis-subheading">
            <h3>类别分布</h3>
            <span>按数字顺序 · Softmax</span>
          </div>
          <table className="probability-table">
            <caption className="sr-only">数字 0 到 9 的预测概率</caption>
            <thead>
              <tr>
                <th scope="col">类别</th>
                <th scope="col">概率分布</th>
                <th scope="col">预测概率</th>
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: 10 }, (_, digit) => (
                <tr
                  key={digit}
                  className={
                    result?.prediction === digit ? "predicted-row" : ""
                  }
                >
                  <th scope="row">
                    <span className="class-digit">{digit}</span>
                    {result?.prediction === digit && (
                      <span className="prediction-marker">预测</span>
                    )}
                  </th>
                  <td>
                    <div className="probability-track" aria-hidden="true">
                      <i
                        style={{
                          width:
                            (result?.probabilities[digit] ?? 0) * 100 + "%",
                        }}
                      />
                    </div>
                  </td>
                  <td className="mono">
                    {result
                      ? (result.probabilities[digit] * 100).toFixed(2) + "%"
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="analysis-note">
            概率用于比较类别，不代表识别一定正确。推理耗时包含预处理、模型执行和结果转换。
          </p>
        </div>
      ) : (
        <div className="feature-analysis">
          <div className="analysis-subheading">
            <h3>逐层观察</h3>
            <span>点击通道查看详细数值</span>
          </div>
          <div className="layer-sequence" role="group" aria-label="网络层">
            {layers.map(([key, label], i) => (
              <button
                key={key}
                aria-pressed={layer === key}
                onClick={() => {
                  setLayer(key);
                  setPage(0);
                }}
              >
                <span className="layer-index">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span>{label}</span>
                <small>
                  {result?.features[key]?.shape.join(" × ") ?? "未获取"}
                </small>
              </button>
            ))}
          </div>
          {!result || !Object.keys(result.features).length ? (
            <div className="analysis-empty">
              <Icon name="model" size={32} />
              <h3>
                {busy
                  ? "等待中间层结果"
                  : result
                    ? "本次未获取中间层特征"
                    : "从一个输入开始观察"}
              </h3>
              <p>
                {result
                  ? "重新识别时获取特征，即可逐层检查网络中的激活。"
                  : "识别成功后，这里会显示卷积、池化与分类前的真实输出。"}
              </p>
              {result && (
                <button className="secondary-button" onClick={enableFeatures}>
                  开启中间结果并重新识别
                </button>
              )}
            </div>
          ) : layer === "embedding" && embedding ? (
            <div className="embedding-analysis">
              <div className="analysis-subheading">
                <h3>分类前特征</h3>
                <span>64 维 · ReLU 后激活</span>
              </div>
              <div
                className="embedding-chart"
                role="img"
                aria-label="64 维分类前激活柱形图，下方提供完整数值表"
              >
                {embedding.map((value, i) => (
                  <i
                    key={i}
                    style={{
                      height:
                        Math.max(
                          0,
                          (value / (Math.max(...embedding) || 1)) * 100,
                        ) + "%",
                    }}
                  />
                ))}
              </div>
              <details>
                <summary>查看全部 64 维数值</summary>
                <div className="embedding-values">
                  <table>
                    <caption className="sr-only">分类前特征值</caption>
                    <thead>
                      <tr>
                        <th scope="col">维度</th>
                        <th scope="col">激活值</th>
                      </tr>
                    </thead>
                    <tbody>
                      {embedding.map((value, i) => (
                        <tr key={i}>
                          <th scope="row">{i + 1}</th>
                          <td className="mono">{value.toFixed(5)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            </div>
          ) : maps ? (
            <>
              <div className="channel-toolbar">
                <span className="mono">
                  {result.features[layer].shape.join(" × ")}
                </span>
                <span>
                  第 {page * 8 + 1}–{Math.min(page * 8 + 8, maps.length)} /{" "}
                  {maps.length} 通道
                </span>
              </div>
              <div className="channel-grid">
                {maps.slice(page * 8, page * 8 + 8).map((values, index) => (
                  <button
                    className="channel-button"
                    key={index}
                    onClick={() =>
                      setSelection({
                        name: layers.find(([key]) => key === layer)![1],
                        channel: page * 8 + index,
                        values,
                      })
                    }
                    aria-label={"检查通道 " + (page * 8 + index + 1)}
                  >
                    <Heatmap
                      values={values}
                      label={"通道 " + (page * 8 + index + 1) + " 特征图"}
                    />
                    <span>
                      <strong>
                        通道 {String(page * 8 + index + 1).padStart(2, "0")}
                      </strong>
                      <Icon name="arrow" size={14} />
                    </span>
                    <small className="mono">
                      {Math.min(...values.flat()).toFixed(2)} —{" "}
                      {Math.max(...values.flat()).toFixed(2)}
                    </small>
                  </button>
                ))}
              </div>
              <div className="channel-pagination">
                <span>各通道独立缩放色阶</span>
                <div>
                  <button
                    className="quiet-button"
                    disabled={page === 0}
                    onClick={() => setPage(page - 1)}
                  >
                    上一组
                  </button>
                  <span className="mono">
                    {page + 1} / {Math.ceil(maps.length / 8)}
                  </span>
                  <button
                    className="quiet-button"
                    disabled={(page + 1) * 8 >= maps.length}
                    onClick={() => setPage(page + 1)}
                  >
                    下一组
                  </button>
                </div>
              </div>
            </>
          ) : (
            <p>该层未返回数据，请重新识别。</p>
          )}
          <p className="analysis-note">
            显示的是实际中间激活，用于观察与验证，不构成预测的因果解释。
          </p>
        </div>
      )}
      <div className="result-provenance">
        {result ? (
          <>
            结果模型 <code>{result.model_id}</code>
          </>
        ) : (
          "结果仅对应本次输入；修改输入后需要重新识别。"
        )}
      </div>
      {selection && active && (
        <Inspector selection={selection} close={() => setSelection(null)} />
      )}
    </section>
  );
}
