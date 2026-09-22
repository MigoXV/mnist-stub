import { useEffect, useRef, useState } from "react";
import { predictImage, useResource } from "./api";
import { Analysis } from "./components/Analysis";
import { InputPanel } from "./components/InputPanel";
import type { InputHandle } from "./components/InputPanel";
import { ModelView } from "./components/ModelView";
import { Icon } from "./components/Primitives";
import type { Metadata, Metrics, Prediction } from "./types";

export default function App() {
  const [view, setView] = useState(() =>
    location.hash === "#/model" ? "model" : "inference",
  );
  const [mobileTask, setMobileTask] = useState("input");
  const [features, setFeatures] = useState(true);
  const [result, setResult] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const input = useRef<InputHandle>(null);
  const request = useRef<AbortController | null>(null);
  const analysisTab = useRef<HTMLButtonElement>(null);
  const health = useResource<{ ready: boolean }>("/readyz", 10000);
  const model = useResource<Metadata>("/api/model");
  const metrics = useResource<Metrics>("/api/metrics");
  const ready = health.data?.ready === true && !health.error;
  const connectionLabel = health.error
    ? "连接中断"
    : health.data
      ? health.data.ready
        ? "推理服务就绪"
        : "服务未就绪"
      : "正在连接";
  useEffect(() => {
    function navigate() {
      if (location.hash === "#/model") setView("model");
      else if (location.hash === "#/inference" || !location.hash)
        setView("inference");
    }
    window.addEventListener("hashchange", navigate);
    return () => window.removeEventListener("hashchange", navigate);
  }, []);
  useEffect(
    () => () => {
      request.current?.abort();
    },
    [],
  );

  function invalidate() {
    request.current?.abort();
    request.current = null;
    setBusy(false);
    setResult(null);
    setError("");
    setMessage("");
  }
  function cancel() {
    request.current?.abort();
    request.current = null;
    setBusy(false);
    setMessage("已取消等待。服务端已开始的计算可能仍会继续。");
  }
  async function predict(withFeatures = features) {
    invalidate();
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    const timer = window.setTimeout(() => {
      if (request.current === controller) {
        controller.abort();
        request.current = null;
        setBusy(false);
        setError("连接等待超过 20 秒，请检查服务状态后重试。");
      }
    }, 20000);
    try {
      const payload = await input.current!.payload();
      if (request.current !== controller) return;
      const output = (await predictImage(
        payload.blob,
        withFeatures,
        payload.polarity,
        controller.signal,
      )) as Prediction;
      if (request.current !== controller) return;
      setResult(output);
      setMessage("识别完成，预测数字为 " + output.prediction + "。");
      setMobileTask("analysis");
      if (window.matchMedia("(max-width: 767px)").matches) {
        requestAnimationFrame(() => analysisTab.current?.focus());
      }
    } catch (failure) {
      if (request.current === controller) {
        setError(
          failure instanceof TypeError
            ? "无法连接推理服务，请重新连接后重试。"
            : failure instanceof Error
              ? failure.message
              : "识别失败，请重试。",
        );
      }
    } finally {
      window.clearTimeout(timer);
      if (request.current === controller) {
        request.current = null;
        setBusy(false);
      }
    }
  }

  return (
    <div className="application">
      <a className="skip-link" href="#main-content">
        跳转到主要内容
      </a>
      <header className="app-header">
        <a
          className="brand"
          href="#/inference"
          aria-label="MNIST 模型观察室首页"
        >
          <span className="brand-symbol">
            <Icon name="model" size={23} />
          </span>
          <span>
            MNIST<span className="brand-subtitle">模型观察室</span>
          </span>
        </a>
        <nav aria-label="主要导航">
          <a
            href="#/inference"
            aria-current={view === "inference" ? "page" : undefined}
          >
            <Icon name="experiment" />
            推理实验
          </a>
          <a
            href="#/model"
            aria-current={view === "model" ? "page" : undefined}
          >
            <Icon name="model" />
            模型与服务
          </a>
        </nav>
        <div
          className={
            "connection-state " + (ready ? "is-ready" : "is-unavailable")
          }
          role="status"
        >
          <i />
          {connectionLabel}
        </div>
      </header>
      <main id="main-content" tabIndex={-1}>
        <div className="page-heading">
          <div>
            <p className="breadcrumb">
              MNIST / {view === "inference" ? "推理实验" : "模型与服务"}
            </p>
            <h1>{view === "inference" ? "推理工作台" : "模型与服务"}</h1>
            <p>
              {view === "inference"
                ? "从手写输入到网络内部，检查每一次真实预测。"
                : "查看当前加载的模型资产与服务运行信息。"}
            </p>
          </div>
          <div className="context-details">
            <span className="context-label">当前模型</span>
            <a href="#/model" className="model-context">
              <Icon name="model" size={16} />
              <code>
                {model.data?.architecture ??
                  (model.error ? "模型信息不可用" : "正在获取模型")}
              </code>
            </a>
            <span className="context-runtime">
              {model.data
                ? model.data.device.toUpperCase() +
                  " · " +
                  model.data.dtype +
                  " · " +
                  model.data.runner
                : "等待模型信息"}
            </span>
            {model.data && (
              <span className="context-id">
                ID <code>{model.data.model_id.slice(0, 12)}</code>
                {model.error && " · 上次快照"}
              </span>
            )}
          </div>
        </div>
        {!ready && (
          <div className="connection-banner">
            <div>
              <strong>{connectionLabel}</strong>
              <span>
                {health.error || "模型尚未就绪，输入会保留。请稍后重新检查。"}
              </span>
            </div>
            <button
              className="secondary-button"
              onClick={health.refresh}
              disabled={health.loading}
            >
              <Icon name="refresh" />
              重新连接
            </button>
          </div>
        )}
        {model.error && view === "inference" && (
          <div className="metadata-warning">
            <span>模型信息暂不可用；服务就绪时仍可识别。</span>
            <button className="quiet-button" onClick={model.refresh}>
              重试模型信息
            </button>
          </div>
        )}
        <div hidden={view !== "inference"}>
          <div
            className="mobile-task-switch"
            role="group"
            aria-label="当前任务"
          >
            <button
              aria-pressed={mobileTask === "input"}
              onClick={() => setMobileTask("input")}
            >
              输入
            </button>
            <button
              ref={analysisTab}
              aria-pressed={mobileTask === "analysis"}
              onClick={() => setMobileTask("analysis")}
            >
              分析
            </button>
          </div>
          <div className="experiment-workspace" data-mobile-task={mobileTask}>
            <InputPanel
              ref={input}
              features={features}
              setFeatures={(value) => {
                invalidate();
                setFeatures(value);
              }}
              onInputChange={invalidate}
              busy={busy}
              ready={ready}
              error={error}
              message={message}
              connectionLabel={connectionLabel}
              onPredict={() => {
                void predict();
              }}
              onCancel={cancel}
            />
            <Analysis
              active={view === "inference"}
              result={result}
              busy={busy}
              enableFeatures={() => {
                setFeatures(true);
                void predict(true);
              }}
            />
          </div>
          <div className="workspace-footnote">
            <span>
              <Icon name="check" size={14} />
              输入与推理结果在当前页面内使用
            </span>
            <span>MNIST · 10 类数字识别</span>
          </div>
        </div>
        <div hidden={view !== "model"}>
          <ModelView model={model} metrics={metrics} />
        </div>
      </main>
    </div>
  );
}
