import type { Resource } from "../api";
import type { Metadata, Metrics } from "../types";
import { CopyButton, Icon, Notice, SectionTitle } from "./Primitives";

function text(value: unknown) {
  if (value === null || value === undefined) return "未提供";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
function ResourceState<T>({
  resource,
  name,
}: {
  resource: Resource<T>;
  name: string;
}) {
  return (
    <>
      {resource.error && (
        <Notice retry={resource.refresh}>
          {name}：{resource.error}
          {resource.data && <p>下方保留上次成功获取的快照，可能已过期。</p>}
        </Notice>
      )}
      {resource.loading && !resource.data && (
        <p role="status">正在获取{name}…</p>
      )}
    </>
  );
}
export function ModelView({
  model,
  metrics,
}: {
  model: Resource<Metadata>;
  metrics: Resource<Metrics>;
}) {
  const data = model.data;
  const stats = metrics.data;
  const fields: [string, unknown][] = data
    ? [
        ["架构", data.architecture],
        ["资产格式", "v" + data.format_version],
        ["类别", data.labels.join(" · ")],
        ["输入形状", data.preprocessing.shape.join(" × ")],
        ["归一化均值", data.preprocessing.mean],
        ["归一化标准差", data.preprocessing.std],
        ["图片适配", data.preprocessing.image_adapter],
        ["预处理版本", data.preprocessing.version],
        ["来源运行", data.metadata?.run_id],
        ["训练 Epoch", data.metadata?.epoch],
        ["权重选择", data.metadata?.selection],
      ]
    : [];
  const counters = [
    ["requests", "请求总数"],
    ["success", "成功"],
    ["invalid", "输入无效"],
    ["errors", "执行错误"],
    ["rejected", "队列拒绝"],
    ["unavailable", "服务未就绪"],
    ["timeout", "超时"],
    ["cancelled", "取消等待"],
    ["late_discarded", "迟到结果丢弃"],
  ];
  return (
    <div className="model-view">
      <section aria-labelledby="asset-title">
        <div className="detail-title">
          <h2 id="asset-title">当前模型资产</h2>
          <button
            className="quiet-button"
            onClick={model.refresh}
            disabled={model.loading}
          >
            <Icon name="refresh" />
            刷新模型信息
          </button>
        </div>
        <ResourceState resource={model} name="模型信息" />
        {data && (
          <>
            <div className="model-identity">
              <span className="small-label">模型 ID / 权重 SHA-256</span>
              <code>{data.model_id}</code>
              <CopyButton text={data.model_id} />
            </div>
            <dl className="property-list">
              {fields.map(([label, value]) => (
                <div key={label}>
                  <dt>{label}</dt>
                  <dd>{text(value)}</dd>
                </div>
              ))}
            </dl>
            <SectionTitle title="执行环境" />
            <dl className="environment-list">
              <div>
                <dt>设备</dt>
                <dd>{data.device}</dd>
              </div>
              <div>
                <dt>精度</dt>
                <dd>{data.dtype}</dd>
              </div>
              <div>
                <dt>执行方式</dt>
                <dd>{data.runner}</dd>
              </div>
            </dl>
            <SectionTitle title="可观测中间层" />
            <table className="plain-table">
              <thead>
                <tr>
                  <th scope="col">层标识</th>
                  <th scope="col">输出形状</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.features).map(([name, shape]) => (
                  <tr key={name}>
                    <th scope="row" className="mono">
                      {name}
                    </th>
                    <td className="mono">{shape.join(" × ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="field-help">
              模型按服务启动时的资产加载。此视图只读，不在浏览器中替换模型。
            </p>
          </>
        )}
        {model.updatedAt && (
          <p className="update-time">
            最后成功更新 {model.updatedAt.toLocaleTimeString("zh-CN")}
          </p>
        )}
      </section>
      <section aria-labelledby="service-title">
        <div className="detail-title">
          <h2 id="service-title">服务观测</h2>
          <button
            className="quiet-button"
            onClick={metrics.refresh}
            disabled={metrics.loading}
          >
            <Icon name="refresh" />
            刷新服务指标
          </button>
        </div>
        <ResourceState resource={metrics} name="服务指标" />
        {stats && (
          <>
            <div className="service-summary">
              <div>
                <span className="small-label">正在执行</span>
                <strong>{stats.active}</strong>
                <span>请求</span>
              </div>
              <div>
                <span className="small-label">等待队列</span>
                <strong>
                  {stats.queued}
                  <small> / {stats.capacity}</small>
                </strong>
                <span>当前 / 容量</span>
              </div>
              <div>
                <span className="small-label">队列最高水位</span>
                <strong>{stats.queue_high_water}</strong>
                <span>本服务进程内</span>
              </div>
            </div>
            <SectionTitle title="请求统计" detail="当前服务进程累计" />
            <table className="plain-table">
              <thead>
                <tr>
                  <th scope="col">事件</th>
                  <th scope="col">数量</th>
                </tr>
              </thead>
              <tbody>
                {counters.map(([key, label]) => (
                  <tr key={key}>
                    <th scope="row">{label}</th>
                    <td className="mono">{stats.counts[key] ?? 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <SectionTitle
              title="响应耗时"
              detail={"最近 " + stats.latency_window + " 个样本"}
            />
            <dl className="latency-list">
              {["p50", "p95", "p99"].map((key) => (
                <div key={key}>
                  <dt>{key.toUpperCase()}</dt>
                  <dd>
                    {stats.latency_window
                      ? stats.latency_ms[key].toFixed(2)
                      : "暂无样本"}
                    {stats.latency_window > 0 && <small> ms</small>}
                  </dd>
                </div>
              ))}
            </dl>
            <p className="field-help">
              最多保留最近 1,000
              次服务请求的耗时；计数在服务重启后重置。统计不代表独立硬件性能基准。
            </p>
          </>
        )}
        {metrics.updatedAt && (
          <p className="update-time">
            最后成功更新 {metrics.updatedAt.toLocaleTimeString("zh-CN")} · 每 15
            秒自动刷新
          </p>
        )}
      </section>
    </div>
  );
}
