import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import type { PointerEvent } from "react";
import { Icon, Notice, SectionTitle } from "./Primitives";

export type InputHandle = {
  payload: () => Promise<{ blob: Blob; polarity: string }>;
};
type Props = {
  features: boolean;
  setFeatures: (value: boolean) => void;
  onInputChange: () => void;
  busy: boolean;
  ready: boolean;
  error: string;
  message: string;
  onPredict: () => void;
  onCancel: () => void;
  connectionLabel: string;
};
export const InputPanel = forwardRef<InputHandle, Props>(function InputPanel(
  {
    features,
    setFeatures,
    onInputChange,
    busy,
    ready,
    error,
    message,
    onPredict,
    onCancel,
    connectionLabel,
  },
  ref,
) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const drawing = useRef<number | null>(null);
  const [mode, setMode] = useState<"draw" | "upload">("draw");
  const [width, setWidth] = useState(19);
  const [ink, setInk] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [fileInfo, setFileInfo] = useState("");
  const [fileError, setFileError] = useState("");
  const [decoding, setDecoding] = useState(false);
  const [polarity, setPolarity] = useState("auto");

  function clear() {
    drawing.current = null;
    const ctx = canvas.current!.getContext("2d")!;
    ctx.fillStyle = "#30251d";
    ctx.fillRect(0, 0, 280, 280);
    setInk(false);
    setFile(null);
    setFileError("");
    onInputChange();
  }
  useEffect(() => {
    const ctx = canvas.current!.getContext("2d")!;
    ctx.fillStyle = "#30251d";
    ctx.fillRect(0, 0, 280, 280);
  }, []);
  useEffect(() => {
    if (!file) {
      setPreview("");
      setFileInfo("");
      setDecoding(false);
      return;
    }
    let active = true;
    const url = URL.createObjectURL(file);
    const image = new Image();
    setDecoding(true);
    image.onload = () => {
      if (!active) return;
      setDecoding(false);
      if (image.width * image.height > 4000000) {
        setFileError("图片超过 400 万像素，请选择较小的图片。");
        setPreview("");
      } else {
        setPreview(url);
        setFileInfo(
          image.width +
            " × " +
            image.height +
            " px · " +
            (file.size / 1024).toFixed(1) +
            " KB",
        );
      }
    };
    image.onerror = () => {
      if (active) {
        setDecoding(false);
        setPreview("");
        setFileError("图片无法解码，请重新选择 PNG 或 JPEG。");
      }
    };
    image.src = url;
    return () => {
      active = false;
      URL.revokeObjectURL(url);
    };
  }, [file]);
  useImperativeHandle(ref, () => ({
    payload: async () => {
      if (mode === "upload") {
        if (decoding) throw new Error("图片正在解码，请稍后识别。");
        if (fileError) throw new Error(fileError);
        if (!file || !preview)
          throw new Error("请先选择一张 PNG 或 JPEG 图片。");
        return { blob: file, polarity };
      }
      if (!ink)
        throw new Error("画布为空白，请先写下一个数字，或切换到上传图片。");
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.current!.toBlob(resolve, "image/png"),
      );
      if (!blob) throw new Error("无法读取画布，请重新输入。");
      return { blob, polarity: "light" };
    },
  }));
  function point(event: PointerEvent<HTMLCanvasElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    return [
      ((event.clientX - box.left) * 280) / box.width,
      ((event.clientY - box.top) * 280) / box.height,
    ];
  }
  function start(event: PointerEvent<HTMLCanvasElement>) {
    if (
      !event.isPrimary ||
      (event.pointerType === "mouse" && event.button !== 0)
    )
      return;
    onInputChange();
    setInk(true);
    drawing.current = event.pointerId;
    event.currentTarget.setPointerCapture(event.pointerId);
    const [x, y] = point(event);
    const ctx = event.currentTarget.getContext("2d")!;
    ctx.strokeStyle = "#ffffff";
    ctx.fillStyle = "#ffffff";
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.arc(x, y, width / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(x, y);
  }
  function move(event: PointerEvent<HTMLCanvasElement>) {
    if (drawing.current !== event.pointerId) return;
    const [x, y] = point(event);
    const ctx = event.currentTarget.getContext("2d")!;
    ctx.lineTo(x, y);
    ctx.stroke();
  }

  return (
    <section className="input-workspace" aria-labelledby="input-title">
      <div className="input-heading">
        <h2 id="input-title">输入样本</h2>
        <button className="quiet-button" onClick={clear}>
          清空
        </button>
      </div>
      <div className="segmented" role="group" aria-label="输入方式">
        <button
          aria-pressed={mode === "draw"}
          onClick={() => {
            setMode("draw");
            onInputChange();
          }}
        >
          <Icon name="pen" />
          手写数字
        </button>
        <button
          aria-pressed={mode === "upload"}
          onClick={() => {
            setMode("upload");
            onInputChange();
          }}
        >
          <Icon name="upload" />
          上传图片
        </button>
      </div>
      <div hidden={mode !== "draw"}>
        <div className="drawing-surface">
          <canvas
            ref={canvas}
            width="280"
            height="280"
            aria-label="手写数字画布"
            aria-describedby="drawing-help"
            onPointerDown={start}
            onPointerMove={move}
            onPointerUp={() => {
              drawing.current = null;
            }}
            onPointerCancel={() => {
              drawing.current = null;
            }}
            onLostPointerCapture={() => {
              drawing.current = null;
            }}
          />
          {!ink && (
            <div className="drawing-placeholder" aria-hidden="true">
              <span>在这里写一个数字</span>
              <small>0 — 9</small>
            </div>
          )}
        </div>
        <p className="field-help" id="drawing-help">
          支持鼠标、触摸和笔输入。键盘用户可使用上传图片。
        </p>
        <label className="range-label" htmlFor="stroke">
          笔画粗细 <span className="mono">{width} px</span>
        </label>
        <input
          id="stroke"
          type="range"
          min="8"
          max="28"
          value={width}
          onChange={(e) => setWidth(Number(e.target.value))}
        />
      </div>
      <div hidden={mode !== "upload"} className="upload-workspace">
        <div className="upload-surface">
          {preview ? (
            <img src={preview} alt="上传的数字" />
          ) : (
            <div>
              <Icon name="upload" size={30} />
              <p>{decoding ? "正在读取图片…" : "选择一张数字图片"}</p>
              <span>PNG / JPEG · 最大 2 MiB</span>
            </div>
          )}
        </div>
        <input
          className="sr-only"
          tabIndex={-1}
          ref={fileInput}
          type="file"
          accept="image/png,image/jpeg"
          aria-label="数字图片文件"
          onChange={(event) => {
            const selected = event.target.files?.[0];
            event.target.value = "";
            if (!selected) return;
            onInputChange();
            setFileError("");
            setPreview("");
            if (selected.size > 2 * 1024 * 1024) {
              setFile(null);
              setFileError("图片超过 2 MiB，请选择较小的图片。");
              return;
            }
            if (!["image/png", "image/jpeg"].includes(selected.type)) {
              setFile(null);
              setFileError("仅支持 PNG 或 JPEG，请重新选择。");
              return;
            }
            setFile(selected);
          }}
        />
        <button
          className="secondary-button full-width"
          onClick={() => fileInput.current!.click()}
        >
          <Icon name="upload" />
          {file ? "更换图片" : "选择图片"}
        </button>
        {file && (
          <div className="file-description">
            <strong>{file.name}</strong>
            <span>{fileInfo}</span>
          </div>
        )}
        {fileError && <Notice>{fileError}</Notice>}
        <label className="field-row" htmlFor="polarity">
          笔迹颜色
          <select
            id="polarity"
            value={polarity}
            onChange={(event) => {
              setPolarity(event.target.value);
              onInputChange();
            }}
          >
            <option value="auto">自动判断</option>
            <option value="dark">深色笔迹</option>
            <option value="light">浅色笔迹</option>
          </select>
        </label>
      </div>
      <div className="request-options">
        <SectionTitle title="推理选项" />
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={features}
            onChange={(e) => setFeatures(e.target.checked)}
          />
          <span>
            获取中间层特征<small>用于观察卷积、池化与分类前激活</small>
          </span>
        </label>
      </div>
      <div className="run-controls">
        <button
          className="primary-button full-width"
          disabled={!ready || busy || decoding}
          onClick={onPredict}
        >
          <Icon name={busy ? "signal" : "arrow"} />
          {busy ? "等待结果…" : error ? "重新识别" : "识别数字"}
        </button>
        {busy && (
          <button className="quiet-button full-width" onClick={onCancel}>
            取消等待
          </button>
        )}
        {!ready && (
          <p className="field-help">{connectionLabel}，连接就绪后可识别。</p>
        )}
        {error && <Notice>{error}</Notice>}
        <p role="status" aria-live="polite" className="operation-status">
          {message}
        </p>
      </div>
      <div className="input-note">
        <span className="note-number">28 × 28</span>
        <p>
          图片会转换为单通道输入。
          <br />
          识别后可核对模型实际接收的像素。
        </p>
      </div>
    </section>
  );
});
