import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

export function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const paths: Record<string, ReactNode> = {
    experiment: (
      <>
        <rect x="4" y="4" width="6" height="6" rx="1" />
        <rect x="14" y="4" width="6" height="6" rx="1" />
        <rect x="4" y="14" width="6" height="6" rx="1" />
        <path d="M14 17h6m-3-3v6" />
      </>
    ),
    model: (
      <>
        <path d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5" />
      </>
    ),
    arrow: <path d="M4 12h15m-6-6 6 6-6 6" />,
    upload: (
      <>
        <path d="M12 16V3m-5 5 5-5 5 5M4 15v5h16v-5" />
      </>
    ),
    pen: (
      <>
        <path d="m5 15 10-10 4 4L9 19l-5 1 1-5Zm8-8 4 4" />
      </>
    ),
    refresh: (
      <>
        <path d="M20 10a8 8 0 0 0-14-5L3 8m0-5v5h5M4 14a8 8 0 0 0 14 5l3-3m0 5v-5h-5" />
      </>
    ),
    close: <path d="m6 6 12 12M6 18 18 6" />,
    copy: (
      <>
        <rect x="8" y="8" width="12" height="13" rx="2" />
        <path d="M16 8V3H3v13h5" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    signal: (
      <>
        <path d="M4 17v3m5-8v8m6-13v13m5-18v18" />
      </>
    ),
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name] ?? paths.model}
    </svg>
  );
}

export function CopyButton({ text }: { text: string }) {
  const [message, setMessage] = useState("");
  useEffect(() => {
    if (!message) return;
    const timer = window.setTimeout(() => setMessage(""), 2500);
    return () => window.clearTimeout(timer);
  }, [message]);
  return (
    <span className="copy-control">
      <button
        type="button"
        className="quiet-button"
        onClick={() => {
          void navigator.clipboard?.writeText(text).then(
            () => setMessage("已复制"),
            () => setMessage("复制失败，请选择文字复制"),
          );
          if (!navigator.clipboard) setMessage("请选择文字复制");
        }}
        aria-label="复制模型 ID"
      >
        <Icon name="copy" />
        复制
      </button>
      <span role="status">{message}</span>
    </span>
  );
}

export function Heatmap({
  values,
  input = false,
  label,
}: {
  values: number[][];
  input?: boolean;
  label: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !values.length) return;
    canvas.width = values[0].length;
    canvas.height = values.length;
    const context = canvas.getContext("2d")!;
    const data = values.flat();
    const min = input ? 0 : Math.min(...data);
    const max = input ? 255 : Math.max(...data);
    const image = context.createImageData(canvas.width, canvas.height);
    data.forEach((value, index) => {
      const t = max === min ? 0 : (value - min) / (max - min);
      image.data.set(
        input
          ? [255 * t, 255 * t, 255 * t, 255]
          : [51 + 204 * t, 36 + 198 * t, 24 + 181 * t, 255],
        index * 4,
      );
    });
    context.putImageData(image, 0, 0);
  }, [values, input]);
  return <canvas className="heatmap" ref={ref} role="img" aria-label={label} />;
}

export function Notice({
  children,
  retry,
}: {
  children: ReactNode;
  retry?: () => void;
}) {
  return (
    <div className="notice" role="alert">
      <div>{children}</div>
      {retry && (
        <button className="quiet-button" onClick={retry}>
          <Icon name="refresh" />
          重试
        </button>
      )}
    </div>
  );
}

export function SectionTitle({
  title,
  detail,
}: {
  title: string;
  detail?: string;
}) {
  return (
    <div className="section-heading">
      <h2>{title}</h2>
      {detail && <span className="muted">{detail}</span>}
    </div>
  );
}
