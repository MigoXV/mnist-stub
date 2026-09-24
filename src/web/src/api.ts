import { useCallback, useEffect, useRef, useState } from "react";

export type Resource<T> = {
  data: T | null;
  error: string;
  loading: boolean;
  updatedAt: Date | null;
  refresh: () => void;
};

export function useResource<T>(url: string, interval = 15000): Resource<T> {
  const [state, setState] = useState<Omit<Resource<T>, "refresh">>({
    data: null,
    error: "",
    loading: true,
    updatedAt: null,
  });
  const current = useRef<AbortController | null>(null);
  const refresh = useCallback(() => {
    current.current?.abort();
    const controller = new AbortController();
    current.current = controller;
    setState((previous) => ({ ...previous, loading: true }));
    const timer = window.setTimeout(() => controller.abort(), 6000);
    void (async () => {
      try {
        const response = await fetch(url, {
          signal: controller.signal,
          cache: "no-store",
        });
        if (!response.ok && !(url === "/readyz" && response.status === 503)) {
          throw new Error("请求失败（HTTP " + response.status + "）");
        }
        const data = (await response.json()) as T;
        if (current.current === controller) {
          setState({ data, error: "", loading: false, updatedAt: new Date() });
        }
      } catch (error) {
        if (current.current === controller) {
          setState((previous) => ({
            ...previous,
            loading: false,
            error: controller.signal.aborted
              ? "连接超时，请重新检查"
              : error instanceof Error
                ? error.message
                : "无法连接服务",
          }));
        }
      } finally {
        window.clearTimeout(timer);
      }
    })();
  }, [url]);
  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, interval);
    return () => {
      window.clearInterval(timer);
      const controller = current.current;
      current.current = null;
      controller?.abort();
    };
  }, [refresh, interval]);
  return { ...state, refresh };
}

export async function predictImage(
  payload: Blob,
  features: boolean,
  polarity: string,
  signal: AbortSignal,
) {
  const response = await fetch(
    "/api/predict?features=" + features + "&polarity=" + polarity,
    {
      method: "POST",
      body: payload,
      signal,
      headers: { "Content-Type": payload.type || "application/octet-stream" },
    },
  );
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const reason =
      typeof body?.detail === "string" ? body.detail : "识别请求失败";
    const recovery =
      response.status === 503
        ? "稍后重试，或检查服务状态。"
        : response.status === 504
          ? "等待超时，可重新识别。"
          : response.status === 422 ||
              response.status === 413 ||
              response.status === 415
            ? "请调整输入后重试。"
            : "请检查服务后重试。";
    throw new Error(reason + "。" + recovery);
  }
  return response.json();
}
