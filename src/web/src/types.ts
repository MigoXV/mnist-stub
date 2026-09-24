export type Feature = { shape: number[]; values: number[][][] | number[] };
export type Prediction = {
  prediction: number;
  probabilities: number[];
  input: number[][];
  model_id: string;
  inference_ms: number;
  queue_ms: number;
  features: Record<string, Feature>;
};
export type Metadata = {
  model_id: string;
  architecture: string;
  device: string;
  dtype: string;
  runner: string;
  format_version: number;
  weights_sha256: string;
  labels: number[];
  preprocessing: {
    version: number;
    shape: number[];
    mean: number;
    std: number;
    image_adapter: string;
  };
  features: Record<string, number[]>;
  metadata?: Record<string, unknown>;
};
export type Metrics = {
  counts: Record<string, number>;
  active: number;
  queued: number;
  capacity: number;
  queue_high_water: number;
  latency_window: number;
  latency_ms: Record<string, number>;
  mean_queue_ms: number;
};
export const layers = [
  ["conv1", "卷积 1"],
  ["pool1", "池化 1"],
  ["conv2", "卷积 2"],
  ["pool2", "池化 2"],
  ["embedding", "分类前特征"],
] as const;
