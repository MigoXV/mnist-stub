import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

async function draw(page: Page) {
  const bounds = (await page.getByLabel("手写数字画布").boundingBox())!;
  await page.mouse.move(
    bounds.x + bounds.width * 0.25,
    bounds.y + bounds.height * 0.23,
  );
  await page.mouse.down();
  await page.mouse.move(
    bounds.x + bounds.width * 0.75,
    bounds.y + bounds.height * 0.23,
    { steps: 10 },
  );
  await page.mouse.move(
    bounds.x + bounds.width * 0.42,
    bounds.y + bounds.height * 0.8,
    { steps: 10 },
  );
  await page.mouse.up();
}
async function recognize(page: Page) {
  await page.getByRole("button", { name: /^(识别数字|重新识别)$/ }).click();
  await expect(page.getByLabel("模型输入预览")).toBeVisible();
}
async function open(page: Page) {
  await page.goto("/");
  await expect(page.getByText("推理服务就绪", { exact: true })).toBeVisible();
}
async function a11y(page: Page) {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"])
    .analyze();
  expect(results.violations).toEqual([]);
}

test("real inference, channel inspector, model details and navigation preservation", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await open(page);
  await page.screenshot({
    path: "test-results/cream-empty.png",
    fullPage: true,
  });
  expect(
    await page.locator(".probability-table tbody").innerText(),
  ).not.toContain("0.00%");
  await draw(page);
  await recognize(page);
  await expect(page.locator(".probability-table tbody tr")).toHaveCount(10);
  await expect(page.locator(".prediction-marker")).toHaveText("预测");
  await a11y(page);
  await page.screenshot({
    path: "test-results/cream-probabilities.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "网络特征" }).click();
  await expect(page.locator(".channel-button")).toHaveCount(8);
  await page.getByRole("button", { name: "下一组" }).click();
  const channel = page.getByRole("button", { name: "检查通道 9", exact: true });
  await channel.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "关闭", exact: true }),
  ).toBeFocused();
  await page.getByText("查看数值矩阵", { exact: true }).click();
  await expect(
    page.getByRole("region", { name: "通道数值矩阵" }),
  ).toBeVisible();
  await a11y(page);
  await page.screenshot({
    path: "test-results/cream-inspector.png",
    fullPage: true,
  });
  await page.keyboard.press("Escape");
  await expect(channel).toBeFocused();
  await page.getByRole("button", { name: /池化 2/ }).click();
  await expect(page.locator(".channel-toolbar")).toContainText("32 × 7 × 7");
  await page.screenshot({
    path: "test-results/cream-features.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: /分类前特征/ }).click();
  await expect(page.locator(".embedding-chart i")).toHaveCount(64);
  await page.getByText("查看全部 64 维数值").click();
  await expect(page.locator(".embedding-values tbody tr")).toHaveCount(64);
  await page.getByRole("link", { name: "模型与服务", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "当前模型资产" }),
  ).toBeVisible();
  await expect(page.getByRole("heading", { name: "服务观测" })).toBeVisible();
  await a11y(page);
  await page.screenshot({
    path: "test-results/cream-model-service.png",
    fullPage: true,
  });
  await page.goBack();
  await expect(page.getByLabel("模型输入预览")).toBeVisible();
  await page.getByRole("button", { name: "清空", exact: true }).click();
  await expect(page.locator(".digit-result strong")).toHaveText("—");
  expect(errors).toEqual([]);
});

test("keyboard upload, feature opt-in recovery and invalid file", async ({
  page,
}) => {
  await open(page);
  await page.getByRole("button", { name: "上传图片", exact: true }).focus();
  await page.keyboard.press("Enter");
  const data = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 96;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, 96, 96);
    ctx.strokeStyle = "black";
    ctx.lineWidth = 8;
    ctx.beginPath();
    ctx.moveTo(20, 20);
    ctx.lineTo(75, 20);
    ctx.lineTo(42, 80);
    ctx.stroke();
    return canvas.toDataURL("image/png").split(",")[1];
  });
  await page.getByRole("button", { name: "选择图片", exact: true }).focus();
  const chooser = page.waitForEvent("filechooser");
  await page.keyboard.press("Enter");
  await (
    await chooser
  ).setFiles({
    name: "seven.png",
    mimeType: "image/png",
    buffer: Buffer.from(data, "base64"),
  });
  await expect(page.getByAltText("上传的数字")).toBeVisible();
  await page.getByRole("checkbox").uncheck();
  await recognize(page);
  await page.getByRole("button", { name: "网络特征" }).click();
  await expect(page.getByText("本次未获取中间层特征")).toBeVisible();
  await page.getByRole("button", { name: "开启中间结果并重新识别" }).click();
  await expect(page.locator(".channel-button")).toHaveCount(8);
  await page
    .locator("input[type=file]")
    .setInputFiles({
      name: "large.png",
      mimeType: "image/png",
      buffer: Buffer.alloc(2097153),
    });
  await expect(page.getByRole("alert")).toContainText("超过 2 MiB");
  await expect(page.locator(".digit-result strong")).toHaveText("—");
});

test("failed request recovers and cancelled late responses never replace the input", async ({
  page,
}) => {
  await open(page);
  await draw(page);
  await page.route("**/api/predict?*", (route) =>
    route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ detail: "推理队列已满" }),
    }),
  );
  await page.getByRole("button", { name: "识别数字", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("稍后重试");
  await page.screenshot({
    path: "test-results/cream-error.png",
    fullPage: true,
  });
  await page.unroute("**/api/predict?*");
  await recognize(page);
  let release: (() => void) | undefined;
  const pending = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route("**/api/predict?*", async (route) => {
    const response = await route.fetch();
    await pending;
    await route.fulfill({ response }).catch(() => {});
  });
  await page.getByRole("button", { name: "识别数字", exact: true }).click();
  await expect(page.getByRole("button", { name: "取消等待" })).toBeVisible();
  await page.getByRole("button", { name: "取消等待" }).click();
  release!();
  await expect(page.locator(".digit-result strong")).toHaveText("—");
  await expect(page.getByText(/已取消等待/)).toBeVisible();
  await page.unroute("**/api/predict?*");
  await recognize(page);
});

test("readiness and model metadata have independent recovery states", async ({
  page,
}) => {
  await page.route("**/readyz", (route) =>
    route.fulfill({ status: 503, json: { ready: false } }),
  );
  await page.route("**/api/model", (route) =>
    route.fulfill({ status: 500, json: { detail: "unavailable" } }),
  );
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "识别数字", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByText("模型信息暂不可用；服务就绪时仍可识别。"),
  ).toBeVisible();
  await page.unroute("**/readyz");
  await page.getByRole("button", { name: "重新连接", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "识别数字", exact: true }),
  ).toBeEnabled();
  await page.unroute("**/api/model");
  await page.getByRole("button", { name: "重试模型信息" }).click();
  await expect(
    page.getByText("模型信息暂不可用；服务就绪时仍可识别。"),
  ).toBeHidden();
  await page.route("**/readyz", (route) => route.abort("failed"));
  await page.waitForTimeout(10500);
  await expect(page.locator(".connection-state")).toHaveText("连接中断");
  await a11y(page);
});

test("small, medium and wide layouts retain their task and have no overflow", async ({
  page,
}) => {
  await open(page);
  await draw(page);
  await recognize(page);
  for (const width of [320, 768, 1280, 1440]) {
    await page.setViewportSize({ width, height: 1000 });
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth),
    ).toBe(width);
    await expect(page.getByLabel("模型输入预览")).toBeVisible();
    await a11y(page);
    await page.screenshot({
      path: "test-results/cream-" + width + ".png",
      fullPage: true,
    });
  }
  await page.setViewportSize({ width: 320, height: 800 });
  await page.getByRole("button", { name: "输入", exact: true }).click();
  await expect(page.getByLabel("手写数字画布")).toBeVisible();
  await expect(page.getByLabel("模型输入预览")).toBeHidden();
  await page.getByRole("button", { name: "分析", exact: true }).click();
  await page.getByRole("button", { name: "网络特征" }).click();
  await page.getByRole("button", { name: "检查通道 1", exact: true }).click();
  await a11y(page);
  await page.keyboard.press("Escape");
  await page.getByRole("link", { name: "模型与服务", exact: true }).click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    320,
  );
  await a11y(page);
  await page.screenshot({
    path: "test-results/cream-model-mobile.png",
    fullPage: true,
  });
});

test("touch, reduced motion and effective 200 percent browser zoom", async ({
  browser,
}) => {
  const context = await browser.newContext({
    viewport: { width: 640, height: 480 },
    deviceScaleFactor: 2,
    hasTouch: true,
    reducedMotion: "reduce",
    baseURL: process.env.MNIST_WEB_URL || "http://127.0.0.1:8000",
  });
  const page = await context.newPage();
  await open(page);
  await page.getByLabel("手写数字画布").tap();
  await recognize(page);
  await expect(
    page.getByRole("button", { name: "分析", exact: true }),
  ).toBeFocused();
  expect(
    await page
      .locator(".probability-track i")
      .first()
      .evaluate((el) => getComputedStyle(el).transitionDuration),
  ).toBe("0s");
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    640,
  );
  await a11y(page);
  await page.screenshot({
    path: "test-results/cream-zoom-200.png",
    fullPage: true,
  });
  await context.close();
});

test("missing provenance and zero sample metrics are represented honestly", async ({
  page,
}) => {
  await page.route("**/api/model", async (route) => {
    const response = await route.fetch();
    const data = await response.json();
    data.metadata = {};
    await route.fulfill({ json: data });
  });
  await page.route("**/api/metrics", (route) =>
    route.fulfill({
      json: {
        counts: {},
        active: 0,
        queued: 0,
        capacity: 8,
        queue_high_water: 0,
        latency_window: 0,
        latency_ms: { p50: 0, p95: 0, p99: 0 },
        mean_queue_ms: 0,
      },
    }),
  );
  await open(page);
  await page.getByRole("link", { name: "模型与服务", exact: true }).click();
  await expect(page.getByText("未提供", { exact: true })).toHaveCount(3);
  await expect(page.getByText("暂无样本", { exact: true })).toHaveCount(3);
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "当前模型资产" }),
  ).toBeVisible();
});

test("semantic text, controls and focus tokens meet contrast requirements", async ({
  page,
}) => {
  await open(page);
  const pairs = await page.evaluate(() => {
    const css = getComputedStyle(document.documentElement);
    function luminance(token: string) {
      const value = css.getPropertyValue(token).trim().slice(1);
      const rgb = [0, 2, 4]
        .map((i) => parseInt(value.slice(i, i + 2), 16) / 255)
        .map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
      return rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
    }
    return [
      ["--text", "--surface", 4.5],
      ["--muted", "--canvas", 4.5],
      ["--muted", "--selection", 4.5],
      ["--text", "--action", 4.5],
      ["--success", "--surface", 4.5],
      ["--error", "--error-surface", 4.5],
      ["--control-line", "--surface", 3],
      ["--focus", "--canvas", 3],
    ].map(([a, b, minimum]) => {
      const x = luminance(a as string),
        y = luminance(b as string);
      return {
        a,
        b,
        minimum: minimum as number,
        contrast: (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05),
      };
    });
  });
  for (const pair of pairs)
    expect(pair.contrast, JSON.stringify(pair)).toBeGreaterThanOrEqual(
      pair.minimum,
    );
});
