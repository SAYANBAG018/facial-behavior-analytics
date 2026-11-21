import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";

const EMOTIONS = [
  "angry",
  "disgust",
  "fear",
  "happy",
  "neutral",
  "sad",
  "surprise",
];

const COLORS = {
  angry: "#b42318",
  disgust: "#667a1f",
  fear: "#6b4fa3",
  happy: "#e0a100",
  neutral: "#6b7c88",
  sad: "#2b6cb0",
  surprise: "#d6336c",
};

const POSE = [
  "head_yaw_degrees",
  "head_pitch_degrees",
  "head_roll_degrees",
];

const POSE_COLORS = ["#0e6b73", "#b45309", "#4338ca"];

const round = (v, d = 2) =>
  typeof v === "number" ? Number(v.toFixed(d)) : v;

function download(name, text, type) {
  const blob = new Blob([text ?? ""], {
    type: `${type};charset=utf-8`,
  });

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");

  a.href = url;
  a.download = name;

  document.body.appendChild(a);
  a.click();
  a.remove();

  // Revoke after the browser has had time to begin the download.
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function toCsv(rows) {
  if (!rows.length) return "";

  const cols = Object.keys(rows[0]);

  const esc = (v) => {
    if (v === null || v === undefined) return "";

    const value = String(v);
    return `"${value.replace(/"/g, '""')}"`;
  };

  return [
    cols.map(esc).join(","),
    ...rows.map((r) => cols.map((c) => esc(r[c])).join(",")),
  ].join("\n");
}

function Table({
  rows,
  digits = 2,
  label = "Data table",
}) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return <p className="cap">No data available.</p>;
  }

  const cols = Object.keys(rows[0] ?? {});

  if (!cols.length) {
    return <p className="cap">No data available.</p>;
  }

  return (
    <div
      className="scroll table-scroll"
      role="region"
      aria-label={label}
      tabIndex={0}
    >
      <table>
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>

        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {cols.map((c) => (
                <td key={c}>
                  {String(round(r[c], digits) ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ---------------- MARKDOWN RENDERING ---------------- */

// Escape HTML to prevent assistant output from injecting HTML.
function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatInlineMarkdown(value) {
  let text = escapeHtml(value);
  const code = [];

  text = text.replace(/`([^`]+)`/g, (_, content) => {
    const token = `@@CODE${code.length}@@`;
    code.push(`<code>${content}</code>`);
    return token;
  });

  text = text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/__(.+?)__/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/_(.+?)_/g, "<em>$1</em>");

  return text.replace(
    /@@CODE(\d+)@@/g,
    (_, index) => code[Number(index)]
  );
}

function markdownToHtml(markdown) {
  const lines = String(markdown ?? "")
    .replace(/\r\n?/g, "\n")
    .split("\n");

  const output = [];

  let paragraph = [];
  let listType = null;
  let listItems = [];
  let codeLines = [];
  let inCode = false;

  const flushParagraph = () => {
    if (paragraph.length) {
      output.push(
        `<p>${paragraph.map(formatInlineMarkdown).join("<br />")}</p>`
      );

      paragraph = [];
    }
  };

  const flushList = () => {
    if (listType && listItems.length) {
      output.push(
        `<${listType}>${listItems
          .map(
            (item) =>
              `<li>${formatInlineMarkdown(item)}</li>`
          )
          .join("")}</${listType}>`
      );
    }

    listType = null;
    listItems = [];
  };

  for (const line of lines) {
    if (/^\s*```/.test(line)) {
      flushParagraph();
      flushList();

      if (inCode) {
        output.push(
          `<pre><code>${escapeHtml(
            codeLines.join("\n")
          )}</code></pre>`
        );

        codeLines = [];
        inCode = false;
      } else {
        inCode = true;
      }

      continue;
    }

    if (inCode) {
      codeLines.push(line);
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      flushList();
      continue;
    }

    const heading = line.match(
      /^\s{0,3}(#{1,4})\s+(.+)$/
    );

    if (heading) {
      flushParagraph();
      flushList();

      const level = heading[1].length;

      output.push(
        `<h${level}>${formatInlineMarkdown(
          heading[2]
        )}</h${level}>`
      );

      continue;
    }

    const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);

    if (unordered || ordered) {
      flushParagraph();

      const nextType = unordered ? "ul" : "ol";

      if (listType && listType !== nextType) {
        flushList();
      }

      listType = nextType;
      listItems.push((unordered || ordered)[1]);

      continue;
    }

    flushList();
    paragraph.push(line);
  }

  if (inCode) {
    output.push(
      `<pre><code>${escapeHtml(
        codeLines.join("\n")
      )}</code></pre>`
    );
  }

  flushParagraph();
  flushList();

  return output.join("");
}

function AssistantMessage({ content }) {
  const html = markdownToHtml(content);

  return (
    <div
      className="markdown-content"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

/* ---------------- API RESPONSE HANDLING ---------------- */

async function readApiResponse(response, fallbackMessage) {
  const raw = await response.text();

  let payload = {};

  try {
    payload = raw ? JSON.parse(raw) : {};
  } catch {
    payload = {};
  }

  if (!response.ok) {
    const detail = payload.detail;

    const message =
      typeof detail === "string"
        ? detail
        : detail?.message ||
          payload.message ||
          fallbackMessage;

    throw new Error(message);
  }

  return payload;
}

/* ---------------- AI ASSISTANT ---------------- */

function Chat({ report }) {
  const [msgs, setMsgs] = useState([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function send(e) {
    e.preventDefault();

    const question = q.trim();

    if (!question || busy) return;

    // Send only previous completed turns as history.
    const history = msgs.map(({ role, content }) => ({
      role,
      content,
    }));

    setMsgs((previous) => [
      ...previous,
      {
        role: "user",
        content: question,
      },
    ]);

    setQ("");
    setErr("");
    setBusy(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          report,
          history,
        }),
      });

      const data = await readApiResponse(
        res,
        "The assistant request failed."
      );

      if (
        typeof data.answer !== "string" ||
        !data.answer.trim()
      ) {
        throw new Error(
          "The assistant returned an empty response. Please try again."
        );
      }

      setMsgs((m) => [
        ...m,
        {
          role: "assistant",
          content: data.answer,
        },
      ]);
    } catch (ex) {
      setErr(
        ex.message || "Unable to contact the AI Assistant."
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="chat">
      <div className="chat-intro">
        <p>
          Ask questions about the video analysis. Answers are
          grounded in the available session report.
        </p>
      </div>

      <div className="msgs" aria-live="polite">
        {!msgs.length && (
          <div className="chat-empty">
            <div className="chat-empty-icon">✦</div>

            <strong>How can I help?</strong>

            <p>
              Ask about expression predictions, head movement,
              or data quality.
            </p>

            <div className="suggestions">
              <button
                type="button"
                className="suggestion"
                onClick={() =>
                  setQ(
                    "Which expression was predicted most often?"
                  )
                }
              >
                Which expression was predicted most often?
              </button>

              <button
                type="button"
                className="suggestion"
                onClick={() =>
                  setQ("Explain the head-pose results.")
                }
              >
                Explain the head-pose results.
              </button>

              <button
                type="button"
                className="suggestion"
                onClick={() =>
                  setQ("Summarize the data quality.")
                }
              >
                Summarize the data quality.
              </button>
            </div>
          </div>
        )}

        {msgs.map((m, i) => (
          <div
            key={`${m.role}-${i}`}
            className={`msg ${m.role}`}
          >
            {m.role === "assistant" ? (
              <AssistantMessage content={m.content} />
            ) : (
              m.content
            )}
          </div>
        ))}

        {busy && (
          <div className="msg assistant">
            Checking the evidence…
          </div>
        )}
      </div>

      {err && (
        <div className="error" role="alert">
          {err}
        </div>
      )}

      <form className="ask" onSubmit={send}>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Ask about your results..."
          aria-label="Question"
          disabled={busy}
        />

        <button disabled={busy || !q.trim()}>
          {busy ? "..." : "Send"}
        </button>
      </form>

      <p className="chat-disclaimer">
        AI-generated answers may contain errors. Verify important
        interpretations against the displayed data.
      </p>
    </div>
  );
}

/* ---------------- DASHBOARD HELPERS ---------------- */

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "expressions", label: "Expressions" },
  { id: "pose", label: "Head pose" },
  { id: "quality", label: "Data quality" },
  { id: "frames", label: "Frames" },
  { id: "export", label: "Export" },
];

const FEATURES = [
  {
    title: "Expression predictions",
    text: "Per-frame labels from a MobileNetV2 classifier, shown with its raw model scores.",
  },
  {
    title: "Head-pose tracking",
    text: "Yaw, pitch and roll estimated from MediaPipe face landmarks on each sampled frame.",
  },
  {
    title: "Grounded assistant",
    text: "Ask questions about the results. Answers use only the session report and metrics.",
  },
];

const VIDEO_TYPES = /\.(mp4|avi|mov|mkv|webm|mpeg|mpg)$/i;

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// Safe number formatting: returns "—" instead of crashing on missing values.
function fmt(value, digits = 1) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";
}

function pct(value, digits = 1) {
  const text = fmt(value, digits);
  return text === "—" ? "—" : `${text}%`;
}

function Kpi({ label, value, sub, accent = false }) {
  return (
    <div className={`kpi ${accent ? "accent" : ""}`}>
      <span className="kpi-label">{label}</span>
      <strong className="kpi-value">{value}</strong>
      {sub && <span className="kpi-sub">{sub}</span>}
    </div>
  );
}

function ChartCard({ title, hint, children, className = "" }) {
  return (
    <section className={`card ${className}`}>
      <div className="card-head">
        <h3>{title}</h3>
        {hint && <span className="card-hint">{hint}</span>}
      </div>
      {children}
    </section>
  );
}

function Empty({ children }) {
  return <p className="cap">{children}</p>;
}

const LEGEND_STYLE = { fontSize: 12, paddingTop: 8 };

/* ---------------- MAIN DASHBOARD ---------------- */

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [interval, setInterval_] = useState(1.0);
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState("");
  const [data, setData] = useState(null);
  const [analysisVersion, setAnalysisVersion] = useState(0);
  const [tab, setTab] = useState("overview");
  const [dragOver, setDragOver] = useState(false);

  // Preview the selected video, and release the preview URL when it changes.
  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Count elapsed seconds while the analysis runs.
  useEffect(() => {
    if (!busy) {
      setElapsed(0);
      return undefined;
    }

    const started = Date.now();
    const timer = window.setInterval(() => {
      setElapsed(Math.floor((Date.now() - started) / 1000));
    }, 500);

    return () => window.clearInterval(timer);
  }, [busy]);

  function chooseFile(selected) {
    if (!selected) return;

    const isVideo =
      selected.type.startsWith("video/") ||
      VIDEO_TYPES.test(selected.name);

    if (!isVideo) {
      setError(
        "Please choose a video file (MP4, AVI, MOV, MKV, WebM or MPEG)."
      );
      return;
    }

    setFile(selected);
    setError("");
  }

  async function analyze() {
    if (!file || busy) return;

    setBusy(true);
    setError("");

    const body = new FormData();

    body.append("file", file);
    body.append("interval", interval);

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        body,
      });

      const json = await readApiResponse(
        res,
        "Video analysis failed."
      );

      if (!json.report || !Array.isArray(json.frames)) {
        throw new Error(
          "The analysis response is missing required report or frame data."
        );
      }

      setData(json);
      setTab("overview");
      setAnalysisVersion((version) => version + 1);
    } catch (ex) {
      setError(
        ex.message || "An unexpected analysis error occurred."
      );
    } finally {
      setBusy(false);
    }
  }

  const r = data?.report;
  const q = r?.data_quality ?? {};

  const frames = Array.isArray(data?.frames) ? data.frames : [];

  const detected = frames.filter((f) =>
    EMOTIONS.includes(f.emotion)
  );

  const counts = (data?.emotion_counts ?? []).filter(
    (c) => c.sample_count > 0
  );

  const dominantRow = counts.find(
    (c) => c.predicted_expression === r?.dominant_predicted_expression
  );

  const qualityRows = Object.entries(q).map(([Metric, Value]) => ({
    Metric,
    Value,
  }));

  const summaryTxt =
    r &&
    [
      "MULTIMODAL FACIAL BEHAVIOR ANALYTICS",
      "=".repeat(40),
      `Video: ${r.source_video}`,
      `Frames analyzed: ${q.frames_analyzed ?? "N/A"}`,
      `Face detection rate: ${q.face_detection_rate_percent ?? "N/A"}%`,
      `Head-pose availability: ${q.head_pose_availability_percent ?? "N/A"}%`,
      `Prediction changes: ${r.prediction_changes}`,
      `Most frequent predicted expression: ${
        r.dominant_predicted_expression ?? "N/A"
      }`,
      "",
      "INTERPRETATION LIMITS",
      ...(Array.isArray(r.interpretation_limits)
        ? r.interpretation_limits.map((l) => `- ${l}`)
        : [
            "- Interpret results cautiously; see the session report for limitations.",
          ]),
    ].join("\n");

  const status = busy ? "Analyzing" : r ? "Results ready" : "Ready";

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="logo" aria-hidden="true">FA</div>
          <div>
            <div className="brand-label">Multimodal analysis</div>
            <h1>Facial Behavior Analytics</h1>
          </div>
        </div>

        <span className={`status ${busy ? "busy" : r ? "done" : ""}`}>
          <span className="status-dot" />
          {status}
        </span>
      </header>

      <main className="page">
        <p className="intro">
          Upload a video to get frame-by-frame expression predictions,
          head-orientation measurements and data-quality checks. Then ask
          the AI assistant about the results.
        </p>

        <div className="notice" role="note">
          <strong>Interpretation note.</strong> Expression labels are
          model predictions, and head pose is an orientation estimate.
          Neither shows a person's actual emotions, attention, intentions
          or psychological state.
        </div>

        {/* Upload */}
        <section className="card upload-card">
          <div className="upload-grid">
            <label
              className={[
                "dropzone",
                dragOver ? "drag" : "",
                busy ? "disabled" : "",
              ].join(" ")}
              onDragOver={(e) => {
                e.preventDefault();
                if (!busy) setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                if (!busy) chooseFile(e.dataTransfer.files?.[0]);
              }}
            >
              <input
                type="file"
                accept="video/*,.mp4,.avi,.mov,.mkv,.webm,.mpeg,.mpg"
                disabled={busy}
                onChange={(e) => chooseFile(e.target.files?.[0])}
              />

              <span className="dz-icon" aria-hidden="true">+</span>

              <strong className="dz-title">
                {file ? file.name : "Drop a video here or click to browse"}
              </strong>

              <span className="dz-meta">
                {file
                  ? `${formatBytes(file.size)} · click to choose another`
                  : "MP4, AVI, MOV, MKV, WebM or MPEG"}
              </span>
            </label>

            {previewUrl && (
              <video
                className="preview"
                src={previewUrl}
                controls
                preload="metadata"
                aria-label="Selected video preview"
              />
            )}
          </div>

          <div className="controls">
            <label className="slider">
              <span>
                Sampling interval <b>{interval.toFixed(1)} s</b>
              </span>

              <input
                type="range"
                min="0.5"
                max="3"
                step="0.5"
                value={interval}
                disabled={busy}
                onChange={(e) => setInterval_(Number(e.target.value))}
              />

              <small>
                Shorter intervals analyze more frames but take longer.
              </small>
            </label>

            <button
              type="button"
              className="btn-primary"
              onClick={analyze}
              disabled={!file || busy}
            >
              {busy ? (
                <>
                  <span className="spinner" aria-hidden="true" />
                  Analyzing…
                </>
              ) : (
                "Analyze video"
              )}
            </button>
          </div>

          {error && (
            <div className="error" role="alert">
              {error}
            </div>
          )}
        </section>

        {/* Loading */}
        {busy && (
          <section className="card loading-card" role="status" aria-live="polite">
            <div className="loader" aria-hidden="true" />
            <div>
              <strong>Analyzing your video</strong>
              <p className="cap">
                Detecting faces, predicting expressions and measuring head
                pose on each sampled frame. Elapsed: {elapsed}s
              </p>
            </div>
          </section>
        )}

        {/* Empty state */}
        {!r && !busy && (
          <section className="features">
            {FEATURES.map((item, i) => (
              <div className="card feature" key={item.title}>
                <span className="feature-num">{i + 1}</span>
                <h3>{item.title}</h3>
                <p>{item.text}</p>
              </div>
            ))}
          </section>
        )}

        {/* Results */}
        {r && (
          <>
            <section className="kpis">
              <Kpi
                label="Dominant prediction"
                value={r.dominant_predicted_expression ?? "—"}
                sub={
                  dominantRow
                    ? `${pct(
                        dominantRow.percentage_of_detected_samples
                      )} of detected samples`
                    : "No dominant expression"
                }
                accent
              />
              <Kpi
                label="Sampled frames"
                value={q.frames_analyzed ?? "—"}
                sub={`One every ${fmt(r.sampling_interval_seconds)} s`}
              />
              <Kpi
                label="Face detection"
                value={pct(q.face_detection_rate_percent)}
                sub="of sampled frames"
              />
              <Kpi
                label="Head-pose availability"
                value={pct(q.head_pose_availability_percent)}
                sub="of sampled frames"
              />
            </section>

            <div className="results">
              <div className="results-main">
                <nav className="tabs" role="tablist" aria-label="Results">
                  {TABS.map((t) => (
                    <button
                      key={t.id}
                      type="button"
                      role="tab"
                      aria-selected={tab === t.id}
                      className={`tab ${tab === t.id ? "active" : ""}`}
                      onClick={() => setTab(t.id)}
                    >
                      {t.label}
                    </button>
                  ))}
                </nav>

                <p className="cap result-caption">
                  Sampled-frame results for {r.source_video}. Last sample at{" "}
                  {fmt(r.last_sampled_timestamp_seconds, 2)} s.
                </p>

                {/* Overview */}
                {tab === "overview" && (
                  <>
                    <div className="grid-2">
                      <ChartCard
                        title="Expression distribution"
                        hint="Share of detected samples"
                      >
                        {counts.length ? (
                          <ResponsiveContainer width="100%" height={280}>
                            <PieChart>
                              <Pie
                                data={counts}
                                dataKey="sample_count"
                                nameKey="predicted_expression"
                                innerRadius={62}
                                outerRadius={98}
                                paddingAngle={2}
                              >
                                {counts.map((c) => (
                                  <Cell
                                    key={c.predicted_expression}
                                    fill={COLORS[c.predicted_expression] ?? "#94a3b8"}
                                  />
                                ))}
                              </Pie>
                              <Tooltip />
                              <Legend verticalAlign="bottom" height={36} />
                            </PieChart>
                          </ResponsiveContainer>
                        ) : (
                          <Empty>No valid expression predictions were recorded.</Empty>
                        )}
                      </ChartCard>

                      <ChartCard title="Prediction frequency" hint="Sample counts">
                        {counts.length ? (
                          <ResponsiveContainer width="100%" height={280}>
                            <BarChart data={counts} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                              <CartesianGrid strokeDasharray="3 3" vertical={false} />
                              <XAxis
                                dataKey="predicted_expression"
                                tick={{ fontSize: 12 }}
                              />
                              <YAxis allowDecimals={false} />
                              <Tooltip />
                              <Bar dataKey="sample_count" radius={[6, 6, 0, 0]}>
                                {counts.map((c) => (
                                  <Cell
                                    key={c.predicted_expression}
                                    fill={COLORS[c.predicted_expression] ?? "#94a3b8"}
                                  />
                                ))}
                              </Bar>
                            </BarChart>
                          </ResponsiveContainer>
                        ) : (
                          <Empty>No valid expression predictions were recorded.</Empty>
                        )}
                      </ChartCard>
                    </div>

                    <ChartCard title="Consecutive prediction runs">
                      <Table rows={data.emotion_runs ?? []} />
                      <p className="cap">
                        Runs are consecutive sampled predictions, not the
                        verified duration of a real expression.
                      </p>
                    </ChartCard>
                  </>
                )}

                {/* Expressions */}
                {tab === "expressions" && (
                  <>
                    <ChartCard
                      title="Model scores over time"
                      hint="Raw outputs, not probabilities"
                    >
                      <ResponsiveContainer width="100%" height={320}>
                        <LineChart data={frames}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} />
                          <XAxis
                            dataKey="timestamp_seconds"
                            tickFormatter={(v) => round(v, 1)}
                            tick={{ fontSize: 12 }}
                          />
                          <YAxis />
                          <Tooltip />
                          <Legend
                            verticalAlign="bottom"
                            align="center"
                            wrapperStyle={LEGEND_STYLE}
                          />
                          {EMOTIONS.map((e) => (
                            <Line
                              key={e}
                              dataKey={`emotion_score_${e}`}
                              name={e}
                              stroke={COLORS[e]}
                              dot={false}
                              strokeWidth={2}
                            />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                      <p className="cap">
                        Scores are raw model outputs, not calibrated
                        probabilities.
                      </p>
                    </ChartCard>

                    <ChartCard title="Prediction timeline" hint="Detected samples only">
                      {detected.length ? (
                        <Table
                          rows={detected.map((f) => ({
                            timestamp_seconds: f.timestamp_seconds,
                            emotion: f.emotion,
                          }))}
                        />
                      ) : (
                        <Empty>No valid expression predictions were recorded.</Empty>
                      )}
                    </ChartCard>
                  </>
                )}

                {/* Head pose */}
                {tab === "pose" && (
                  <>
                    <ChartCard title="Head-pose movement" hint="Degrees over time">
                      {(data.pose_series ?? []).length ? (
                        <ResponsiveContainer width="100%" height={320}>
                          <LineChart data={data.pose_series}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                            <XAxis
                              dataKey="timestamp_seconds"
                              tickFormatter={(v) => round(v, 1)}
                              tick={{ fontSize: 12 }}
                            />
                            <YAxis unit="°" />
                            <Tooltip />
                            <Legend
                              verticalAlign="bottom"
                              align="center"
                              wrapperStyle={LEGEND_STYLE}
                            />
                            {POSE.map((p, i) => (
                              <Line
                                key={p}
                                dataKey={p}
                                name={p.replace("head_", "").replace("_degrees", "")}
                                stroke={POSE_COLORS[i]}
                                dot={false}
                                strokeWidth={2}
                              />
                            ))}
                          </LineChart>
                        </ResponsiveContainer>
                      ) : (
                        <Empty>No valid head-pose measurements are available.</Empty>
                      )}
                    </ChartCard>

                    <div className="grid-2">
                      <ChartCard title="Orientation statistics">
                        <Table rows={data.pose_summary ?? []} />
                      </ChartCard>

                      <ChartCard title="Angle changes and rates">
                        <Table
                          rows={(data.movement_summary ?? []).map((m) =>
                            m["Value (degrees)"] != null
                              ? {
                                  Metric: m.Metric,
                                  Value: m["Value (degrees)"],
                                  Unit: "degrees",
                                }
                              : {
                                  Metric: m.Metric,
                                  Value: m["Value (degrees/second)"],
                                  Unit: "degrees/second",
                                }
                          )}
                        />
                        <p className="cap">
                          Changes are absolute differences between consecutive
                          valid samples. Rates are approximate and can reflect
                          pose-estimation noise.
                        </p>
                      </ChartCard>
                    </div>
                  </>
                )}

                {/* Data quality */}
                {tab === "quality" && (
                  <>
                    <ChartCard title="Data quality summary">
                      <Table rows={qualityRows} />
                    </ChartCard>

                    <ChartCard title="Face visibility over time" hint="1 = face detected">
                      <ResponsiveContainer width="100%" height={200}>
                        <LineChart
                          data={frames.map((f) => ({
                            t: f.timestamp_seconds,
                            visible: f.face_visible ? 1 : 0,
                          }))}
                        >
                          <CartesianGrid strokeDasharray="3 3" vertical={false} />
                          <XAxis
                            dataKey="t"
                            tickFormatter={(v) => round(v, 1)}
                            tick={{ fontSize: 12 }}
                          />
                          <YAxis domain={[0, 1]} ticks={[0, 1]} />
                          <Tooltip />
                          <Line
                            type="stepAfter"
                            dataKey="visible"
                            stroke="#0e6b73"
                            strokeWidth={2}
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </ChartCard>
                  </>
                )}

                {/* Frames */}
                {tab === "frames" && (
                  <ChartCard title="Frame-by-frame results">
                    <Table
                      rows={frames}
                      digits={3}
                      label="Frame-by-frame sampled video results"
                    />
                  </ChartCard>
                )}

                {/* Export */}
                {tab === "export" && (
                  <ChartCard title="Export session results">
                    <div className="downloads">
                      <button
                        type="button"
                        className="btn-secondary"
                        onClick={() =>
                          download(
                            "facial_behavior_results.csv",
                            `\uFEFF${toCsv(frames)}`,
                            "text/csv"
                          )
                        }
                      >
                        Download CSV
                      </button>

                      <button
                        type="button"
                        className="btn-secondary"
                        onClick={() =>
                          download(
                            "facial_behavior_report.json",
                            JSON.stringify(r, null, 2),
                            "application/json"
                          )
                        }
                      >
                        Download report JSON
                      </button>

                      <button
                        type="button"
                        className="btn-secondary"
                        onClick={() =>
                          download(
                            "facial_behavior_summary.txt",
                            summaryTxt,
                            "text/plain"
                          )
                        }
                      >
                        Download summary TXT
                      </button>
                    </div>

                    <details className="raw">
                      <summary>Inspect report JSON</summary>
                      <pre>{JSON.stringify(r, null, 2)}</pre>
                    </details>
                  </ChartCard>
                )}
              </div>

              {/* AI assistant */}
              <aside className="assistant">
                <div className="assistant-sticky">
                  <div className="assistant-heading">
                    <div className="assistant-icon" aria-hidden="true">✦</div>
                    <div>
                      <h2>AI Assistant</h2>
                      <p>Ask about your analysis</p>
                    </div>
                  </div>

                  <Chat key={analysisVersion} report={r} />
                </div>
              </aside>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
