import { useState, useRef, useCallback } from "react";

const PIPELINE_STAGES = [
  { id: "ocr", label: "OCR — Reading Labels", icon: "⬡", detail: "Multi-pass Tesseract, 5 preprocessing variants" },
  { id: "walls", label: "Wall Extraction", icon: "⬡", detail: "Adaptive threshold + morphological processing" },
  { id: "segment", label: "Room Segmentation", icon: "⬡", detail: "Voronoi flood-fill from OCR seed points" },
  { id: "classify", label: "Room Classification", icon: "⬡", detail: "Label normalisation + canonical type mapping" },
  { id: "furniture", label: "Furniture Placement", icon: "⬡", detail: "Wall-constrained, scale-aware, non-overlapping" },
  { id: "symbols", label: "Symbol Drawing", icon: "⬡", detail: "Architectural symbols for ControlNet input" },
  { id: "render", label: "3D Render", icon: "⬡", detail: "ControlNet + Stable Diffusion v1.5" },
];

const OUTPUT_TABS = [
  { id: "colored_plan", label: "Room Map", file: "colored_plan.png" },
  { id: "furnished_plan", label: "Furnished", file: "furnished_plan.png" },
  { id: "line_drawing", label: "Line Art", file: "line_drawing.png" },
  { id: "isometric_render", label: "3D Render", file: "isometric_render.png" },
];

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | uploading | processing | done | error
  const [currentStage, setCurrentStage] = useState(-1);
  const [completedStages, setCompletedStages] = useState([]);
  const [results, setResults] = useState(null);
  const [activeTab, setActiveTab] = useState("colored_plan");
  const [renderEnabled, setRenderEnabled] = useState(false);
  const [error, setError] = useState(null);
  const [jobId, setJobId] = useState(null);
  const fileRef = useRef();
  const pollRef = useRef();

  const handleFile = (f) => {
    if (!f) return;
    if (!["image/jpeg", "image/png", "image/jpg"].includes(f.type)) {
      setError("Please upload a JPG or PNG floor plan image.");
      return;
    }
    setFile(f);
    setError(null);
    setStatus("idle");
    setResults(null);
    setCompletedStages([]);
    setCurrentStage(-1);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(f);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  const onDragOver = (e) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);

  const pollJob = (id) => {
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`/api/status/${id}`);
        const data = await res.json();
        if (data.stage_index !== undefined) setCurrentStage(data.stage_index);
        if (data.completed_stages) setCompletedStages(data.completed_stages);
        if (data.status === "done") {
          clearInterval(pollRef.current);
          setStatus("done");
          setResults(data.outputs);
          setCurrentStage(-1);
        } else if (data.status === "error") {
          clearInterval(pollRef.current);
          setStatus("error");
          setError(data.message || "Processing failed. Check your floor plan image.");
        }
      } catch {
        clearInterval(pollRef.current);
        setStatus("error");
        setError("Lost connection to server.");
      }
    }, 1500);
  };

  const handleSubmit = async () => {
    if (!file) return;
    setStatus("uploading");
    setError(null);
    setCompletedStages([]);
    setCurrentStage(0);
    const form = new FormData();
    form.append("file", file);
    form.append("render", renderEnabled ? "true" : "false");
    try {
      const res = await fetch("/api/process", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Upload failed");
      setJobId(data.job_id);
      setStatus("processing");
      pollJob(data.job_id);
    } catch (e) {
      setStatus("error");
      setError(e.message);
    }
  };

  const reset = () => {
    clearInterval(pollRef.current);
    setFile(null); setPreview(null); setStatus("idle");
    setResults(null); setCurrentStage(-1); setCompletedStages([]);
    setError(null); setJobId(null); setActiveTab("colored_plan");
  };

  const isProcessing = status === "uploading" || status === "processing";

  return (
    <div className="app">
      {/* Background grid */}
      <div className="bg-grid" aria-hidden />

      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <rect x="2" y="2" width="13" height="13" rx="1" stroke="var(--accent)" strokeWidth="1.5"/>
              <rect x="17" y="2" width="13" height="8" rx="1" stroke="var(--accent)" strokeWidth="1.5"/>
              <rect x="2" y="17" width="8" height="13" rx="1" stroke="var(--accent)" strokeWidth="1.5"/>
              <rect x="12" y="17" width="18" height="13" rx="1" stroke="var(--accent)" strokeWidth="1.5"/>
              <rect x="17" y="12" width="13" height="3" rx="0.5" fill="var(--accent)" opacity="0.4"/>
            </svg>
            <span className="logo-text">FloorPlan <span className="logo-ai">AI</span></span>
          </div>
          <nav className="nav-links">
            <a href="https://github.com/Janavee01/floorplan" target="_blank" rel="noreferrer" className="nav-link">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/></svg>
              GitHub
            </a>
          </nav>
        </div>
      </header>

      <main className="main">
        {/* Hero */}
        <section className="hero">
          <p className="hero-eyebrow">Computer Vision Pipeline</p>
          <h1 className="hero-title">
            2D Floor Plan<br />
            <span className="hero-accent">→ 3D Reality</span>
          </h1>
          <p className="hero-sub">
            Upload any architectural floor plan. Get colour-coded room segmentation,
            intelligent furniture placement, and a photorealistic 3D isometric render.
          </p>
        </section>

        <div className="workspace">
          {/* Left: Upload + Controls */}
          <div className="left-col">
            <div
              className={`dropzone ${dragging ? "drag-over" : ""} ${preview ? "has-preview" : ""} ${isProcessing ? "locked" : ""}`}
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onClick={() => !isProcessing && !preview && fileRef.current.click()}
            >
              <input
                ref={fileRef}
                type="file"
                accept="image/jpeg,image/png"
                style={{ display: "none" }}
                onChange={(e) => handleFile(e.target.files[0])}
              />
              {preview ? (
                <div className="preview-container">
                  <img src={preview} alt="Floor plan preview" className="preview-img" />
                  <div className="preview-overlay">
                    <p className="preview-filename">{file.name}</p>
                    <p className="preview-size">{(file.size / 1024).toFixed(0)} KB</p>
                    {!isProcessing && (
                      <button className="change-btn" onClick={(e) => { e.stopPropagation(); reset(); }}>
                        Change image
                      </button>
                    )}
                  </div>
                </div>
              ) : (
                <div className="dropzone-content">
                  <div className="upload-icon">
                    <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
                      <rect x="1" y="1" width="38" height="38" rx="4" stroke="var(--border)" strokeWidth="1.5" strokeDasharray="4 3"/>
                      <path d="M20 26V14M14 20l6-6 6 6" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <p className="drop-label">Drop your floor plan here</p>
                  <p className="drop-sub">or <span className="drop-link" onClick={() => fileRef.current.click()}>browse files</span></p>
                  <p className="drop-hint">JPG · PNG · Up to 20MB</p>
                </div>
              )}
            </div>

            {error && (
              <div className="error-box">
                <span className="error-icon">⚠</span>
                <p>{error}</p>
              </div>
            )}

            {/* Options */}
            <div className="options-card">
              <h3 className="options-title">Pipeline Options</h3>
              <label className="toggle-row">
                <div className="toggle-info">
                  <p className="toggle-label">3D Isometric Render</p>
                  <p className="toggle-desc">ControlNet + Stable Diffusion v1.5 — requires GPU, ~20 min</p>
                </div>
                <button
                  className={`toggle ${renderEnabled ? "on" : "off"}`}
                  onClick={() => setRenderEnabled(v => !v)}
                  disabled={isProcessing}
                  aria-label="Toggle 3D render"
                >
                  <span className="toggle-thumb" />
                </button>
              </label>
            </div>

            {/* Run button */}
            {status !== "done" && (
              <button
                className={`run-btn ${!file || isProcessing ? "disabled" : ""}`}
                onClick={handleSubmit}
                disabled={!file || isProcessing}
              >
                {isProcessing ? (
                  <span className="btn-processing">
                    <span className="spinner" />
                    Processing…
                  </span>
                ) : "Run Pipeline →"}
              </button>
            )}

            {status === "done" && (
              <button className="run-btn reset" onClick={reset}>
                ← Process Another
              </button>
            )}
          </div>

          {/* Right: Pipeline + Results */}
          <div className="right-col">
            {/* Pipeline tracker */}
            <div className="pipeline-card">
              <h3 className="pipeline-title">Pipeline</h3>
              <div className="stages">
                {PIPELINE_STAGES.map((s, i) => {
                  const isDone = completedStages.includes(s.id);
                  const isActive = currentStage === i && isProcessing;
                  const isPending = !isDone && !isActive;
                  return (
                    <div key={s.id} className={`stage ${isDone ? "done" : isActive ? "active" : "pending"}`}>
                      <div className="stage-indicator">
                        {isDone ? (
                          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                            <circle cx="8" cy="8" r="7" fill="var(--accent)" opacity="0.15"/>
                            <path d="M5 8l2 2 4-4" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                        ) : isActive ? (
                          <span className="stage-spinner" />
                        ) : (
                          <span className="stage-dot" />
                        )}
                        {i < PIPELINE_STAGES.length - 1 && <span className={`stage-line ${isDone ? "done" : ""}`} />}
                      </div>
                      <div className="stage-text">
                        <p className="stage-name">{s.label}</p>
                        <p className="stage-detail">{s.detail}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Results */}
            {status === "done" && results && (
              <div className="results-card">
                <h3 className="results-title">Results</h3>
                <div className="result-tabs">
                  {OUTPUT_TABS.filter(t => results[t.id]).map(t => (
                    <button
                      key={t.id}
                      className={`result-tab ${activeTab === t.id ? "active" : ""}`}
                      onClick={() => setActiveTab(t.id)}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
                <div className="result-image-wrap">
                  {results[activeTab] && (
                    <img
                      src={`/api/output/${jobId}/${OUTPUT_TABS.find(t => t.id === activeTab)?.file}`}
                      alt={activeTab}
                      className="result-img"
                    />
                  )}
                </div>
                <div className="download-row">
                  {OUTPUT_TABS.filter(t => results[t.id]).map(t => (
                    <a
                      key={t.id}
                      href={`/api/output/${jobId}/${t.file}`}
                      download={t.file}
                      className="download-btn"
                    >
                      ↓ {t.label}
                    </a>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Tech stack strip */}
        <section className="tech-strip">
          {["Tesseract OCR", "OpenCV 4.8", "Voronoi Segmentation", "PyTorch 2.0", "Stable Diffusion v1.5", "ControlNet"].map(t => (
            <span key={t} className="tech-pill">{t}</span>
          ))}
        </section>
      </main>

      <footer className="footer">
        <p>FloorPlan AI — Built by Janavee · <a href="https://github.com/Janavee01/floorplan" target="_blank" rel="noreferrer">View on GitHub</a></p>
      </footer>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
          --bg: #0a0c0f;
          --surface: #111316;
          --surface2: #16191e;
          --border: #222529;
          --border2: #2a2e35;
          --text: #e8eaed;
          --text2: #8b919c;
          --text3: #555b64;
          --accent: #7EFFA0;
          --accent-dim: rgba(126,255,160,0.12);
          --accent-glow: rgba(126,255,160,0.06);
          --red: #ff6b6b;
          --radius: 10px;
          --font: 'Syne', sans-serif;
          --mono: 'DM Mono', monospace;
        }

        html { background: var(--bg); color: var(--text); font-family: var(--font); }

        .app { min-height: 100vh; display: flex; flex-direction: column; position: relative; overflow-x: hidden; }

        .bg-grid {
          position: fixed; inset: 0; z-index: 0; pointer-events: none;
          background-image:
            linear-gradient(var(--border) 1px, transparent 1px),
            linear-gradient(90deg, var(--border) 1px, transparent 1px);
          background-size: 48px 48px;
          opacity: 0.35;
          mask-image: radial-gradient(ellipse 80% 60% at 50% 0%, black 30%, transparent 100%);
        }

        /* Header */
        .header {
          position: sticky; top: 0; z-index: 100;
          background: rgba(10,12,15,0.85);
          backdrop-filter: blur(12px);
          border-bottom: 1px solid var(--border);
        }
        .header-inner {
          max-width: 1200px; margin: 0 auto;
          padding: 0 32px; height: 60px;
          display: flex; align-items: center; justify-content: space-between;
        }
        .logo { display: flex; align-items: center; gap: 10px; }
        .logo-text { font-size: 17px; font-weight: 700; letter-spacing: -0.3px; }
        .logo-ai { color: var(--accent); }
        .nav-links { display: flex; gap: 8px; }
        .nav-link {
          display: flex; align-items: center; gap: 6px;
          font-family: var(--mono); font-size: 12px; color: var(--text2);
          text-decoration: none; padding: 6px 12px;
          border: 1px solid var(--border2); border-radius: 6px;
          transition: color 0.2s, border-color 0.2s;
        }
        .nav-link:hover { color: var(--text); border-color: var(--accent); }

        /* Main */
        .main { flex: 1; max-width: 1200px; margin: 0 auto; width: 100%; padding: 0 32px 80px; position: relative; z-index: 1; }

        /* Hero */
        .hero { padding: 72px 0 56px; }
        .hero-eyebrow {
          font-family: var(--mono); font-size: 11px; font-weight: 500;
          letter-spacing: 2px; text-transform: uppercase;
          color: var(--accent); margin-bottom: 16px;
        }
        .hero-title {
          font-size: clamp(40px, 6vw, 72px); font-weight: 800;
          line-height: 1.05; letter-spacing: -2px;
          color: var(--text);
        }
        .hero-accent { color: var(--accent); }
        .hero-sub {
          margin-top: 20px; max-width: 520px;
          font-size: 16px; line-height: 1.65; color: var(--text2); font-weight: 400;
        }

        /* Workspace */
        .workspace {
          display: grid;
          grid-template-columns: 380px 1fr;
          gap: 24px;
          align-items: start;
        }

        /* Drop zone */
        .dropzone {
          border: 1px solid var(--border2);
          border-radius: var(--radius);
          background: var(--surface);
          aspect-ratio: 4/3;
          display: flex; align-items: center; justify-content: center;
          cursor: pointer;
          transition: border-color 0.2s, background 0.2s;
          overflow: hidden;
          position: relative;
        }
        .dropzone:hover:not(.has-preview):not(.locked) { border-color: var(--accent); background: var(--accent-glow); }
        .dropzone.drag-over { border-color: var(--accent); background: var(--accent-dim); }
        .dropzone.locked { cursor: default; }

        .dropzone-content { text-align: center; padding: 24px; }
        .upload-icon { margin-bottom: 16px; }
        .drop-label { font-size: 15px; font-weight: 600; color: var(--text); margin-bottom: 6px; }
        .drop-sub { font-size: 13px; color: var(--text2); margin-bottom: 12px; }
        .drop-link { color: var(--accent); cursor: pointer; text-decoration: underline; text-underline-offset: 3px; }
        .drop-hint { font-family: var(--mono); font-size: 11px; color: var(--text3); letter-spacing: 0.5px; }

        .preview-container { width: 100%; height: 100%; position: relative; }
        .preview-img { width: 100%; height: 100%; object-fit: contain; }
        .preview-overlay {
          position: absolute; bottom: 0; left: 0; right: 0;
          background: linear-gradient(transparent, rgba(0,0,0,0.85));
          padding: 24px 16px 14px;
        }
        .preview-filename { font-size: 12px; font-weight: 600; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .preview-size { font-family: var(--mono); font-size: 11px; color: var(--text2); margin-top: 2px; }
        .change-btn {
          margin-top: 8px; padding: 4px 10px;
          font-family: var(--mono); font-size: 11px; color: var(--text2);
          background: rgba(255,255,255,0.08); border: 1px solid var(--border2);
          border-radius: 4px; cursor: pointer; transition: color 0.2s;
        }
        .change-btn:hover { color: var(--text); }

        /* Error */
        .error-box {
          margin-top: 10px; padding: 10px 14px;
          background: rgba(255,107,107,0.08); border: 1px solid rgba(255,107,107,0.3);
          border-radius: 6px; display: flex; gap: 8px; align-items: flex-start;
        }
        .error-icon { color: var(--red); font-size: 14px; flex-shrink: 0; margin-top: 1px; }
        .error-box p { font-size: 13px; color: var(--red); line-height: 1.4; }

        /* Options */
        .options-card {
          margin-top: 14px;
          padding: 16px 18px;
          background: var(--surface);
          border: 1px solid var(--border2);
          border-radius: var(--radius);
        }
        .options-title { font-size: 11px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: var(--text3); margin-bottom: 14px; font-family: var(--mono); }
        .toggle-row { display: flex; align-items: center; justify-content: space-between; gap: 16px; cursor: pointer; }
        .toggle-info { flex: 1; }
        .toggle-label { font-size: 14px; font-weight: 600; color: var(--text); margin-bottom: 3px; }
        .toggle-desc { font-size: 12px; color: var(--text2); line-height: 1.4; font-family: var(--mono); }
        .toggle {
          flex-shrink: 0; width: 42px; height: 24px;
          border-radius: 12px; border: none; cursor: pointer;
          position: relative; transition: background 0.25s;
        }
        .toggle.off { background: var(--border2); }
        .toggle.on { background: var(--accent); }
        .toggle:disabled { opacity: 0.5; cursor: default; }
        .toggle-thumb {
          position: absolute; top: 3px; width: 18px; height: 18px;
          background: white; border-radius: 50%;
          transition: left 0.25s;
          left: var(--thumb-x, 3px);
        }
        .toggle.off .toggle-thumb { left: 3px; }
        .toggle.on .toggle-thumb { left: 21px; }

        /* Run button */
        .run-btn {
          margin-top: 14px; width: 100%; padding: 14px;
          background: var(--accent); color: #0a0c0f;
          font-family: var(--font); font-size: 15px; font-weight: 700;
          border: none; border-radius: var(--radius); cursor: pointer;
          letter-spacing: -0.3px;
          transition: opacity 0.2s, transform 0.1s;
        }
        .run-btn:hover:not(.disabled) { opacity: 0.9; transform: translateY(-1px); }
        .run-btn.disabled { opacity: 0.35; cursor: not-allowed; }
        .run-btn.reset { background: var(--surface2); color: var(--text); border: 1px solid var(--border2); }
        .run-btn.reset:hover { border-color: var(--accent); color: var(--accent); }
        .btn-processing { display: flex; align-items: center; justify-content: center; gap: 10px; }
        .spinner {
          width: 16px; height: 16px; border-radius: 50%;
          border: 2px solid rgba(10,12,15,0.3);
          border-top-color: #0a0c0f;
          animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Pipeline card */
        .pipeline-card {
          background: var(--surface); border: 1px solid var(--border2);
          border-radius: var(--radius); padding: 20px 22px;
          margin-bottom: 20px;
        }
        .pipeline-title { font-size: 11px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: var(--text3); margin-bottom: 18px; font-family: var(--mono); }
        .stages { display: flex; flex-direction: column; gap: 0; }
        .stage { display: flex; gap: 14px; }
        .stage-indicator { display: flex; flex-direction: column; align-items: center; flex-shrink: 0; padding-top: 2px; }
        .stage-line { width: 1px; flex: 1; min-height: 14px; background: var(--border2); margin: 4px 0; }
        .stage-line.done { background: var(--accent); opacity: 0.4; }
        .stage-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--border2); }
        .stage-spinner {
          width: 14px; height: 14px; border-radius: 50%;
          border: 2px solid var(--accent-dim); border-top-color: var(--accent);
          animation: spin 0.8s linear infinite;
          display: block;
        }
        .stage-text { padding-bottom: 16px; }
        .stage:last-child .stage-text { padding-bottom: 0; }
        .stage-name { font-size: 13px; font-weight: 600; color: var(--text2); transition: color 0.2s; }
        .stage.done .stage-name { color: var(--text); }
        .stage.active .stage-name { color: var(--accent); }
        .stage-detail { font-family: var(--mono); font-size: 11px; color: var(--text3); margin-top: 2px; }
        .stage.active .stage-detail { color: var(--text2); }

        /* Results */
        .results-card {
          background: var(--surface); border: 1px solid var(--border2);
          border-radius: var(--radius); overflow: hidden;
        }
        .results-title { font-size: 11px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: var(--text3); padding: 18px 22px 0; font-family: var(--mono); }
        .result-tabs { display: flex; gap: 4px; padding: 14px 22px 0; border-bottom: 1px solid var(--border); }
        .result-tab {
          padding: 6px 14px 10px; font-family: var(--font); font-size: 13px; font-weight: 600;
          color: var(--text2); background: none; border: none; cursor: pointer;
          border-bottom: 2px solid transparent; margin-bottom: -1px;
          transition: color 0.15s, border-color 0.15s;
        }
        .result-tab:hover { color: var(--text); }
        .result-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
        .result-image-wrap { padding: 16px 22px; }
        .result-img { width: 100%; border-radius: 6px; display: block; border: 1px solid var(--border); }
        .download-row { display: flex; flex-wrap: wrap; gap: 8px; padding: 0 22px 18px; }
        .download-btn {
          font-family: var(--mono); font-size: 11px; color: var(--text2);
          text-decoration: none; padding: 6px 12px;
          border: 1px solid var(--border2); border-radius: 5px;
          transition: color 0.15s, border-color 0.15s;
        }
        .download-btn:hover { color: var(--accent); border-color: var(--accent); }

        /* Tech strip */
        .tech-strip { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 56px; padding-top: 32px; border-top: 1px solid var(--border); }
        .tech-pill {
          font-family: var(--mono); font-size: 11px; color: var(--text3);
          padding: 5px 11px; border: 1px solid var(--border); border-radius: 20px;
          letter-spacing: 0.3px;
        }

        /* Footer */
        .footer { padding: 24px 32px; border-top: 1px solid var(--border); text-align: center; position: relative; z-index: 1; }
        .footer p { font-family: var(--mono); font-size: 12px; color: var(--text3); }
        .footer a { color: var(--text2); text-decoration: none; }
        .footer a:hover { color: var(--accent); }

        /* Responsive */
        @media (max-width: 900px) {
          .workspace { grid-template-columns: 1fr; }
          .hero { padding: 48px 0 36px; }
          .main { padding: 0 20px 60px; }
          .header-inner { padding: 0 20px; }
        }
      `}</style>
    </div>
  );
}
