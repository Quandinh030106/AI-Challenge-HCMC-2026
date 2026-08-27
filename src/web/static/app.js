// ==============================================================================
// AIC 2026 - FRONTEND INTERACTION LOGIC & SEARCH ENGINE CONTROLLER
// ==============================================================================

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("search-form");
  const taskTabs = document.querySelectorAll(".task-tab");
  const taskTypeInput = document.getElementById("task_type");
  const loadingState = document.getElementById("loading-state");
  const resultsGrid = document.getElementById("results-grid");
  const metricsPills = document.getElementById("metrics-pills");
  const metricTime = document.getElementById("metric-time");
  const metricCount = document.getElementById("metric-count");
  const metricIntent = document.getElementById("metric-intent");
  const debugChips = document.getElementById("debug-chips");

  // Task Selector Tabs
  taskTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      taskTabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      const taskVal = tab.getAttribute("data-task");
      taskTypeInput.value = taskVal;

      const qidInput = document.getElementById("query_id");
      if (qidInput) {
        qidInput.value = `query-p1-1-${taskVal}`;
      }
    });
  });

  // Form Submit Execution
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);

    // Reset UI State
    resultsGrid.innerHTML = "";
    loadingState.style.display = "flex";
    metricsPills.style.display = "none";
    debugChips.style.display = "none";

    try {
      const response = await fetch("/api/v1/search", {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      loadingState.style.display = "none";

      if (data.status === "success") {
        renderResults(data);
      } else {
        alert("Lỗi tìm kiếm: " + (data.detail || "Không thể thực thi."));
      }
    } catch (err) {
      loadingState.style.display = "none";
      alert("Lỗi kết nối Server: " + err.message);
    }
  });

  function renderResults(data) {
    // Render Top Bar Metrics
    metricsPills.style.display = "flex";
    metricTime.textContent = `${data.elapsed_seconds}s`;
    metricCount.textContent = data.total_results;
    metricIntent.textContent = data.parsed_schema.intent || "VISUAL_SCENE";

    // Render Debug Chips
    debugChips.innerHTML = "";
    debugChips.style.display = "flex";

    const prompts = data.parsed_schema.golden_english_prompts || [];
    prompts.forEach(p => {
      const chip = document.createElement("span");
      chip.className = "chip prompt";
      chip.textContent = `Prompt: "${p}"`;
      debugChips.appendChild(chip);
    });

    const keywords = data.parsed_schema.bm25_keywords || [];
    keywords.forEach(k => {
      const chip = document.createElement("span");
      chip.className = "chip keyword";
      chip.textContent = `BM25: ${k}`;
      debugChips.appendChild(chip);
    });

    const objects = data.parsed_schema.openimages_classes || [];
    objects.forEach(o => {
      const chip = document.createElement("span");
      chip.className = "chip object";
      chip.textContent = `Obj: ${o}`;
      debugChips.appendChild(chip);
    });

    // Render Candidate Video Cards
    resultsGrid.innerHTML = "";

    data.results.forEach(res => {
      const card = document.createElement("div");
      card.className = "card";

      const fallbackImg = `https://images.unsplash.com/photo-1518173946687-a4c8a383392e?w=600&auto=format&fit=crop&q=80`;

      card.innerHTML = `
        <div class="card-header">
          <span class="card-rank">#${res.rank}</span>
          <span class="card-video-id">${res.video_id}</span>
          <span class="card-score">Score: ${res.score}</span>
        </div>
        <div class="media-preview">
          <img src="${res.image_path || fallbackImg}" class="keyframe-img" alt="${res.video_id}" onerror="this.src='${fallbackImg}'">
          <span class="time-badge">⏱ ${res.timestamp_formatted} (${res.pts_time.toFixed(1)}s)</span>
        </div>
        <div class="card-body">
          <div class="submission-meta">
            <span class="label">Mã Nộp BTC (Frame ID):</span>
            <span class="value">${res.frame_id}</span>
          </div>
          ${res.answer ? `<div class="submission-meta"><span class="label">Đáp án Q&A:</span><span class="value">${res.answer}</span></div>` : ''}
          <p class="card-caption">${res.keyframe_caption || 'Khung hình ứng viên có độ tương đồng cao.'}</p>
          <div class="card-actions">
            <a href="${res.youtube_url}" target="_blank" class="btn-play-yt">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
              <span>Phát Đúng Mốc thời gian (${res.timestamp_formatted})</span>
            </a>
          </div>
        </div>
      `;

      resultsGrid.appendChild(card);
    });
  }
});
