(function () {
  var root = document.getElementById("dashboard-live");
  if (!root) return;

  var status = document.getElementById("dashboard-refresh");
  var refreshUrl = root.dataset.refreshUrl;
  var hasData = root.dataset.hasData === "true";
  var initialAnalyzed = Number(root.dataset.analyzed || 0);
  var charts = {};

  function createCharts(data) {
    if (typeof Chart === "undefined") return;

    var css = getComputedStyle(document.documentElement);
    var text = css.getPropertyValue("--muted").trim() || "#666";
    var grid = css.getPropertyValue("--border").trim() || "#ddd";
    var palette = ["#4f6df5", "#e5484d", "#f5a524", "#12a594", "#8e4ec6", "#6b7280", "#0ea5e9", "#d6409f"];
    var severityColors = ["#c62828", "#ef6c00", "#f9a825", "#5c8fd6"];
    var options = { responsive: true, maintainAspectRatio: false };

    Chart.defaults.color = text;
    Chart.defaults.font.family = "inherit";
    charts.scores = new Chart(document.getElementById("c-scores"), {
      type: "bar",
      data: { labels: data.scores.labels, datasets: [{ data: data.scores.values, backgroundColor: "#4f6df5", borderRadius: 6 }] },
      options: Object.assign({}, options, { plugins: { legend: { display: false } },
        scales: { y: { min: 0, max: 100, grid: { color: grid } }, x: { grid: { display: false } } } })
    });
    charts.categories = new Chart(document.getElementById("c-categories"), {
      type: "doughnut",
      data: { labels: data.categories.labels, datasets: [{ data: data.categories.values, backgroundColor: palette, borderWidth: 0 }] },
      options: Object.assign({}, options, { plugins: { legend: { position: "right" } }, cutout: "62%" })
    });
    charts.severity = new Chart(document.getElementById("c-severity"), {
      type: "bar",
      data: { labels: data.severity.labels, datasets: [{ data: data.severity.values, backgroundColor: severityColors, borderRadius: 6 }] },
      options: Object.assign({}, options, { plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, grid: { color: grid }, ticks: { precision: 0 } }, x: { grid: { display: false } } } })
    });
  }

  function updateCharts(data) {
    Object.keys(charts).forEach(function (name) {
      charts[name].data.labels = data[name].labels;
      charts[name].data.datasets[0].data = data[name].values;
      charts[name].update("none");
    });
  }

  function updateMetrics(data) {
    Object.keys(data).forEach(function (name) {
      var value = document.querySelector('[data-metric="' + name + '"]');
      if (!value) return;
      value.textContent = data[name] === null ? "—" : data[name];
      if (name === "avg_score" || name === "avg_performance_score" || name === "avg_documentation_score") {
        var score = Number(data[name]);
        var scoreClass = data[name] === null ? "na" : score >= 80 ? "good" : score >= 60 ? "ok" : "bad";
        value.classList.remove("good", "ok", "bad", "na");
        value.classList.add(scoreClass);
        var bar = document.querySelector('[data-score-metric="' + name + '"]');
        if (bar) {
          bar.classList.remove("good", "ok", "bad", "na");
          bar.classList.add(scoreClass);
          bar.style.width = (Number.isFinite(score) ? Math.max(0, Math.min(100, score)) : 0) + "%";
        }
      }
    });
  }

  document.querySelectorAll(".bar i[data-progress]").forEach(function (bar) {
    var progress = Number(bar.dataset.progress);
    bar.style.width = (Number.isFinite(progress) ? Math.max(0, Math.min(100, progress)) : 0) + "%";
  });

  var chartData = document.getElementById("chart-data");
  if (chartData && hasData) createCharts(JSON.parse(chartData.textContent));

  function refresh() {
    fetch(refreshUrl, { credentials: "same-origin", headers: { "Accept": "application/json" } })
      .then(function (response) {
        return response.json().then(function (data) {
          if (!response.ok) throw new Error(data.error || "Server returned " + response.status);
          return data;
        });
      })
      .then(function (data) {
        if (data.has_data !== hasData || data.analyzed !== initialAnalyzed) {
          window.location.reload();
          return;
        }
        updateMetrics(data);
        updateCharts(data.chart_data);
        status.textContent = "Live data · updated " + new Date().toLocaleTimeString();
        status.classList.remove("dashboard-refresh-error");
      })
      .catch(function (error) {
        status.textContent = "Live update failed: " + error.message + " · retrying";
        status.classList.add("dashboard-refresh-error");
      })
      .finally(function () {
        window.setTimeout(refresh, 15000);
      });
  }

  window.setTimeout(refresh, 15000);
})();
