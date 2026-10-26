(function () {
  var el = document.getElementById("chart-data");
  if (!el || typeof Chart === "undefined") return;
  var d = JSON.parse(el.textContent);
  var css = getComputedStyle(document.documentElement);
  var text = css.getPropertyValue("--muted").trim() || "#666";
  var grid = css.getPropertyValue("--border").trim() || "#ddd";
  Chart.defaults.color = text;
  Chart.defaults.font.family = "inherit";
  var palette = ["#4f6df5", "#e5484d", "#f5a524", "#12a594", "#8e4ec6", "#6b7280", "#0ea5e9", "#d6409f"];
  var sevColors = ["#c62828", "#ef6c00", "#f9a825", "#5c8fd6"];
  var opts = { responsive: true, maintainAspectRatio: false };

  new Chart(document.getElementById("c-scores"), {
    type: "bar",
    data: { labels: d.scores.labels, datasets: [{ data: d.scores.values, backgroundColor: "#4f6df5", borderRadius: 6 }] },
    options: Object.assign({}, opts, { plugins: { legend: { display: false } },
      scales: { y: { min: 0, max: 100, grid: { color: grid } }, x: { grid: { display: false } } } })
  });
  new Chart(document.getElementById("c-categories"), {
    type: "doughnut",
    data: { labels: d.categories.labels, datasets: [{ data: d.categories.values, backgroundColor: palette, borderWidth: 0 }] },
    options: Object.assign({}, opts, { plugins: { legend: { position: "right" } }, cutout: "62%" })
  });
  new Chart(document.getElementById("c-severity"), {
    type: "bar",
    data: { labels: d.severity.labels, datasets: [{ data: d.severity.values, backgroundColor: sevColors, borderRadius: 6 }] },
    options: Object.assign({}, opts, { plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, grid: { color: grid }, ticks: { precision: 0 } }, x: { grid: { display: false } } } })
  });
  new Chart(document.getElementById("c-languages"), {
    type: "bar",
    data: { labels: d.languages.labels, datasets: [{ data: d.languages.values, backgroundColor: "#12a594", borderRadius: 6 }] },
    options: Object.assign({}, opts, { indexAxis: "y", plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true, max: 100, grid: { color: grid } }, y: { grid: { display: false } } } })
  });
})();
