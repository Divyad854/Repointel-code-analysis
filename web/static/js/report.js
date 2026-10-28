// Progress polling for running analyses + severity/category filters for finished reports.
(function () {
  document.querySelectorAll(".ring[data-score]").forEach(function (ring) {
    var score = Number(ring.dataset.score);
    if (Number.isFinite(score)) {
      ring.style.setProperty("--p", String(Math.max(0, Math.min(100, score))));
    }
  });
  document.querySelectorAll(".bar i[data-progress]").forEach(function (bar) {
    var progress = Number(bar.dataset.progress);
    if (Number.isFinite(progress)) {
      bar.style.width = Math.max(0, Math.min(100, progress)) + "%";
    }
  });

  var box = document.getElementById("progress");
  if (box) {
    var url = box.dataset.statusUrl;
    var warn = document.getElementById("progress-warning");
    var failures = 0;
    function tick() {
      fetch(url, { headers: { "Accept": "application/json" }, credentials: "same-origin" })
        .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
        .then(function (d) {
          failures = 0;
          (d.done_steps || []).forEach(function (k) {
            var li = box.querySelector('li[data-key="' + k + '"]');
            if (li) li.classList.add("done");
          });
          if (d.warning) { warn.textContent = d.warning; warn.hidden = false; } else { warn.hidden = true; }
          if (d.finished) { window.location.reload(); return; }
          setTimeout(tick, 2000);
        })
        .catch(function () {
          failures += 1;
          if (failures > 5) { warn.textContent = "Lost connection to the server. Refresh the page to retry."; warn.hidden = false; return; }
          setTimeout(tick, 4000);
        });
    }
    setTimeout(tick, 1500);
  }

  var sevBtns = document.querySelectorAll("#sev-filter button");
  var catSel = document.getElementById("cat-filter");
  if (sevBtns.length && catSel) {
    var sev = "all";
    function apply() {
      var shown = 0;
      document.querySelectorAll(".finding").forEach(function (el) {
        var ok = (sev === "all" || el.dataset.sev === sev) && (catSel.value === "all" || el.dataset.cat === catSel.value);
        el.hidden = !ok;
        if (ok) shown += 1;
      });
      var none = document.getElementById("no-match");
      if (none) none.hidden = shown !== 0 || !document.querySelector(".finding");
    }
    sevBtns.forEach(function (b) {
      b.addEventListener("click", function () {
        sevBtns.forEach(function (x) { x.classList.remove("on"); });
        b.classList.add("on");
        sev = b.dataset.sev;
        apply();
      });
    });
    catSel.addEventListener("change", apply);
  }
})();
