/* =========================================================
   m3.js · gráficas del Módulo 3 (no supervisado)
   ========================================================= */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const K = [2, 3, 4, 5, 6, 7, 8];
  const INERTIA = [490.8, 129.8, 31.1, 27.9, 25.0, 21.9, 19.0];
  const SIL = [0.593, 0.766, 0.802, 0.683, 0.580, 0.450, 0.355];

  function lineK(id, values, label, color, yTitle, best) {
    if (!$(id)) return;
    Viz.chart(id, (P) => ({
      type: "line",
      data: { labels: K, datasets: [{ label, data: values, borderColor: P.s[color], backgroundColor: P.s[color], borderWidth: 2,
        pointRadius: K.map((k) => (k === best ? 6 : 3.5)), pointBorderColor: P.surface, pointBorderWidth: 2, pointHoverRadius: 6 }] },
      options: Viz.baseOptions(P, {
        plugins: { tooltip: { callbacks: { title: (it) => "k = " + it[0].label, label: (c) => ` ${label}: ${c.parsed.y}` } } },
        scales: { x: { title: { display: true, text: "número de clusters k" } }, y: { title: { display: true, text: yTitle }, beginAtZero: true } },
      }),
    }));
  }

  function moons(id, key) {
    if (!$(id) || !window.DATA) return;
    const d = window.DATA.moons, lab = d[key];
    Viz.chart(id, (P) => {
      const groups = [{ k: 0, name: "cluster 0", c: P.s[0] }, { k: 1, name: "cluster 1", c: P.s[1] }, { k: -1, name: "ruido", c: P.muted }];
      return {
        type: "scatter",
        data: { datasets: groups.map((g) => ({
          label: g.name, data: d.x.map((x, i) => (lab[i] === g.k ? { x, y: d.y[i] } : null)).filter(Boolean),
          backgroundColor: g.c, borderColor: P.surface, borderWidth: g.k === -1 ? 2 : 0.5, pointRadius: g.k === -1 ? 6 : 3.2, pointHoverRadius: 6,
        })).filter((ds) => ds.data.length) },
        options: Viz.baseOptions(P, {
          plugins: { title: { display: true, text: key === "km" ? "K-means (k = 2)" : "DBSCAN (ε = 0.25)", color: P.ink1, font: { size: 12, weight: "600" } },
            tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: (${c.parsed.x.toFixed(2)}, ${c.parsed.y.toFixed(2)})` } } },
          scales: { x: { type: "linear", min: -2.2, max: 2.6, ticks: { maxTicksLimit: 6 } }, y: { min: -2.4, max: 2.8, ticks: { maxTicksLimit: 6 } } },
        }),
      };
    });
  }

  function kdist() {
    if (!$("chart-kdist") || !window.DATA) return;
    const d = window.DATA.kdist;
    Viz.chart("chart-kdist", (P) => ({
      type: "line",
      data: { labels: d.i, datasets: [
        { label: "5-distancia", data: d.d, borderColor: P.s[2], backgroundColor: P.s[2], borderWidth: 2, pointRadius: 0, pointHoverRadius: 4 },
        { label: "ε = 0.25", data: d.d.map(() => 0.25), borderColor: P.ink3, borderWidth: 1, pointRadius: 0 },
      ] },
      options: Viz.baseOptions(P, {
        interaction: { mode: "index", intersect: false },
        plugins: { tooltip: { filter: (i) => i.datasetIndex === 0, callbacks: { title: (it) => "punto #" + it[0].label + " (ordenado)", label: (c) => ` distancia al 5.º vecino: ${c.parsed.y.toFixed(3)}` } } },
        scales: { x: { title: { display: true, text: "puntos ordenados por distancia" }, ticks: { maxTicksLimit: 8 } }, y: { min: 0, max: 0.8, title: { display: true, text: "distancia (los 3 atípicos se salen por arriba)" } } },
      }),
    }));
  }

  document.addEventListener("DOMContentLoaded", () => {
    try {
      lineK("chart-elbow", INERTIA, "inercia", 0, "inercia (WCSS)", 4);
      lineK("chart-sil", SIL, "silueta", 2, "silueta promedio", 4);
      moons("chart-moons-km", "km");
      moons("chart-moons-db", "db");
      kdist();
    } catch (e) { console.error(e); }
  });
})();
