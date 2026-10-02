/* =========================================================
   m3.js · gráficas del Módulo 3 (no supervisado)
   ========================================================= */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const K = [1, 2, 3, 4, 5, 6, 7, 8];
  // Old Faithful (272 erupciones, variables estandarizadas): resultados de m3_02_kmeans.py
  const INERTIA = [544.0, 79.6, 56.3, 43.9, 34.3, 27.3, 23.8, 20.8];
  const SIL = [null, 0.745, 0.485, 0.388, 0.365, 0.393, 0.387, 0.361];   // la silueta no existe con k = 1

  function lineK(id, ks, values, label, color, yTitle, best) {
    if (!$(id)) return;
    Viz.chart(id, (P) => ({
      type: "line",
      data: { labels: ks, datasets: [{ label, data: values, borderColor: P.s[color], backgroundColor: P.s[color], borderWidth: 2,
        pointRadius: ks.map((k) => (k === best ? 6 : 3.5)), pointBorderColor: P.surface, pointBorderWidth: 2, pointHoverRadius: 6 }] },
      options: Viz.baseOptions(P, {
        plugins: { tooltip: { callbacks: { title: (it) => "k = " + it[0].label, label: (c) => ` ${label}: ${c.parsed.y}` } } },
        scales: { x: { title: { display: true, text: "número de clusters k" } }, y: { title: { display: true, text: yTitle }, beginAtZero: true } },
      }),
    }));
  }

  /* ---------- Old Faithful: los dos grupos de K-means (k = 2) ---------- */
  function faithful() {
    if (!$("chart-faithful") || !window.DATA || !window.DATA.faithful) return;
    const d = window.DATA.faithful;
    Viz.chart("chart-faithful", (P) => ({
      type: "scatter",
      data: { datasets: [0, 1].map((k) => ({
        label: k === 0 ? "erupciones cortas" : "erupciones largas",
        data: d.x.map((x, i) => (d.k[i] === k ? { x, y: d.y[i] } : null)).filter(Boolean),
        backgroundColor: Viz.alpha(P.s[k], 0.8), borderColor: P.surface, borderWidth: 0.5, pointRadius: 3.5, pointHoverRadius: 6,
      })) },
      options: Viz.baseOptions(P, {
        plugins: { tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: duración ${c.parsed.x.toFixed(2)} min, espera ${c.parsed.y} min` } } },
        scales: { x: { type: "linear", min: 1.5, max: 5.5, title: { display: true, text: "duración de la erupción (min)" } },
          y: { min: 40, max: 100, title: { display: true, text: "espera hasta la siguiente (min)" } } },
      }),
    }));
  }

  /* ---------- Sismos de Fiyi agrupados con DBSCAN (ε = 2°, min_samples = 10) ---------- */
  function fiji() {
    if (!$("chart-fiji") || !window.DATA || !window.DATA.fiji) return;
    const d = window.DATA.fiji;
    Viz.chart("chart-fiji", (P) => {
      const groups = [{ k: 0, name: "zona oriental", c: P.s[0] }, { k: 1, name: "zona occidental", c: P.s[1] }, { k: -1, name: "ruido", c: P.muted }];
      return {
        type: "scatter",
        data: { datasets: groups.map((g) => ({
          label: g.name, data: d.x.map((x, i) => (d.db[i] === g.k ? { x, y: d.y[i], z: d.z[i], m: d.m[i] } : null)).filter(Boolean),
          backgroundColor: g.k === -1 ? g.c : Viz.alpha(g.c, 0.7), borderColor: P.surface, borderWidth: g.k === -1 ? 2 : 0.5,
          pointRadius: g.k === -1 ? 6 : 3, pointHoverRadius: 6,
        })) },
        options: Viz.baseOptions(P, {
          plugins: { tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${Math.abs(c.raw.y).toFixed(2)}° S, ${c.raw.x.toFixed(2)}° E · ${c.raw.z} km de profundidad · magnitud ${c.raw.m.toFixed(1)}` } } },
          scales: { x: { type: "linear", min: 164, max: 190, title: { display: true, text: "longitud (grados al este de Greenwich)" } },
            y: { min: -40, max: -9, title: { display: true, text: "latitud (grados; negativo = sur)" } } },
        }),
      };
    });
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
          plugins: { title: { display: true, text: key === "km" ? "K-means (k = 2), datos sintéticos" : "DBSCAN (ε = 0.25), datos sintéticos", color: P.ink1, font: { size: 12, weight: "600" } },
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
        { label: "distancia al 10.º vecino", data: d.d, borderColor: P.s[2], backgroundColor: P.s[2], borderWidth: 2, pointRadius: 0, pointHoverRadius: 4 },
        { label: "ε = 2°", data: d.d.map(() => 2), borderColor: P.ink3, borderWidth: 1, pointRadius: 0 },
      ] },
      options: Viz.baseOptions(P, {
        interaction: { mode: "index", intersect: false },
        plugins: { tooltip: { filter: (i) => i.datasetIndex === 0, callbacks: { title: (it) => "sismo n.º " + it[0].label + " (ordenado)", label: (c) => ` distancia al 10.º vecino: ${c.parsed.y.toFixed(2)}°` } } },
        scales: { x: { title: { display: true, text: "sismos ordenados por distancia" }, ticks: { maxTicksLimit: 8 } }, y: { min: 0, max: 3.5, title: { display: true, text: "distancia (grados)" } } },
      }),
    }));
  }

  document.addEventListener("DOMContentLoaded", () => {
    try {
      lineK("chart-elbow", K, INERTIA, "inercia", 0, "inercia (WCSS)", 2);
      lineK("chart-sil", K.slice(1), SIL.slice(1), "silueta", 2, "silueta promedio", 2);
      faithful();
      fiji();
      moons("chart-moons-km", "km");
      moons("chart-moons-db", "db");
      kdist();
    } catch (e) { console.error(e); }
  });
})();
