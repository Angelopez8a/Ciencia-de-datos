/* =========================================================
   m2.js · gráficas del Módulo 2 (modelación supervisada)
   ========================================================= */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);

  /* ---------- Residuales: homocedasticidad vs heterocedasticidad ---------- */
  function residuals() {
    const make = (id, hetero) => {
      if (!$(id)) return;
      const r = Viz.rng(hetero ? 7 : 3), pts = [];
      for (let i = 0; i < 220; i++) { const x = r() * 10, fit = 3 + 2 * x; pts.push({ x: +fit.toFixed(3), y: +(r.normal() * (hetero ? 0.45 * x + 0.1 : 1.2)).toFixed(3) }); }
      Viz.chart(id, (P) => ({
        type: "scatter",
        data: { datasets: [
          { label: "residual", data: pts, backgroundColor: Viz.alpha(P.s[hetero ? 1 : 0], 0.75), pointRadius: 3, pointHoverRadius: 5 },
          { label: "cero", data: [{ x: 3, y: 0 }, { x: 23, y: 0 }], type: "line", borderColor: P.ink3, borderWidth: 1, pointRadius: 0 },
        ] },
        options: Viz.baseOptions(P, {
          plugins: { title: { display: true, text: hetero ? "Heterocedástico (cono)" : "Homocedástico", color: P.ink1, font: { size: 12, weight: "600" } },
            tooltip: { filter: (i) => i.datasetIndex === 0, callbacks: { label: (c) => ` ajustado ${c.parsed.x.toFixed(2)}, residual ${c.parsed.y.toFixed(2)}` } } },
          scales: { x: { type: "linear", min: 3, max: 23, title: { display: true, text: "valor ajustado ŷ" } }, y: { min: -10, max: 10, title: { display: true, text: "residual" } } },
        }),
      }));
    };
    make("chart-res-ok", false); make("chart-res-bad", true);
  }

  /* ---------- Curva logística ---------- */
  function logit() {
    if (!$("chart-logit")) return;
    const h = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6], y = [0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1];
    const b0 = -4.632, b1 = 1.425, curve = [];
    for (let x = 0; x <= 7.0001; x += 0.1) curve.push({ x: +x.toFixed(2), y: 1 / (1 + Math.exp(-(b0 + b1 * x))) });
    Viz.chart("chart-logit", (P) => ({
      type: "scatter",
      data: { datasets: [
        { label: "P(aprobar)", data: curve, type: "line", borderColor: P.s[0], backgroundColor: P.s[0], borderWidth: 2, pointRadius: 0, pointHoverRadius: 4 },
        { label: "observación", data: h.map((x, i) => ({ x, y: y[i] })), backgroundColor: P.s[1], borderColor: P.surface, borderWidth: 2, pointRadius: 6, pointHoverRadius: 8 },
      ] },
      options: Viz.baseOptions(P, {
        plugins: { tooltip: { callbacks: { label: (c) => c.datasetIndex === 0 ? ` ${c.parsed.x.toFixed(1)} h → P = ${c.parsed.y.toFixed(3)}` : ` ${c.parsed.x} h → ${c.parsed.y ? "aprobó" : "reprobó"}` } } },
        scales: { x: { type: "linear", min: 0, max: 7, title: { display: true, text: "horas de estudio" } }, y: { min: -0.05, max: 1.05, title: { display: true, text: "probabilidad de aprobar" } } },
      }),
    }));
  }

  /* ---------- Camino de LASSO ---------- */
  function lasso() {
    if (!$("chart-lasso") || !window.DATA) return;
    const d = window.DATA.lasso, hi = { bmi: 0, s5: 1, bp: 2, s3: 6 };
    Viz.chart("chart-lasso", (P) => ({
      type: "line",
      data: { datasets: Object.entries(d.coefs).map(([name, c]) => ({
        label: name, data: d.alphas.map((a, i) => ({ x: a, y: c[i] })),
        borderColor: name in hi ? P.s[hi[name]] : P.muted, backgroundColor: name in hi ? P.s[hi[name]] : P.muted,
        borderWidth: name in hi ? 2 : 1.25, pointRadius: 0, pointHoverRadius: 4, order: name in hi ? 0 : 1,
      })) },
      options: Viz.baseOptions(P, {
        interaction: { mode: "nearest", intersect: false },
        plugins: { tooltip: { callbacks: { title: (it) => "α = " + it[0].parsed.x.toFixed(2), label: (c) => ` ${c.dataset.label}: ${c.parsed.y.toFixed(2)}` } } },
        scales: { x: { type: "logarithmic", min: 0.1, max: 60, title: { display: true, text: "α (escala log) → más penalización" }, ticks: { callback: (v) => ([0.1, 0.5, 1, 5, 10, 50].includes(+v.toFixed(1)) ? v : "") } },
          y: { title: { display: true, text: "coeficiente" } } },
      }),
    }));
  }

  /* ---------- Gradiente descendente: x por iteración ---------- */
  function gd() {
    if (!$("chart-gd")) return;
    const lrs = [0.1, 0.5, 0.9], K = 20;
    const series = lrs.map((lr) => { let x = 3; const s = [x]; for (let k = 0; k < K; k++) { x = x - lr * (2 * x - 2); s.push(x); } return s; });
    Viz.chart("chart-gd", (P) => ({
      type: "line",
      data: { labels: Array.from({ length: K + 1 }, (_, k) => k), datasets: series.map((s, i) => ({ label: "α = " + lrs[i], data: s, borderColor: [P.s[0], P.s[2], P.s[1]][i], backgroundColor: [P.s[0], P.s[2], P.s[1]][i], borderWidth: 2, pointRadius: 2.5, pointHoverRadius: 5 })) },
      options: Viz.baseOptions(P, {
        interaction: { mode: "index", intersect: false },
        plugins: { tooltip: { callbacks: { title: (it) => "iteración " + it[0].label, label: (c) => ` ${c.dataset.label}: x = ${c.parsed.y.toFixed(4)}` } } },
        scales: { x: { title: { display: true, text: "iteración" } }, y: { min: -0.8, max: 3.2, title: { display: true, text: "x (mínimo en 1)" } } },
      }),
    }));
  }

  /* ---------- Ensambles ---------- */
  function ens() {
    if (!$("chart-ens")) return;
    const D = [["Árbol individual", 0.924], ["Bosque aleatorio", 0.936], ["Bagging (100 árboles)", 0.942], ["Gradient Boosting", 0.942],
      ["Votación dura", 0.959], ["Votación suave", 0.959], ["AdaBoost", 0.959], ["Stacking", 0.959]];
    Viz.chart("chart-ens", (P) => ({
      type: "bar",
      data: { labels: D.map((d) => d[0]), datasets: [{ label: "accuracy", data: D.map((d) => d[1]), backgroundColor: D.map((d, i) => (i === 0 ? P.muted : P.s[0])), borderRadius: 4, borderSkipped: "start", maxBarThickness: 22 }] },
      options: Viz.baseOptions(P, {
        indexAxis: "y",
        plugins: { tooltip: { callbacks: { label: (c) => ` accuracy = ${c.parsed.x.toFixed(3)}` } } },
        scales: { x: { min: 0.9, max: 0.97, title: { display: true, text: "accuracy en prueba" } }, y: { grid: { display: false } } },
      }),
    }));
  }

  /* ---------- ROC ---------- */
  function roc() {
    if (!$("chart-roc") || !window.DATA) return;
    const d = window.DATA.roc;
    Viz.chart("chart-roc", (P) => ({
      type: "scatter",
      data: { datasets: [
        { label: "ROC", data: d.fpr.map((x, i) => ({ x, y: d.tpr[i] })), showLine: true, borderColor: P.s[0], backgroundColor: Viz.alpha(P.s[0], 0.1), fill: true, borderWidth: 2, pointRadius: 4, pointBackgroundColor: P.s[0], pointBorderColor: P.surface, pointBorderWidth: 2 },
        { label: "azar", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], showLine: true, borderColor: P.ink3, borderWidth: 1, pointRadius: 0 },
      ] },
      options: Viz.baseOptions(P, {
        plugins: { tooltip: { filter: (i) => i.datasetIndex === 0, callbacks: { label: (c) => { const t = d.thr[c.dataIndex]; return ` corte ${t == null ? "∞" : "≥ " + t}: FPR ${c.parsed.x.toFixed(2)}, TPR ${c.parsed.y.toFixed(2)}`; } } } },
        scales: { x: { type: "linear", min: 0, max: 1, title: { display: true, text: "FPR (1 − especificidad)" } }, y: { min: 0, max: 1.02, title: { display: true, text: "TPR (recall)" } } },
      }),
    }));
  }

  /* ---------- Lift ---------- */
  function lift() {
    if (!$("chart-lift")) return;
    const L = [2.68, 1.91, 1.38, 1.12, 0.73, 0.69, 0.4, 0.45, 0.27, 0.37];
    Viz.chart("chart-lift", (P) => ({
      type: "bar",
      data: { labels: L.map((_, i) => "D" + (i + 1)), datasets: [
        { type: "line", label: "azar (lift = 1)", data: L.map(() => 1), borderColor: P.ink3, borderWidth: 1, pointRadius: 0 },
        { label: "lift", data: L, backgroundColor: P.s[0], borderRadius: { topLeft: 4, topRight: 4 }, maxBarThickness: 24 },
      ] },
      options: Viz.baseOptions(P, {
        plugins: { tooltip: { filter: (i) => i.datasetIndex === 1, callbacks: { label: (c) => ` lift = ${c.parsed.y.toFixed(2)}` } } },
        scales: { x: { grid: { display: false }, title: { display: true, text: "decil (por score)" } }, y: { min: 0, max: 3, title: { display: true, text: "lift" } } },
      }),
    }));
  }

  /* ---------- PSI en el tiempo ---------- */
  function psi() {
    if (!$("chart-psi")) return;
    const yrs = Array.from({ length: 28 }, (_, i) => 1989 + i);
    const v = [0, 0.425, 0.052, 0.168, 0.148, 0.125, 0.068, 0.368, 0.135, 0.11, 0.075, 0.146, 0.069, 0.188, 0.047, 0.029, 0.162, 0.034, 0.01, 0.068, 0.038, 0.09, 0.017, 0.053, 0.103, 0.308, 0.398, 0.362];
    Viz.chart("chart-psi", (P) => ({
      type: "line",
      data: { labels: yrs, datasets: [
        { label: "PSI", data: v, borderColor: P.s[0], backgroundColor: P.s[0], borderWidth: 2, pointRadius: 3, pointBorderColor: P.surface, pointBorderWidth: 1.5, pointHoverRadius: 5 },
        { label: "control 0.10", data: yrs.map(() => 0.1), borderColor: P.warn, borderWidth: 1.5, pointRadius: 0 },
        { label: "control 0.25", data: yrs.map(() => 0.25), borderColor: P.crit, borderWidth: 1.5, pointRadius: 0 },
      ] },
      options: Viz.baseOptions(P, {
        interaction: { mode: "index", intersect: false },
        plugins: { tooltip: { filter: (i) => i.datasetIndex === 0, callbacks: { label: (c) => { const x = c.parsed.y; return ` PSI = ${x.toFixed(3)} → ${x < 0.1 ? "estable" : x <= 0.25 ? "cambio menor" : "cambio MAYOR"}`; } } } },
        scales: { x: { ticks: { maxTicksLimit: 10 } }, y: { min: 0, max: 0.45, title: { display: true, text: "PSI" } } },
      }),
    }));
  }

  document.addEventListener("DOMContentLoaded", () => {
    [residuals, logit, lasso, gd, ens, roc, lift, psi].forEach((fn) => { try { fn(); } catch (e) { console.error(fn.name, e); } });
  });
})();
