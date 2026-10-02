/* =========================================================
   m2.js · gráficas del Módulo 2 (modelación supervisada)
   ========================================================= */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);

  /* ---------- Residuales reales: Galton (parejos) vs Telco (cono) ---------- */
  function residuals() {
    if (!window.DATA) return;
    const make = (id, key, hetero, unidad, rango) => {
      if (!$(id) || !window.DATA[key]) return;
      const d = window.DATA[key], pts = d.f.map((x, i) => ({ x, y: d.r[i] }));
      const xs = d.f.slice().sort((a, b) => a - b), x0 = xs[0], x1 = xs[xs.length - 1];
      const fmt = (v) => (unidad === "$" ? "$" + Math.round(v).toLocaleString("es-MX") : v.toFixed(1) + " cm");
      Viz.chart(id, (P) => ({
        type: "scatter",
        data: { datasets: [
          { label: "residual", data: pts, backgroundColor: Viz.alpha(P.s[hetero ? 1 : 0], hetero ? 0.6 : 0.5), pointRadius: 2.5, pointHoverRadius: 5 },
          { label: "cero", data: [{ x: x0, y: 0 }, { x: x1, y: 0 }], type: "line", borderColor: P.ink3, borderWidth: 1, pointRadius: 0 },
        ] },
        options: Viz.baseOptions(P, {
          plugins: { title: { display: true, text: hetero ? "Telco: cargos totales ~ antigüedad" : "Galton: estatura ~ padres + sexo", color: P.ink1, font: { size: 12, weight: "600" } },
            tooltip: { filter: (i) => i.datasetIndex === 0, callbacks: { label: (c) => ` ajustado ${fmt(c.parsed.x)}, residual ${fmt(c.parsed.y)}` } } },
          scales: { x: { type: "linear", title: { display: true, text: "valor ajustado ŷ (" + (unidad === "$" ? "dólares" : "cm") + ")" } },
            y: { min: -rango, max: rango, title: { display: true, text: "residual (" + (unidad === "$" ? "dólares" : "cm") + ")" } } },
        }),
      }));
    };
    make("chart-res-ok", "res_galton", false, "cm", 30);
    make("chart-res-bad", "res_telco", true, "$", 4000);
  }

  /* ---------- Curva logística: Challenger ---------- */
  function logit() {
    if (!$("chart-logit")) return;
    // 23 vuelos previos al Challenger: temperatura al despegar (°F) y si hubo daño en alguna junta tórica
    const vuelos = [[53, 1], [57, 1], [58, 1], [63, 1], [66, 0], [67, 0], [67, 0], [67, 0], [68, 0], [69, 0], [70, 1], [70, 0],
      [70, 1], [70, 0], [72, 0], [73, 0], [75, 0], [75, 1], [76, 0], [76, 0], [78, 0], [79, 0], [81, 0]];
    const b0 = 15.0429, b1 = -0.2322, prob = (t) => 1 / (1 + Math.exp(-(b0 + b1 * t)));
    const aC = (t) => ((t - 32) / 1.8).toFixed(1).replace("-", "−") + " °C";
    const curve = [];
    for (let t = 30; t <= 85.0001; t += 0.5) curve.push({ x: +t.toFixed(1), y: prob(t) });
    const visto = {}, puntos = vuelos.map(([t, f]) => {            // apila los vuelos con la misma temperatura y resultado
      const k = t + "-" + f; visto[k] = (visto[k] || 0) + 1;
      return { x: t, y: f ? 1 - 0.035 * (visto[k] - 1) : 0.035 * (visto[k] - 1), falla: f };
    });
    Viz.chart("chart-logit", (P) => ({
      type: "scatter",
      data: { datasets: [
        { label: "P(daño)", data: curve, type: "line", borderColor: P.s[0], backgroundColor: P.s[0], borderWidth: 2, pointRadius: 0, pointHoverRadius: 4 },
        { label: "vuelo", data: puntos, backgroundColor: P.s[1], borderColor: P.surface, borderWidth: 2, pointRadius: 6, pointHoverRadius: 8 },
        { label: "pronóstico", data: [{ x: 31, y: prob(31) }], backgroundColor: P.s[2], borderColor: P.surface, borderWidth: 2, pointRadius: 7, pointHoverRadius: 9, pointStyle: "rectRot" },
      ] },
      options: Viz.baseOptions(P, {
        plugins: { tooltip: { callbacks: { label: (c) => {
          if (c.datasetIndex === 0) return ` ${c.parsed.x} °F (${aC(c.parsed.x)}) → P = ${c.parsed.y.toFixed(3)}`;
          if (c.datasetIndex === 2) return ` Pronóstico del 28 de enero de 1986: 31 °F (${aC(31)}) → P = ${prob(31).toFixed(4)}`;
          return ` Vuelo a ${c.parsed.x} °F (${aC(c.parsed.x)}): ${c.raw.falla ? "con daño" : "sin daño"}`;
        } } } },
        scales: { x: { type: "linear", min: 30, max: 85, title: { display: true, text: "temperatura al despegar (°F)" } },
          y: { min: -0.05, max: 1.05, ticks: { stepSize: 0.2, includeBounds: false }, title: { display: true, text: "probabilidad de daño" } } },
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
    const L = [2.86, 2.09, 1.59, 1.41, 0.84, 0.62, 0.39, 0.12, 0.05, 0.02];   // modelo de abandono (Telco), conjunto de prueba
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

  /* ---------- PSI en el tiempo: PIB per cápita de 142 países (Gapminder) ---------- */
  function psi() {
    if (!$("chart-psi")) return;
    const yrs = [1957, 1962, 1967, 1972, 1977, 1982, 1987, 1992, 1997, 2002, 2007];
    const ref = [0.025, 0.11, 0.223, 0.37, 0.525, 0.575, 0.656, 0.649, 0.812, 0.884, 1.119];
    const ant = [0.025, 0.038, 0.031, 0.041, 0.06, 0.019, 0.03, 0.023, 0.014, 0.004, 0.033];
    const lectura = (x) => (x < 0.1 ? "estable" : x <= 0.25 ? "cambio menor" : "cambio mayor");
    Viz.chart("chart-psi", (P) => ({
      type: "line",
      data: { labels: yrs, datasets: [
        { label: "contra 1952", data: ref, borderColor: P.s[0], backgroundColor: P.s[0], borderWidth: 2, pointRadius: 3, pointBorderColor: P.surface, pointBorderWidth: 1.5, pointHoverRadius: 5 },
        { label: "contra el periodo anterior", data: ant, borderColor: P.s[1], backgroundColor: P.s[1], borderWidth: 2, pointRadius: 3, pointBorderColor: P.surface, pointBorderWidth: 1.5, pointHoverRadius: 5 },
        { label: "control 0.10", data: yrs.map(() => 0.1), borderColor: P.warn, borderWidth: 1.5, pointRadius: 0 },
        { label: "control 0.25", data: yrs.map(() => 0.25), borderColor: P.crit, borderWidth: 1.5, pointRadius: 0 },
      ] },
      options: Viz.baseOptions(P, {
        interaction: { mode: "index", intersect: false },
        plugins: { tooltip: { filter: (i) => i.datasetIndex < 2, callbacks: { label: (c) => ` ${c.dataset.label}: PSI = ${c.parsed.y.toFixed(3)} → ${lectura(c.parsed.y)}` } } },
        scales: { x: { title: { display: true, text: "año" } }, y: { min: 0, max: 1.2, title: { display: true, text: "PSI" } } },
      }),
    }));
  }

  document.addEventListener("DOMContentLoaded", () => {
    [residuals, logit, lasso, gd, ens, roc, lift, psi].forEach((fn) => { try { fn(); } catch (e) { console.error(fn.name, e); } });
  });
})();
