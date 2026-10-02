/* =========================================================
   m1.js · simuladores y gráficas del Módulo 1
   ========================================================= */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const num = (id, d) => { const v = parseFloat($(id) && $(id).value); return isFinite(v) ? v : d; };

  /* ---------- Ventanas de tiempo ---------- */
  function ventanas() {
    const svg = $("vt-svg"); if (!svg) return;
    const T = 20;
    function draw() {
      const vobs = num("vt-obs", 5), vdes = num("vt-des", 1);
      const a = $("vt-ancla");
      a.min = vobs; a.max = T - vdes;
      let ancla = Math.min(Math.max(num("vt-ancla", 10), vobs), T - vdes);
      a.value = ancla;
      $("vt-obsv").textContent = vobs; $("vt-desv").textContent = vdes; $("vt-anclav").textContent = ancla;
      const x0 = 20, w = 32;
      let s = "";
      for (let t = 1; t <= T; t++) {
        const x = x0 + (t - 1) * w;
        const inObs = t > ancla - vobs && t <= ancla, inDes = t > ancla && t <= ancla + vdes;
        const cls = inObs ? "d-fill-1" : inDes ? "d-fill-8" : "d-box";
        s += `<rect x="${x + 1}" y="40" width="${w - 2}" height="34" rx="5" class="${cls}"/>`;
        s += `<text x="${x + w / 2}" y="62" text-anchor="middle" class="d-text-2">${t}</text>`;
      }
      const ax = x0 + ancla * w;
      s += `<line x1="${ax}" y1="22" x2="${ax}" y2="92" stroke="var(--ink-1)" stroke-width="2"/>`;
      s += `<text x="${ax}" y="16" text-anchor="middle" class="d-text" font-weight="700">ancla t=${ancla}</text>`;
      const ox = x0 + (ancla - vobs) * w, dx = x0 + ancla * w;
      s += `<text x="${ox + (vobs * w) / 2}" y="110" text-anchor="middle" class="d-text-2">observación → X</text>`;
      s += `<text x="${dx + (vdes * w) / 2}" y="110" text-anchor="middle" class="d-text-2">desempeño → y</text>`;
      svg.innerHTML = s;
      const n = T - vobs - vdes + 1;
      $("vt-out").textContent =
        `X se calcula con t ∈ [${ancla - vobs + 1}, ${ancla}]  (sum, min, max, mean, std, rachas…)\n` +
        `y se toma de t ∈ [${ancla + 1}, ${ancla + vdes}]\n` +
        `Anclas válidas con ${T} periodos: de ${vobs} a ${T - vdes} → ${n} renglones por unidad muestral.\n` +
        `(En el notebook: 700 periodos, vobs = 20, vdes = 1 → anclas 20…699 → 680 renglones por estación.)`;
    }
    ["vt-obs", "vt-des", "vt-ancla"].forEach((id) => $(id).addEventListener("input", draw));
    draw();
  }

  /* ---------- PCA: varianza explicada ---------- */
  function pcaChart() {
    if (!$("chart-pca") || !window.DATA) return;
    const d = window.DATA.pca_wine, labels = d.ind.map((_, i) => "PC" + (i + 1));
    Viz.chart("chart-pca", (P) => ({
      type: "bar",
      data: {
        labels,
        datasets: [
          { type: "line", label: "acumulada", data: d.cum, borderColor: P.s[1], backgroundColor: P.s[1], borderWidth: 2, pointRadius: 4, pointBorderColor: P.surface, pointBorderWidth: 2, order: 0 },
          { type: "bar", label: "individual", data: d.ind, backgroundColor: P.s[0], borderRadius: { topLeft: 4, topRight: 4 }, maxBarThickness: 24, order: 1 },
        ],
      },
      options: Viz.baseOptions(P, {
        interaction: { mode: "index", intersect: false },
        plugins: { tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${c.parsed.y.toFixed(1)} %` } } },
        scales: { x: { grid: { display: false } }, y: { min: 0, max: 100, title: { display: true, text: "% de varianza" }, ticks: { callback: (v) => v + "%" } } },
      }),
    }));
    Viz.dataTable("tbl-pca", ["Componente", "Individual (%)", "Acumulada (%)"], labels.map((l, i) => [l, d.ind[i].toFixed(2), d.cum[i].toFixed(2)]));
  }

  /* ---------- WoE por edad ---------- */
  function woeChart() {
    if (!$("chart-woe")) return;
    const labels = ["18–29", "30–40", "41–51", "52–63", "64–74"], woe = [-0.8648, -0.3855, 0.2281, 0.8025, 0.9933];
    Viz.chart("chart-woe", (P) => ({
      type: "bar",
      data: { labels, datasets: [{ label: "WoE", data: woe, backgroundColor: woe.map((v) => (v < 0 ? P.s[7] : P.s[0])), borderRadius: 4, borderSkipped: false, maxBarThickness: 24 }] },
      options: Viz.baseOptions(P, {
        indexAxis: "y",
        plugins: { tooltip: { callbacks: { label: (c) => ` WoE = ${c.parsed.x.toFixed(4)}` } } },
        scales: { x: { min: -1.1, max: 1.1, title: { display: true, text: "WoE = ln(% no evento / % evento)" } }, y: { grid: { display: false }, title: { display: true, text: "edad" } } },
      }),
    }));
  }

  /* ---------- Calculadora WoE / IV ---------- */
  function woeCalc() {
    const tb = document.querySelector("#woe-tbl tbody"); if (!tb) return;
    // Valores iniciales = variable edad del programa m1_07 (IV = 0.4792)
    const ini = [["18–29", 391, 657], ["30–40", 262, 711], ["41–51", 163, 817], ["52–63", 102, 908], ["64–74", 84, 905]];
    tb.innerHTML = ini.map(([b, e, ne], i) =>
      `<tr><td>${b}</td><td class="num"><input class="inp" type="number" min="0" step="1" value="${e}" data-r="${i}" data-c="e" style="width:90px" aria-label="Eventos ${b}"></td>` +
      `<td class="num"><input class="inp" type="number" min="0" step="1" value="${ne}" data-r="${i}" data-c="n" style="width:90px" aria-label="No eventos ${b}"></td>` +
      `<td class="num" id="woe-pe${i}"></td><td class="num" id="woe-pn${i}"></td><td class="num" id="woe-w${i}"></td><td class="num" id="woe-iv${i}"></td></tr>`).join("");
    const fmt = (v, d) => (isFinite(v) ? v.toFixed(d) : v > 0 ? "∞" : v < 0 ? "−∞" : "—");
    function calc() {
      const rows = ini.map((_, i) => ({
        e: Math.max(0, parseFloat(tb.querySelector(`[data-r="${i}"][data-c="e"]`).value) || 0),
        n: Math.max(0, parseFloat(tb.querySelector(`[data-r="${i}"][data-c="n"]`).value) || 0),
      }));
      const E = rows.reduce((s, r) => s + r.e, 0), N = rows.reduce((s, r) => s + r.n, 0);
      let iv = 0;
      rows.forEach((r, i) => {
        const pe = E ? r.e / E : 0, pn = N ? r.n / N : 0;
        const w = Math.log(pn / pe), ivi = (pn - pe) * w;
        iv += isNaN(ivi) ? 0 : ivi;
        $("woe-pe" + i).textContent = pe.toFixed(4); $("woe-pn" + i).textContent = pn.toFixed(4);
        $("woe-w" + i).textContent = fmt(w, 4); $("woe-iv" + i).textContent = fmt(ivi, 4);
      });
      const poder = !isFinite(iv) ? "infinito: algún bin no tiene eventos o no eventos (¿la variable da la respuesta?)"
        : iv < 0.02 ? "no ayuda a la predicción" : iv < 0.1 ? "bajo" : iv < 0.3 ? "regular" : iv < 0.5 ? "fuerte" : "sobrepredictiva (revisar)";
      $("woe-out").textContent = `Total: ${E} eventos y ${N} no eventos  (tasa de evento = ${E + N ? (E / (E + N)).toFixed(3) : "—"})\n` +
        `IV = Σ (%no evento − %evento) · WoE = ${fmt(iv, 4)}  →  poder predictivo ${poder}`;
    }
    tb.addEventListener("input", calc);
    calc();
  }

  /* ---------- Calculadora Factor / Offset / PDO ---------- */
  function pdoCalc() {
    if (!$("pdo-out")) return;
    function calc() {
      const pdo = num("pdo-pdo", 20), sref = num("pdo-score", 600), oref = num("pdo-odds", 50), ocli = num("pdo-cli", 6.14);
      if (pdo <= 0 || oref <= 0 || ocli <= 0) { $("pdo-out").textContent = "PDO y momios deben ser positivos."; return; }
      const factor = pdo / Math.LN2, offset = sref - factor * Math.log(oref), score = offset + factor * Math.log(ocli);
      $("pdo-out").textContent =
        `Factor = PDO / ln 2 = ${pdo} / 0.6931 = ${factor.toFixed(4)}\n` +
        `Offset = Score − Factor · ln(odds) = ${sref} − ${factor.toFixed(4)} · ln(${oref}) = ${offset.toFixed(4)}\n` +
        `Score(momios ${ocli}) = Offset + Factor · ln(${ocli}) = ${score.toFixed(1)}\n` +
        `Score(momios ${(2 * ocli).toFixed(2)}) = ${(score + pdo).toFixed(1)}   ← +${pdo} puntos al duplicar los momios\n` +
        `P(malo) del cliente = 1 / (1 + momios) = ${(1 / (1 + ocli)).toFixed(4)}`;
    }
    ["pdo-pdo", "pdo-score", "pdo-odds", "pdo-cli"].forEach((id) => $(id).addEventListener("input", calc));
    calc();
  }

  document.addEventListener("DOMContentLoaded", () => {
    [ventanas, pcaChart, woeChart, woeCalc, pdoCalc].forEach((fn) => { try { fn(); } catch (e) { console.error(fn.name, e); } });
  });
})();
