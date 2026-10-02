/* =========================================================
   m4.js · simuladores y gráficas del Módulo 4 (Deep Learning)
   ========================================================= */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const fmt = (v, d = 4) => (Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : Number(v).toFixed(d));
  const num = (id, def) => { const v = parseFloat($(id) && $(id).value); return isFinite(v) ? v : def; };
  const nf = new Intl.NumberFormat("es-MX");

  /* ---------- 1. Lotes, iteraciones y épocas ---------- */
  function batchCalc() {
    if (!$("bc-out")) return;
    const upd = () => {
      const n = Math.max(1, Math.round(num("bc-n", 1000))), b = Math.max(1, Math.round(num("bc-b", 32))), e = Math.max(1, Math.round(num("bc-e", 10)));
      const it = Math.ceil(n / b), last = n - (it - 1) * b;
      $("bc-out").textContent =
        `Iteraciones (batches) por época = ceil(${n} / ${b}) = ${it}` + (last !== b ? `   (el último lote tiene ${last} ejemplos)` : "") +
        `\nActualizaciones de pesos totales = ${it} × ${e} épocas = ${nf.format(it * e)}` +
        `\nCada época la red ve los ${nf.format(n)} ejemplos una vez (forward + backward).`;
    };
    ["bc-n", "bc-b", "bc-e"].forEach((id) => $(id).addEventListener("input", upd));
    upd();
  }

  /* ---------- 2. Funciones de activación ---------- */
  function activationChart() {
    if (!$("chart-activ")) return;
    let deriv = false;
    const xs = []; for (let x = -5; x <= 5.0001; x += 0.1) xs.push(+x.toFixed(2));
    const sig = (x) => 1 / (1 + Math.exp(-x));
    const F = {
      fn: [(x) => sig(x), (x) => Math.tanh(x), (x) => Math.max(0, x)],
      d: [(x) => sig(x) * (1 - sig(x)), (x) => 1 - Math.tanh(x) ** 2, (x) => (x > 0 ? 1 : 0)],
    };
    const names = ["Sigmoid", "Tanh", "ReLU"];
    const entry = Viz.chart("chart-activ", (P) => ({
      type: "line",
      data: {
        datasets: names.map((n, i) => ({
          label: n, data: xs.map((x) => ({ x, y: (deriv ? F.d : F.fn)[i](x) })),
          borderColor: P.s[i], backgroundColor: P.s[i], borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0,
          stepped: deriv && i === 2 ? "before" : false,
        })),
      },
      options: Viz.baseOptions(P, {
        interaction: { mode: "index", intersect: false },
        plugins: { tooltip: { callbacks: { title: (it) => "x = " + it[0].parsed.x.toFixed(1), label: (c) => ` ${c.dataset.label}: ${c.parsed.y.toFixed(3)}` } } },
        scales: {
          x: { type: "linear", min: -5, max: 5, title: { display: true, text: "x (suma ponderada z)" } },
          y: { min: deriv ? -0.05 : -1.1, max: deriv ? 1.1 : 3, title: { display: true, text: deriv ? "derivada g′(x)" : "g(x)" } },
        },
      }),
    }));
    const setMode = (d) => {
      deriv = d; $("activ-deriv").classList.toggle("primary", d); $("activ-fn").classList.toggle("primary", !d); entry.render();
    };
    $("activ-fn").addEventListener("click", () => setMode(false));
    $("activ-deriv").addEventListener("click", () => setMode(true));
  }

  /* ---------- 3. Softmax ---------- */
  function softmaxWidget() {
    if (!$("sm-bars")) return;
    const names = ["manzana", "pera", "naranja"];
    const upd = () => {
      const z = [num("sm-z1", 0), num("sm-z2", 0), num("sm-z3", 0)];
      const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0), p = e.map((v) => v / s);
      const best = p.indexOf(Math.max(...p));
      $("sm-bars").innerHTML = p.map((v, i) =>
        `<div class="bar-row"><span>${names[i]}</span><div class="bar-track"><div class="bar-fill" style="width:${(v * 100).toFixed(1)}%;background:var(--s${[1, 3, 2][i]})"></div></div><span class="val">${v.toFixed(3)}</span></div>`).join("");
      $("sm-readout").textContent =
        `z = [${z.map((v) => v.toFixed(1)).join(", ")}]\n` +
        `e^z = [${z.map((v) => Math.exp(v).toFixed(3)).join(", ")}]   suma = ${z.map(Math.exp).reduce((a, b) => a + b, 0).toFixed(3)}\n` +
        `softmax = [${p.map((v) => v.toFixed(3)).join(", ")}]   suma = ${p.reduce((a, b) => a + b, 0).toFixed(3)}\n` +
        `Predicción: ${names[best]}  |  pérdida si la real fuera "${names[0]}": −ln(${p[0].toFixed(3)}) = ${(-Math.log(p[0])).toFixed(3)}`;
    };
    ["sm-z1", "sm-z2", "sm-z3"].forEach((id) => $(id).addEventListener("input", upd));
    upd();
  }

  /* ---------- 4. Calculadora de parámetros ---------- */
  function paramCalc() {
    if (!$("pc-out")) return;
    const cfg = {
      dense: { a: "Entradas", b: "Neuronas", c: null, def: [784, 128] },
      conv: { a: "Canales de entrada", b: "Filtros", c: "Kernel k", def: [3, 32, 3] },
      rnn: { a: "Dim. de entrada", b: "Unidades", c: null, def: [8, 16] },
      lstm: { a: "Dim. de entrada", b: "Unidades", c: null, def: [8, 16] },
      gru: { a: "Dim. de entrada", b: "Unidades", c: null, def: [8, 16] },
      emb: { a: "Vocabulario (input_dim)", b: "Dimensión (output_dim)", c: null, def: [10000, 16] },
    };
    const upd = () => {
      const t = $("pc-type").value, a = Math.round(num("pc-a", 1)), b = Math.round(num("pc-b", 1)), k = Math.round(num("pc-c", 3));
      let r, f;
      if (t === "dense") { r = a * b + b; f = `${a}·${b} + ${b}  (pesos + 1 sesgo por neurona)`; }
      if (t === "conv") { r = (k * k * a + 1) * b; f = `(${k}·${k}·${a} + 1) · ${b}  (cada filtro: k·k·C pesos + 1 sesgo)`; }
      if (t === "rnn") { r = b * (b + a + 1); f = `${b}·(${b} + ${a} + 1)  (W_hh + W_xh + b)`; }
      if (t === "lstm") { r = 4 * b * (b + a + 1); f = `4·${b}·(${b} + ${a} + 1)  (4 bloques: f, i, c̃, o)`; }
      if (t === "gru") { r = 3 * b * (b + a + 2); f = `3·${b}·(${b} + ${a} + 2)  (3 bloques; Keras usa 2 sesgos por bloque)`; }
      if (t === "emb") { r = a * b; f = `${a}·${b}  (una fila de ${b} números por palabra)`; }
      $("pc-out").textContent = `Parámetros = ${f}\n          = ${nf.format(r)}`;
    };
    const setType = () => {
      const c = cfg[$("pc-type").value];
      $("pc-la").textContent = c.a; $("pc-lb").textContent = c.b;
      $("pc-wc").style.display = c.c ? "" : "none";
      $("pc-a").value = c.def[0]; $("pc-b").value = c.def[1]; if (c.c) $("pc-c").value = c.def[2];
      upd();
    };
    $("pc-type").addEventListener("change", setType);
    ["pc-a", "pc-b", "pc-c"].forEach((id) => $(id).addEventListener("input", upd));
    setType();
  }

  /* ---------- 5. Convolución paso a paso ---------- */
  function convWidget() {
    if (!$("conv-input")) return;
    const IMG = [[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]];
    const K = {
      slide: [[1, 0, 1], [0, 1, 0], [1, 0, 1]], edgeh: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]], edgev: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
      sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], ident: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    };
    let step = -1, timer = null, out = [], X = [], ker = K.slide, pad = 0, n = 3;
    const grid = (el, rows, cls) => {
      el.style.gridTemplateColumns = `repeat(${rows[0].length}, auto)`;
      el.innerHTML = rows.map((r, i) => r.map((v, j) => `<div class="cell ${cls ? cls(i, j) : ""}" data-i="${i}" data-j="${j}">${v}</div>`).join("")).join("");
    };
    function setup() {
      stop();
      ker = K[$("conv-preset").value]; pad = $("conv-pad").checked ? 1 : 0;
      X = IMG.map((r) => r.slice());
      if (pad) { X = [Array(7).fill(0)].concat(X.map((r) => [0, ...r, 0])).concat([Array(7).fill(0)]); }
      n = X.length - 3 + 1;
      out = [];
      for (let i = 0; i < n; i++) { out.push([]); for (let j = 0; j < n; j++) { let s = 0; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) s += X[i + a][j + b] * ker[a][b]; out[i].push(s); } }
      step = -1; draw();
    }
    function draw() {
      const ci = step >= 0 ? Math.floor(step / n) : -1, cj = step >= 0 ? step % n : -1;
      grid($("conv-input"), X, (i, j) => {
        const isPad = pad && (i === 0 || j === 0 || i === X.length - 1 || j === X.length - 1);
        const hl = step >= 0 && i >= ci && i < ci + 3 && j >= cj && j < cj + 3;
        return (isPad ? "pad " : "") + (hl ? "hl" : "");
      });
      grid($("conv-kernel"), ker, () => "k");
      const shown = out.map((r, i) => r.map((v, j) => (i * n + j <= step ? v : "")));
      grid($("conv-output"), shown, (i, j) => (i * n + j === step ? "cur" : i * n + j < step ? "done" : ""));
      if (step < 0) {
        $("conv-readout").textContent = `Entrada ${X.length}×${X.length}, kernel 3×3, stride 1, padding ${pad} → salida ${n}×${n}  (O = (${5} − 3 + 2·${pad})/1 + 1 = ${n})\nPresiona "Siguiente paso".`;
      } else {
        const terms = []; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) terms.push(`${X[ci + a][cj + b]}·${ker[a][b]}`);
        $("conv-readout").textContent = `Posición (${ci + 1}, ${cj + 1}):  ${terms.join(" + ")} = ${out[ci][cj]}` +
          (step === n * n - 1 ? `\n¡Listo! Mapa completo ${n}×${n}. Con el kernel de la diapositiva: [[4,3,4],[2,4,3],[2,3,4]].` : "");
      }
    }
    function next() { if (step < n * n - 1) { step++; draw(); } else stop(); }
    function stop() { if (timer) { clearInterval(timer); timer = null; } $("conv-play").textContent = "Reproducir"; }
    $("conv-step").addEventListener("click", () => { stop(); next(); });
    $("conv-play").addEventListener("click", () => {
      if (timer) { stop(); return; }
      if (step >= n * n - 1) step = -1;
      $("conv-play").textContent = "Pausa"; timer = setInterval(() => { next(); if (step >= n * n - 1) stop(); }, 650);
    });
    $("conv-reset").addEventListener("click", setup);
    $("conv-preset").addEventListener("change", setup);
    $("conv-pad").addEventListener("change", setup);
    setup();
  }

  /* ---------- 6. Forma de salida de Conv2D ---------- */
  function outSize() {
    if (!$("os-out")) return;
    const upd = () => {
      const W = num("os-w", 28), K = num("os-k", 3), P = num("os-p", 0), S = Math.max(1, num("os-s", 1)), C = num("os-c", 1), F = num("os-f", 32);
      const raw = (W - K + 2 * P) / S + 1, O = Math.floor((W - K + 2 * P) / S) + 1;
      $("os-out").textContent =
        `O = (W − K + 2P)/S + 1 = (${W} − ${K} + 2·${P})/${S} + 1 = ${Number.isInteger(raw) ? raw : raw.toFixed(2) + " → se redondea hacia abajo = " + O}\n` +
        `Forma de salida: (${O}, ${O}, ${F})   ← la profundidad = número de filtros\n` +
        `Parámetros: (K·K·C + 1)·F = (${K}·${K}·${C} + 1)·${F} = ${nf.format((K * K * C + 1) * F)}\n` +
        `Si después aplicas MaxPooling 2×2 (stride 2): (${Math.floor(O / 2)}, ${Math.floor(O / 2)}, ${F})`;
    };
    ["os-w", "os-k", "os-p", "os-s", "os-c", "os-f"].forEach((id) => $(id).addEventListener("input", upd));
    upd();
  }

  /* ---------- 7. Pooling ---------- */
  function poolWidget() {
    if (!$("pool-input")) return;
    const X = [[2, 2, 7, 3], [9, 4, 6, 1], [8, 5, 2, 4], [3, 1, 2, 6]];
    const col = ["var(--s1)", "var(--s3)", "var(--s4)", "var(--s8)"];
    const blk = (i, j) => Math.floor(i / 2) * 2 + Math.floor(j / 2);
    const names = ["superior izquierdo", "superior derecho", "inferior izquierdo", "inferior derecho"];
    const vals = (b) => { const r = Math.floor(b / 2) * 2, c = (b % 2) * 2; return [X[r][c], X[r][c + 1], X[r + 1][c], X[r + 1][c + 1]]; };
    const draw = (hb) => {
      const mode = $("pool-mode").value;
      const tint = (b, strong) => `background:color-mix(in srgb, ${col[b]} ${strong ? 45 : 18}%, var(--bg))`;
      $("pool-input").style.gridTemplateColumns = "repeat(4, auto)";
      $("pool-input").innerHTML = X.map((r, i) => r.map((v, j) => `<div class="cell" style="${tint(blk(i, j), hb === blk(i, j))}">${v}</div>`).join("")).join("");
      $("pool-output").style.gridTemplateColumns = "repeat(2, auto)";
      $("pool-output").innerHTML = [0, 1, 2, 3].map((b) => {
        const v = vals(b), r = mode === "max" ? Math.max(...v) : v.reduce((a, c) => a + c, 0) / 4;
        return `<div class="cell" data-b="${b}" tabindex="0" style="${tint(b, true)};width:48px">${mode === "max" ? r : r.toFixed(2)}</div>`;
      }).join("");
      $("pool-output").querySelectorAll(".cell").forEach((c) => {
        const b = +c.dataset.b;
        const show = () => { draw(b); };
        c.addEventListener("mouseenter", show); c.addEventListener("focus", show); c.addEventListener("click", show);
      });
      const b = hb == null ? 0 : hb, v = vals(b);
      $("pool-readout").textContent = `Bloque ${names[b]}: [${v.join(", ")}] → ` +
        (mode === "max" ? `max = ${Math.max(...v)}` : `promedio = (${v.join(" + ")})/4 = ${(v.reduce((a, c) => a + c, 0) / 4).toFixed(2)}`) +
        `\n4×4 = 16 valores → 2×2 = 4 valores (cada dimensión a la mitad; queda 1/4).`;
    };
    $("pool-mode").addEventListener("change", () => draw(0));
    $("pool-output").addEventListener("mouseleave", () => {});
    draw(0);
  }

  /* ---------- 8. Dropout ---------- */
  function dropoutWidget() {
    const svg = $("do-svg"); if (!svg) return;
    const layers = [4, 6, 6, 2], W = 560, H = 260, xs = [70, 220, 370, 500];
    let it = 0, mask = [];
    const ys = (n) => Array.from({ length: n }, (_, i) => H / 2 + (i - (n - 1) / 2) * 38);
    function sample() {
      const p = num("do-p", 0.5);
      mask = layers.map((n, l) => Array.from({ length: n }, () => (l === 0 || l === 3 ? 1 : Math.random() >= p ? 1 : 0)));
      it++; draw();
    }
    function draw() {
      const p = num("do-p", 0.5); let s = "";
      for (let l = 0; l < 3; l++) {
        ys(layers[l]).forEach((y1, i) => ys(layers[l + 1]).forEach((y2, j) => {
          const off = !mask[l][i] || !mask[l + 1][j];
          s += `<line x1="${xs[l]}" y1="${y1}" x2="${xs[l + 1]}" y2="${y2}" class="d-line${off ? " d-off" : ""}"/>`;
        }));
      }
      layers.forEach((n, l) => ys(n).forEach((y, i) => {
        const on = mask[l][i]; const cls = l === 0 ? "d-in" : l === 3 ? "d-out" : "d-hid";
        s += `<circle cx="${xs[l]}" cy="${y}" r="13" class="d-node ${cls}${on ? "" : " d-off"}"/>`;
        if (!on) s += `<path d="M${xs[l] - 10} ${y - 10} L${xs[l] + 10} ${y + 10} M${xs[l] + 10} ${y - 10} L${xs[l] - 10} ${y + 10}" stroke="var(--crit)" stroke-width="2.5" stroke-linecap="round"/>`;
      }));
      ["entrada", "oculta 1", "oculta 2", "salida"].forEach((t, l) => { s += `<text x="${xs[l]}" y="${H - 6}" text-anchor="middle" class="d-text-3">${t}</text>`; });
      svg.innerHTML = s;
      const act = mask[1].reduce((a, b) => a + b, 0) + mask[2].reduce((a, b) => a + b, 0);
      $("do-readout").textContent = `Iteración ${it}: neuronas ocultas activas = ${act}/12   (esperado = 12·(1 − p) = ${(12 * (1 - p)).toFixed(1)})\n` +
        `Las neuronas vivas se escalan por 1/(1−p) = ${(1 / (1 - p)).toFixed(2)} en entrenamiento. En predicción se usan las 12.`;
    }
    $("do-p").addEventListener("input", () => { $("do-pv").textContent = num("do-p", 0.5).toFixed(1); sample(); });
    $("do-step").addEventListener("click", sample);
    sample();
  }

  /* ---------- 9. Arquitecturas CNN (dispersión) ---------- */
  function archChart() {
    if (!$("chart-arch")) return;
    const D = [
      ["AlexNet", 60, 63.3, 2012], ["Inception V1", 5, 69.8, 2014], ["VGG 16", 138, 74.4, 2014], ["VGG 19", 144, 74.5, 2014],
      ["Inception V2", 11.2, 74.8, 2015], ["ResNet-50", 26, 77.15, 2015], ["ResNet-152", 60, 78.57, 2015], ["Inception V3", 27, 78.8, 2015],
      ["DenseNet-121", 8, 74.98, 2016], ["DenseNet-264", 22, 77.85, 2016], ["BiT-L (ResNet)", 928, 87.54, 2019],
      ["NoisyStudent EffNet-L2", 480, 88.4, 2020], ["Meta Pseudo Labels", 480, 90.2, 2021],
    ];
    const labelThese = new Set(["AlexNet", "Inception V1", "VGG 16", "DenseNet-121", "ResNet-50", "BiT-L (ResNet)", "Meta Pseudo Labels"]);
    const labels = {
      id: "pointLabels",
      afterDatasetsDraw(chart) {
        const ctx = chart.ctx, meta = chart.getDatasetMeta(0), P = Viz.palette();
        ctx.save(); ctx.font = "600 11px Nunito, system-ui, sans-serif"; ctx.fillStyle = P.ink2;
        meta.data.forEach((pt, i) => {
          const d = D[i]; if (!labelThese.has(d[0])) return;
          const right = pt.x < chart.chartArea.right - 120;
          ctx.textAlign = right ? "left" : "right";
          ctx.fillText(d[0], pt.x + (right ? 9 : -9), pt.y + (d[0] === "VGG 16" ? 12 : -6));
        });
        ctx.restore();
      },
    };
    Viz.chart("chart-arch", (P) => ({
      type: "scatter",
      data: { datasets: [{ label: "Modelos", data: D.map((d) => ({ x: d[1], y: d[2] })), backgroundColor: P.s[0], borderColor: P.surface, borderWidth: 2, pointRadius: 6, pointHoverRadius: 8 }] },
      options: Viz.baseOptions(P, {
        plugins: { tooltip: { callbacks: { title: (it) => D[it[0].dataIndex][0], label: (c) => { const d = D[c.dataIndex]; return [` ${d[1]} M parámetros`, ` Top-1: ${d[2]} %`, ` Año: ${d[3]}`]; } } } },
        scales: {
          x: { type: "logarithmic", min: 3, max: 1500, title: { display: true, text: "Parámetros (millones, escala log)" }, ticks: { maxRotation: 0, autoSkip: false, callback: (v) => ([5, 10, 20, 50, 100, 200, 500, 1000].includes(v) ? v : "") } },
          y: { min: 60, max: 92, title: { display: true, text: "ImageNet Top-1 accuracy (%)" } },
        },
      }),
      plugins: [labels],
    }));
    Viz.dataTable("tbl-arch", ["Modelo", "Parámetros (M)", "Top-1 (%)", "Año"], D.map((d) => [d[0], d[1], d[2], d[3]]));
  }

  /* ---------- 10. Learning rate (GD en θ²) ---------- */
  function lrChart() {
    if (!$("chart-lr")) return;
    const curve = []; for (let x = -3.4; x <= 3.4001; x += 0.05) curve.push({ x: +x.toFixed(2), y: x * x });
    const path = (lr) => { const pts = []; let t = 2.8; for (let k = 0; k <= 14; k++) { pts.push({ x: t, y: t * t }); t = t - lr * 2 * t; } return pts; };
    const entry = Viz.chart("chart-lr", (P) => {
      const lr = num("lr-range", 0.1), pts = path(lr);
      return {
        type: "scatter",
        data: {
          datasets: [
            { label: "J(θ) = θ²", data: curve, showLine: true, borderColor: P.muted, borderWidth: 2, pointRadius: 0, pointHoverRadius: 0 },
            { label: "Pasos de GD", data: pts, showLine: true, borderColor: P.s[1], backgroundColor: P.s[1], borderWidth: 2, pointRadius: 4, pointBorderColor: P.surface, pointBorderWidth: 2 },
          ],
        },
        options: Viz.baseOptions(P, {
          plugins: { tooltip: { filter: (i) => i.datasetIndex === 1, callbacks: { label: (c) => ` paso ${c.dataIndex}: θ = ${c.parsed.x.toFixed(4)}, J = ${c.parsed.y.toFixed(4)}` } } },
          scales: { x: { type: "linear", min: -3.5, max: 3.5, title: { display: true, text: "θ" } }, y: { min: -0.3, max: 12, title: { display: true, text: "J(θ)" } } },
        }),
      };
    });
    const upd = () => {
      const lr = num("lr-range", 0.1); $("lr-val").textContent = lr.toFixed(2); entry.render();
      const pts = path(lr), last = pts[pts.length - 1].x, factor = 1 - 2 * lr;
      const verdict = Math.abs(factor) >= 1 ? "DIVERGE: cada paso se aleja más del mínimo (|1 − 2α| ≥ 1)." :
        factor < 0 ? "Converge oscilando de un lado a otro (1 − 2α < 0)." : lr < 0.1 ? "Converge, pero muy lento: muchos pasos pequeños." : "Converge rápido.";
      $("lr-readout").textContent = `θ ← θ − α·2θ = (1 − 2α)·θ = ${factor.toFixed(2)}·θ\n` +
        `θ: ${pts.slice(0, 6).map((p) => p.x.toFixed(3)).join(" → ")} → …  (paso 14: ${fmt(last)})\n${verdict}`;
    };
    $("lr-range").addEventListener("input", upd);
    document.querySelectorAll("[data-lr]").forEach((b) => b.addEventListener("click", () => { $("lr-range").value = b.dataset.lr; upd(); }));
    upd();
  }

  /* ---------- 11. Carrera de optimizadores ---------- */
  function optRace() {
    const svg = $("opt-svg"); if (!svg) return;
    const f = (x, y) => 0.5 * x * x + 10 * y * y, g = (x, y) => [x, 20 * y];
    const W = 640, H = 300, X0 = -5.2, X1 = 2.2, Y0 = -1.5, Y1 = 1.5;
    const sx = (x) => 20 + ((x - X0) / (X1 - X0)) * (W - 40), sy = (y) => H - 15 - ((y - Y0) / (Y1 - Y0)) * (H - 30);
    function run(steps, lr) {
      const start = [-4.5, 1.2];
      const opt = {
        SGD: (() => { let p = start.slice(); const o = [p.slice()]; for (let k = 0; k < steps; k++) { const d = g(...p); p = [p[0] - lr * d[0], p[1] - lr * d[1]]; o.push(p.slice()); } return o; })(),
        Momentum: (() => { let p = start.slice(), v = [0, 0]; const o = [p.slice()]; for (let k = 0; k < steps; k++) { const d = g(...p); v = [0.85 * v[0] - 0.02 * d[0], 0.85 * v[1] - 0.02 * d[1]]; p = [p[0] + v[0], p[1] + v[1]]; o.push(p.slice()); } return o; })(),
        RMSProp: (() => { let p = start.slice(), s = [0, 0]; const a = 0.15, o = [p.slice()]; for (let k = 0; k < steps; k++) { const d = g(...p); s = s.map((v, i) => 0.9 * v + 0.1 * d[i] * d[i]); p = p.map((v, i) => v - (a * d[i]) / (Math.sqrt(s[i]) + 1e-8)); o.push(p.slice()); } return o; })(),
        Adam: (() => { let p = start.slice(), m = [0, 0], v = [0, 0]; const a = 0.15, o = [p.slice()]; for (let k = 1; k <= steps; k++) { const d = g(...p); m = m.map((q, i) => 0.9 * q + 0.1 * d[i]); v = v.map((q, i) => 0.999 * q + 0.001 * d[i] * d[i]); p = p.map((q, i) => q - (a * (m[i] / (1 - 0.9 ** k))) / (Math.sqrt(v[i] / (1 - 0.999 ** k)) + 1e-8)); o.push(p.slice()); } return o; })(),
      };
      return opt;
    }
    function draw() {
      const steps = num("opt-steps", 40), lr = num("opt-lr", 0.09);
      $("opt-stepsv").textContent = steps; $("opt-lrv").textContent = lr.toFixed(2);
      let s = "";
      [0.5, 2, 5, 10].forEach((c) => {
        const rx = Math.sqrt(2 * c), ry = Math.sqrt(c / 10);
        s += `<ellipse cx="${sx(0)}" cy="${sy(0)}" rx="${(rx / (X1 - X0)) * (W - 40)}" ry="${(ry / (Y1 - Y0)) * (H - 30)}" class="d-line"/>`;
      });
      s += `<path d="M${sx(0) - 7} ${sy(0)} H${sx(0) + 7} M${sx(0)} ${sy(0) - 7} V${sy(0) + 7}" stroke="var(--ink-2)" stroke-width="2"/>`;
      s += `<text x="${sx(0) + 10}" y="${sy(0) + 18}" class="d-text-3">mínimo (0, 0)</text>`;
      const runs = run(steps, lr), colors = { SGD: "var(--s1)", Momentum: "var(--s2)", RMSProp: "var(--s3)", Adam: "var(--s7)" };
      const clip = (p) => [Math.max(X0, Math.min(X1, p[0])), Math.max(Y0, Math.min(Y1, p[1]))];
      Object.entries(runs).forEach(([name, pts]) => {
        s += `<polyline fill="none" stroke="${colors[name]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="${pts.map((p) => { const q = clip(p); return sx(q[0]).toFixed(1) + "," + sy(q[1]).toFixed(1); }).join(" ")}"/>`;
        const e = clip(pts[pts.length - 1]);
        s += `<circle cx="${sx(e[0])}" cy="${sy(e[1])}" r="5" fill="${colors[name]}" stroke="var(--bg)" stroke-width="2"/>`;
      });
      s += `<circle cx="${sx(-4.5)}" cy="${sy(1.2)}" r="5" fill="var(--ink-2)" stroke="var(--bg)" stroke-width="2"/><text x="${sx(-4.5) + 8}" y="${sy(1.2) - 8}" class="d-text-3">inicio</text>`;
      svg.innerHTML = s;
      $("opt-readout").textContent = Object.entries(runs).map(([n, pts]) => { const p = pts[pts.length - 1]; const v = f(p[0], p[1]); return `${n.padEnd(9)} f = ${isFinite(v) ? fmt(v) : "∞ (divergió)"}   en (${fmt(p[0], 3)}, ${fmt(p[1], 3)})`; }).join("\n") +
        `\nFijos: Momentum α = 0.02, β = 0.85 · RMSProp α = 0.15 · Adam α = 0.15. SGD usa el α del control (diverge si α ≥ 0.1).`;
    }
    ["opt-steps", "opt-lr"].forEach((id) => $(id).addEventListener("input", draw));
    draw();
  }

  /* ---------- 12. Gradiente que se desvanece/explota ---------- */
  function vanishChart() {
    if (!$("chart-vanish")) return;
    const T = Array.from({ length: 51 }, (_, t) => t);
    const fs = [0.5, 0.9, 1.0, 1.1];
    Viz.chart("chart-vanish", (P) => ({
      type: "line",
      data: { labels: T, datasets: fs.map((fct, i) => ({ label: "factor " + fct, data: T.map((t) => Math.pow(fct, t)), borderColor: [P.s[0], P.s[1], P.muted, P.s[6]][i], backgroundColor: [P.s[0], P.s[1], P.muted, P.s[6]][i], borderWidth: 2, pointRadius: 0, pointHoverRadius: 4 })) },
      options: Viz.baseOptions(P, {
        interaction: { mode: "index", intersect: false },
        plugins: { tooltip: { callbacks: { title: (it) => "paso t = " + it[0].label, label: (c) => ` ${c.dataset.label}: ${c.parsed.y.toExponential(2)}` } } },
        scales: { x: { title: { display: true, text: "pasos hacia atrás en el tiempo" }, ticks: { maxTicksLimit: 11 } }, y: { type: "logarithmic", min: 1e-16, max: 1e3, title: { display: true, text: "magnitud del gradiente (log)" }, ticks: { callback: (v) => { const e = Math.log10(v); return Number.isInteger(e) && e % 3 === 0 ? "1e" + e : ""; } } } },
      }),
    }));
  }

  /* ---------- 13. Mapa de embeddings ---------- */
  function embWidget() {
    const svg = $("emb-svg"); if (!svg) return;
    // Vectores reales de GloVe (glove-wiki-gigaword-50, Stanford NLP): 50 dimensiones por palabra.
    // P = proyección PCA a 2D de estas 16 palabras (conserva el 47 % de su variación), solo para dibujar.
    const P = {"cat": [-0.207, 0.045], "dog": [-0.22, 0.134], "horse": [-0.03, 0.106], "mouse": [-0.426, -0.025], "king": [0.535, -0.197], "queen": [0.557, -0.273], "prince": [0.693, -0.225], "princess": [0.536, -0.371], "apple": [-0.464, -0.19], "banana": [-0.418, -0.471], "orange": [-0.289, -0.337], "grape": [-0.588, -0.239], "good": [0.013, 0.568], "great": [0.321, 0.379], "excellent": [-0.014, 0.6], "bad": [0.001, 0.496]};
    const V = {
      cat: [0.45281, -0.50108, -0.53714, -0.015697, 0.22191, 0.54602, -0.67301, -0.6891, 0.63493, -0.19726, 0.33685, 0.7735, 0.90094, 0.38488, 0.38367, 0.2657, -0.08057, 0.61089, -1.2894, -0.22313, -0.61578, 0.21697, 0.35614, 0.44499, 0.60885, -1.1633, -1.1579, 0.36118, 0.10466, -0.78325, 1.4352, 0.18629, -0.26112, 0.83275, -0.23123, 0.32481, 0.14485, -0.44552, 0.33497, -0.95946, -0.097479, 0.48138, -0.43352, 0.69455, 0.91043, -0.28173, 0.41637, -1.2609, 0.71278, 0.23782],
      dog: [0.11008, -0.38781, -0.57615, -0.27714, 0.70521, 0.53994, -1.0786, -0.40146, 1.1504, -0.5678, 0.0038977, 0.52878, 0.64561, 0.47262, 0.48549, -0.18407, 0.1801, 0.91397, -1.1979, -0.5778, -0.37985, 0.33606, 0.772, 0.75555, 0.45506, -1.7671, -1.0503, 0.42566, 0.41893, -0.68327, 1.5673, 0.27685, -0.61708, 0.64638, -0.076996, 0.37118, 0.1308, -0.45137, 0.25398, -0.74392, -0.086199, 0.24068, -0.64819, 0.83549, 1.2502, -0.51379, 0.04224, -0.88118, 0.7158, 0.38519],
      horse: [-0.20454, 0.23321, -0.59158, -0.29205, 0.29391, 0.31169, -0.94937, 0.055974, 1.0031, -1.0761, -0.0094648, 0.18381, -0.048405, -0.35717, 0.26004, -0.41028, 0.51489, 1.2009, -1.6136, -1.1003, -0.23455, -0.81654, -0.15103, 0.37068, 0.477, -1.7027, -1.2183, 0.038898, 0.23327, 0.028245, 1.6588, 0.26703, -0.29938, 0.99149, 0.34263, 0.15477, 0.028372, 0.56276, -0.62823, -0.67923, -0.163, -0.49922, -0.8599, 0.85469, 0.75059, -1.0399, -0.11033, -1.4237, 0.65984, -0.3198],
      mouse: [0.92126, -0.81902, 0.23245, 0.52219, 0.51949, 0.88641, -0.39156, -1.5406, 0.60995, 0.22859, 0.62091, 0.66589, 0.72291, 0.66718, 0.10358, 0.5369, -0.14446, 0.48026, -0.80902, -0.35139, -0.61242, -0.53007, 0.11605, 0.48692, 0.95751, -0.69058, -0.19449, 0.12434, -0.1614, -1.5485, 1.5875, -0.083181, -0.60285, -0.045127, -0.28487, 0.5245, -0.15309, -0.10653, -0.11039, -0.6564, 0.10237, 0.61861, -1.0989, 0.32814, 0.74616, -0.132, 1.6579, -1.1101, -0.029862, 0.087264],
      king: [0.50451, 0.68607, -0.59517, -0.022801, 0.60046, -0.13498, -0.08813, 0.47377, -0.61798, -0.31012, -0.076666, 1.493, -0.034189, -0.98173, 0.68229, 0.81722, -0.51874, -0.31503, -0.55809, 0.66421, 0.1961, -0.13495, -0.11476, -0.30344, 0.41177, -2.223, -1.0756, -1.0783, -0.34354, 0.33505, 1.9927, -0.04234, -0.64319, 0.71125, 0.49159, 0.16754, 0.34344, -0.25663, -0.8523, 0.1661, 0.40102, 1.1685, -1.0137, -0.21585, -0.15155, 0.78321, -0.91241, -1.6106, -0.64426, -0.51042],
      queen: [0.37854, 1.8233, -1.2648, -0.1043, 0.35829, 0.60029, -0.17538, 0.83767, -0.056798, -0.75795, 0.22681, 0.98587, 0.60587, -0.31419, 0.28877, 0.56013, -0.77456, 0.071421, -0.5741, 0.21342, 0.57674, 0.3868, -0.12574, 0.28012, 0.28135, -1.8053, -1.0421, -0.19255, -0.55375, -0.054526, 1.5574, 0.39296, -0.2475, 0.34251, 0.45365, 0.16237, 0.52464, -0.070272, -0.83744, -1.0326, 0.45946, 0.25302, -0.17837, -0.73398, -0.20025, 0.2347, -0.56095, -2.2839, 0.0092753, -0.60284],
      prince: [0.98846, 1.4535, -0.53081, 0.10509, 0.84058, 0.14018, 0.066562, 1.3341, -0.75813, -0.35223, 0.16588, 1.0016, 0.019623, -0.66392, 0.092825, 0.25132, -0.16274, -0.11954, -0.50072, 0.67374, 0.66886, -0.038679, 0.20223, -0.15211, 0.12169, -1.8324, -0.85664, -0.62454, -0.31896, 0.60221, 1.411, 0.50157, -0.11413, 0.51808, 0.847, 0.17618, -0.035265, 0.56405, -0.38524, 0.6027, 0.34331, 1.1836, -0.37197, -1.1069, 0.00012758, -0.18202, -1.3696, -1.497, 0.40618, -0.42445],
      princess: [1.4992, 1.6053, -1.1699, 0.69597, 0.63491, 1.0803, -0.15271, 1.0974, -0.12842, -0.74608, 0.59572, 0.76493, 0.18664, -0.47217, 0.72322, 0.48368, -1.4334, -0.032644, -0.1652, 0.38196, 0.71329, 0.89524, -0.26091, 0.44074, 0.52343, -1.2422, -1.8396, -0.10232, -0.081415, -0.087913, 0.91303, 0.69385, 0.26242, 1.0921, 0.66015, -0.19193, -0.028172, -0.13914, -0.54633, -0.75658, 0.45368, 0.066445, 0.49774, -1.2702, 0.10915, -0.46564, -0.65682, -2.0678, 0.72255, -0.35506],
      apple: [0.52042, -0.8314, 0.49961, 1.2893, 0.1151, 0.057521, -1.3753, -0.97313, 0.18346, 0.47672, -0.15112, 0.35532, 0.25912, -0.77857, 0.52181, 0.47695, -1.4251, 0.858, 0.59821, -1.0903, 0.33574, -0.60891, 0.41742, 0.21569, -0.07417, -0.5822, -0.4502, 0.17253, 0.16448, -0.38413, 2.3283, -0.66682, -0.58181, 0.74389, 0.095015, -0.47865, -0.84591, 0.38704, 0.23693, -1.5523, 0.64802, -0.16521, -1.4719, -0.16224, 0.79857, 0.97391, 0.40027, -0.21912, -0.30938, 0.26581],
      banana: [-0.25522, -0.75249, -0.86655, 1.1197, 0.12887, 1.0121, -0.57249, -0.36224, 0.44341, -0.12211, 0.073524, 0.21387, 0.96744, -0.068611, 0.51452, -0.053425, -0.21966, 0.23012, 1.043, -0.77016, -0.16753, -1.0952, 0.24837, 0.20019, -0.40866, -0.48037, 0.10674, 0.5316, 1.111, -0.19322, 1.4768, -0.51783, -0.79569, 1.7971, -0.33392, -0.14545, -1.5454, 0.0135, 0.10684, -0.30722, -0.54572, 0.38938, 0.24659, -0.85166, 0.54966, 0.82679, -0.68081, -0.77864, -0.028242, -0.82872],
      orange: [-0.42783, 0.43089, -0.50351, 0.5776, 0.097786, 0.2608, -0.68767, -0.31936, -0.25337, -0.37255, -0.045907, -0.53688, 0.97511, -0.44595, -0.50414, -0.086751, -1.0645, 0.36625, -0.52428, -1.3413, -0.2391, -0.58808, 0.56378, -0.062501, -1.7429, -0.88077, -0.27933, 1.4705, 0.50436, -0.69174, 2.0018, 0.26663, -0.85679, -0.18893, -0.021125, -0.055118, -0.50337, -0.67157, 0.55502, -0.8009, 0.10695, 0.1459, -0.55588, -0.64971, 0.22046, 0.67415, -0.45119, -1.1462, 0.16348, -0.62946],
      grape: [0.23017, -0.33868, -1.4997, -0.35237, 0.16379, 0.44121, -0.99947, -0.92632, 0.35382, 0.92485, 0.50485, -0.21886, 1.2231, -1.4518, 0.71129, -0.21506, 0.83317, 0.72531, 0.11544, -1.5128, -1.1301, -1.5405, 0.84343, 0.8874, -0.54008, 0.31496, -0.59696, 0.91315, 0.16258, 0.055713, 1.0624, -0.92158, -0.96032, 0.44075, 1.0285, -0.45583, -1.656, 0.59739, 0.49696, 0.0047636, -0.09192, -0.11237, -0.30139, -0.81769, 1.1218, 0.10276, 0.47661, -0.12207, -0.30815, -0.4438],
      good: [-0.35586, 0.5213, -0.6107, -0.30131, 0.94862, -0.31539, -0.59831, 0.12188, -0.031943, 0.55695, -0.10621, 0.63399, -0.4734, -0.075895, 0.38247, 0.081569, 0.82214, 0.2222, -0.0083764, -0.7662, -0.56253, 0.61759, 0.20292, -0.048598, 0.87815, -1.6549, -0.77418, 0.15435, 0.94823, -0.3952, 3.7302, 0.82855, -0.14104, 0.016395, 0.21115, -0.036085, -0.15587, 0.86583, 0.26309, -0.71015, -0.03677, 0.0018282, -0.17704, 0.27032, 0.11026, 0.14133, -0.057322, 0.27207, 0.31305, 0.92771],
      great: [-0.026567, 1.3357, -1.028, -0.3729, 0.52012, -0.12699, -0.35433, 0.37824, -0.29716, 0.093894, -0.034122, 0.92961, -0.14023, -0.63299, 0.020801, -0.21533, 0.96923, 0.47654, -1.0039, -0.24013, -0.36325, -0.004757, -0.5148, -0.4626, 1.2447, -1.8316, -1.5581, -0.37465, 0.53362, 0.20883, 3.2209, 0.64549, 0.37438, -0.17657, -0.024164, 0.33786, -0.419, 0.40081, -0.11449, 0.051232, -0.15205, 0.29855, -0.44052, 0.11089, -0.24633, 0.66251, -0.26949, -0.49658, -0.41618, -0.2549],
      excellent: [-0.40431, 0.78002, -0.67538, -0.097149, 0.54242, -0.51283, -0.47758, -0.11429, 0.55194, 0.4953, 0.25681, 0.41516, 0.38223, 0.033936, -0.51981, -0.5752, 0.50975, 0.44029, -0.16885, -0.75687, -0.36747, 0.37269, -0.0867, -0.31057, 0.7034, -0.46849, -0.47845, 0.15794, 0.23969, 0.16104, 2.7632, 0.33381, 0.47953, -0.3731, 0.71095, 0.6879, 0.067399, 1.4177, -0.3563, -0.48159, 0.53281, -0.074593, 0.016028, 0.023371, -0.02953, -0.12463, 0.12501, 0.72255, 0.4759, 0.79514],
      bad: [-0.17981, -0.40407, -0.1653, -0.60687, -0.39656, 0.12688, -0.053049, 0.38024, -0.51008, 0.46593, -0.30818, 0.79362, -0.85766, -0.25143, 1.0448, 0.18628, 0.13688, 0.092588, -0.2236, -0.13604, -0.19482, 0.057702, 0.56133, 0.24823, 0.627, -1.8437, -1.2573, 0.64482, 1.2787, -0.29522, 3.0493, 0.62079, 0.90369, -0.030099, -0.13091, 0.30525, -0.070138, -0.12912, 0.72277, -0.79774, -0.70277, 0.038009, 0.27192, 0.35679, 0.26493, 0.13037, -0.01369, 0.33713, 0.99956, 0.72031],
    };
    const ES = { cat: "gato", dog: "perro", horse: "caballo", mouse: "ratón", king: "rey", queen: "reina", prince: "príncipe", princess: "princesa",
      apple: "manzana", banana: "plátano", orange: "naranja", grape: "uva", good: "bueno", great: "genial", excellent: "excelente", bad: "malo" };
    const words = Object.keys(P);
    ["emb-a", "emb-b"].forEach((id, k) => { $(id).innerHTML = words.map((w) => `<option value="${w}"${w === (k ? "dog" : "cat") ? " selected" : ""}>${w} (${ES[w]})</option>`).join(""); });
    const cx = 320, cy = 170, sc = 250;
    const dot = (a, b) => a.reduce((t, x, i) => t + x * b[i], 0);
    const cos = (a, b) => dot(a, b) / Math.sqrt(dot(a, a) * dot(b, b));
    function draw() {
      const A = $("emb-a").value, B = $("emb-b").value;
      const xy = (w) => [cx + P[w][0] * sc, cy - P[w][1] * sc];
      let s = "";
      words.forEach((w) => {
        const [px, py] = xy(w);
        const izq = w === "good";                                   // evita que se encime con "excellent"
        if (w !== A && w !== B) s += `<circle cx="${px}" cy="${py}" r="4" fill="var(--muted-series)"/><text x="${px + (izq ? -7 : 7)}" y="${py + 4}" class="d-text-3"${izq ? ' text-anchor="end"' : ""}>${w}</text>`;
      });
      const [ax, ay] = xy(A), [bx, by] = xy(B);
      s += `<line x1="${ax}" y1="${ay}" x2="${bx}" y2="${by}" class="d-line-2 d-dash"/>`;
      [[A, "var(--s1)"], [B, "var(--s2)"]].forEach(([w, c]) => {
        const [px, py] = xy(w);
        const izq = w === "good";
        s += `<circle cx="${px}" cy="${py}" r="6.5" fill="${c}" stroke="var(--bg)" stroke-width="2"/><text x="${px + (izq ? -9 : 9)}" y="${py - 8}" class="d-text" font-weight="700"${izq ? ' text-anchor="end"' : ""}>${w}</text>`;
      });
      svg.innerHTML = s;
      const c = cos(V[A], V[B]), ang = (Math.acos(Math.max(-1, Math.min(1, c))) * 180) / Math.PI;
      const msg = c >= 0.75 ? "muy similares: aparecen en contextos muy parecidos" : c >= 0.5 ? "relacionadas" : c >= 0.25 ? "poco relacionadas" : "prácticamente sin relación (casi ortogonales)";
      const ini = (w) => "[" + V[w].slice(0, 4).join(", ") + ", …]";
      $("emb-out").textContent = `${A} = ${ini(A)}   ${B} = ${ini(B)}   (50 números cada una)\n` +
        `cos(θ) en 50 dimensiones = ${c.toFixed(4)}   → ángulo ≈ ${ang.toFixed(1)}°\n` +
        `Interpretación: ${msg}.  (Con one-hot, cualquier par de palabras distintas daría cos = 0.)`;
    }
    ["emb-a", "emb-b"].forEach((id) => $(id).addEventListener("change", draw));
    draw();
  }

  /* ---------- 14. Curvas de transfer learning: experimento real con MNIST (m4_k5_transferencia_mnist.py) ---------- */
  function transferChart() {
    if (!$("chart-transfer")) return;
    const E = Array.from({ length: 15 }, (_, k) => k + 1);
    const cero = [0.624, 0.838, 0.863, 0.881, 0.888, 0.919, 0.925, 0.92, 0.921, 0.931, 0.939, 0.943, 0.944, 0.945, 0.945];
    const congelada = [0.794, 0.869, 0.893, 0.909, 0.914, 0.918, 0.924, 0.927, 0.93, 0.933, 0.935, 0.937, 0.938, 0.941, 0.942];
    const fino = [0.809, 0.87, 0.893, 0.907, 0.918, 0.923, 0.92, 0.915, 0.916, 0.923, 0.937, 0.949, 0.949, 0.951, 0.949];
    Viz.chart("chart-transfer", (P) => ({
      type: "line",
      data: { labels: E, datasets: [
        { label: "base congelada", data: congelada, borderColor: P.s[0], backgroundColor: P.s[0], borderWidth: 2, pointRadius: 3, pointBorderColor: P.surface, pointBorderWidth: 1.5, pointHoverRadius: 5 },
        { label: "ajuste fino", data: fino, borderColor: P.s[1], backgroundColor: P.s[1], borderWidth: 2, pointRadius: 3, pointBorderColor: P.surface, pointBorderWidth: 1.5, pointHoverRadius: 5 },
        { label: "desde cero", data: cero, borderColor: P.muted, backgroundColor: P.muted, borderWidth: 2, pointRadius: 3, pointBorderColor: P.surface, pointBorderWidth: 1.5, pointHoverRadius: 5 },
      ] },
      options: Viz.baseOptions(P, {
        interaction: { mode: "index", intersect: false },
        plugins: { tooltip: { callbacks: { title: (it) => "época " + it[0].label, label: (c) => ` ${c.dataset.label}: ${(c.parsed.y * 100).toFixed(1)} %` } } },
        scales: { x: { title: { display: true, text: "época de entrenamiento en la tarea destino" } },
          y: { min: 0.6, max: 1, title: { display: true, text: "accuracy en prueba (dígitos 5 a 9)" }, ticks: { callback: (v) => Math.round(v * 100) + " %" } } },
      }),
    }));
  }

  document.addEventListener("DOMContentLoaded", () => {
    [batchCalc, activationChart, softmaxWidget, paramCalc, convWidget, outSize, poolWidget, dropoutWidget,
      archChart, lrChart, optRace, vanishChart, embWidget, transferChart].forEach((fn) => {
      try { fn(); } catch (e) { console.error(fn.name, e); }
    });
  });
})();
