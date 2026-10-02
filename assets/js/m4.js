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
    const W = {
      gato: [0.9, 2.6], perro: [1.25, 2.45], "ratón": [0.55, 2.2],
      rey: [2.6, 1.0], reina: [2.4, 1.35], "príncipe": [2.15, 0.75],
      manzana: [-2.2, 1.6], pera: [-2.0, 1.95], naranja: [-2.45, 1.25],
      correr: [-0.45, -2.4], caminar: [-0.8, -2.15],
      bueno: [2.0, -1.55], excelente: [2.45, -1.85], genial: [2.2, -1.3],
    };
    const words = Object.keys(W);
    ["emb-a", "emb-b"].forEach((id, k) => { $(id).innerHTML = words.map((w) => `<option${w === (k ? "perro" : "gato") ? " selected" : ""}>${w}</option>`).join(""); });
    const cx = 320, cy = 175, sc = 52;
    const cos = (a, b) => (a[0] * b[0] + a[1] * b[1]) / (Math.hypot(...a) * Math.hypot(...b));
    function draw() {
      const A = $("emb-a").value, B = $("emb-b").value;
      let s = `<line x1="20" y1="${cy}" x2="620" y2="${cy}" class="d-line"/><line x1="${cx}" y1="10" x2="${cx}" y2="330" class="d-line"/>`;
      words.forEach((w) => {
        const [x, y] = W[w], px = cx + x * sc, py = cy - y * sc, sel = w === A || w === B;
        if (!sel) s += `<circle cx="${px}" cy="${py}" r="4" fill="var(--muted-series)"/><text x="${px + 7}" y="${py + 4}" class="d-text-3">${w}</text>`;
      });
      [[A, "var(--s1)"], [B, "var(--s2)"]].forEach(([w, c]) => {
        const [x, y] = W[w], px = cx + x * sc, py = cy - y * sc;
        s += `<line x1="${cx}" y1="${cy}" x2="${px}" y2="${py}" stroke="${c}" stroke-width="2.5"/><circle cx="${px}" cy="${py}" r="6" fill="${c}" stroke="var(--bg)" stroke-width="2"/><text x="${px + 9}" y="${py - 7}" class="d-text" font-weight="700">${w}</text>`;
      });
      svg.innerHTML = s;
      const c = cos(W[A], W[B]), ang = (Math.acos(Math.max(-1, Math.min(1, c))) * 180) / Math.PI;
      const msg = c > 0.9 ? "muy similares (mismo grupo semántico)" : c > 0.5 ? "algo relacionadas" : c > -0.2 ? "poco relacionadas (casi ortogonales)" : "direcciones opuestas";
      $("emb-out").textContent = `${A} = [${W[A].join(", ")}]   ${B} = [${W[B].join(", ")}]\n` +
        `cos(θ) = (${W[A][0]}·${W[B][0]} + ${W[A][1]}·${W[B][1]}) / (${Math.hypot(...W[A]).toFixed(3)}·${Math.hypot(...W[B]).toFixed(3)}) = ${c.toFixed(4)}   → ángulo ≈ ${ang.toFixed(1)}°\n` +
        `Interpretación: ${msg}.  (Con one-hot, cualquier par daría cos = 0.)`;
    }
    ["emb-a", "emb-b"].forEach((id) => $(id).addEventListener("change", draw));
    draw();
  }

  /* ---------- 14. Curvas de transfer learning ---------- */
  function transferChart() {
    if (!$("chart-transfer")) return;
    const T = Array.from({ length: 41 }, (_, t) => t);
    const without = T.map((t) => 0.12 + 0.68 * (1 - Math.exp(-t / 14)));
    const withT = T.map((t) => 0.38 + 0.52 * (1 - Math.exp(-t / 5)));
    const ann = {
      id: "ann",
      afterDatasetsDraw(chart) {
        const { ctx, scales } = chart, P = Viz.palette();
        ctx.save(); ctx.font = "600 11px Nunito, system-ui, sans-serif"; ctx.fillStyle = P.ink2;
        ctx.fillText("↑ comienzo superior", scales.x.getPixelForValue(0.6), scales.y.getPixelForValue(0.42));
        ctx.fillText("pendiente más alta", scales.x.getPixelForValue(5), scales.y.getPixelForValue(0.66));
        ctx.textAlign = "right"; ctx.fillText("asíntota superior ↑", scales.x.getPixelForValue(40), scales.y.getPixelForValue(0.935));
        ctx.restore();
      },
    };
    Viz.chart("chart-transfer", (P) => ({
      type: "line",
      data: { labels: T, datasets: [
        { label: "con transferencia", data: withT, borderColor: P.s[0], backgroundColor: P.s[0], borderWidth: 2, pointRadius: 0, pointHoverRadius: 4 },
        { label: "sin transferencia", data: without, borderColor: P.muted, backgroundColor: P.muted, borderWidth: 2, pointRadius: 0, pointHoverRadius: 4 },
      ] },
      options: Viz.baseOptions(P, {
        interaction: { mode: "index", intersect: false },
        plugins: { tooltip: { callbacks: { title: (it) => "época " + it[0].label, label: (c) => ` ${c.dataset.label}: ${(c.parsed.y * 100).toFixed(1)} %` } } },
        scales: { x: { title: { display: true, text: "entrenamiento (épocas)" }, ticks: { maxTicksLimit: 9 } }, y: { min: 0, max: 1, title: { display: true, text: "desempeño (accuracy)" }, ticks: { callback: (v) => v * 100 + "%" } } },
      }),
      plugins: [ann],
    }));
  }

  document.addEventListener("DOMContentLoaded", () => {
    [batchCalc, activationChart, softmaxWidget, paramCalc, convWidget, outSize, poolWidget, dropoutWidget,
      archChart, lrChart, optRace, vanishChart, embWidget, transferChart].forEach((fn) => {
      try { fn(); } catch (e) { console.error(fn.name, e); }
    });
  });
})();
