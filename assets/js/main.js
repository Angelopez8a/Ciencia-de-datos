/* =========================================================
   main.js · comportamiento común de todas las páginas
   ========================================================= */
(function () {
  "use strict";

  // ---------- almacenamiento seguro (puede fallar en modo privado) ----------
  const store = {
    get(k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { window.localStorage.setItem(k, v); } catch (e) { /* sin persistencia */ } },
  };
  window.SafeStore = store;

  // ---------- Tema claro / oscuro ----------
  const root = document.documentElement;
  const saved = store.get("apuntes-theme");
  if (saved === "dark" || saved === "light") root.setAttribute("data-theme", saved);

  function currentTheme() {
    const t = root.getAttribute("data-theme");
    if (t) return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  window.currentTheme = currentTheme;

  const ICON_SUN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
  const ICON_MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';

  function paintThemeButton() {
    const b = document.getElementById("theme-btn");
    if (!b) return;
    const dark = currentTheme() === "dark";
    b.innerHTML = dark ? ICON_SUN : ICON_MOON;
    b.setAttribute("aria-label", dark ? "Cambiar a modo claro" : "Cambiar a modo oscuro");
    b.title = b.getAttribute("aria-label");
  }

  function toggleTheme() {
    const next = currentTheme() === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    store.set("apuntes-theme", next);
    paintThemeButton();
    document.dispatchEvent(new CustomEvent("themechange"));
  }

  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
      if (!root.getAttribute("data-theme")) { paintThemeButton(); document.dispatchEvent(new CustomEvent("themechange")); }
    });
  }

  // ---------- utilidades ----------
  function slug(s) {
    return s.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "")
      .replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "").slice(0, 60);
  }

  // ---------- Índice lateral automático ----------
  function buildToc() {
    const toc = document.getElementById("toc");
    const content = document.querySelector(".content");
    if (!toc || !content) return [];
    const heads = content.querySelectorAll("h2, h3");
    const used = new Set();
    const items = [];
    heads.forEach(function (h) {
      if (h.closest(".no-toc")) return;
      if (!h.id) {
        let id = slug(h.textContent) || "sec";
        let n = 2; const base = id;
        while (used.has(id) || document.getElementById(id)) id = base + "-" + n++;
        h.id = id;
      }
      used.add(h.id);
      const li = document.createElement("li");
      if (h.tagName === "H3") li.className = "sub";
      const a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.getAttribute("data-toc") || h.textContent;
      li.appendChild(a);
      toc.appendChild(li);
      items.push({ h: h, a: a });
    });
    toc.addEventListener("click", function (e) {
      if (e.target.tagName === "A") document.body.classList.remove("toc-open");
    });
    return items;
  }

  function scrollSpy(items) {
    if (!items.length) return;
    let active = null, ticking = false;
    const sb = document.querySelector(".sidebar");
    function update() {
      ticking = false;
      const line = 120;                       // px desde arriba del viewport
      let cur = items[0];
      for (let i = 0; i < items.length; i++) {
        if (items[i].h.getBoundingClientRect().top - line <= 0) cur = items[i]; else break;
      }
      if (cur === active) return;
      if (active) active.a.classList.remove("active");
      cur.a.classList.add("active");
      active = cur;
      if (sb && window.innerWidth > 1000) {   // mantener visible el enlace activo dentro del índice
        const r = cur.a.getBoundingClientRect(), sr = sb.getBoundingClientRect();
        if (r.top < sr.top + 40 || r.bottom > sr.bottom - 40) sb.scrollTop += r.top - sr.top - sr.height / 2;
      }
    }
    window.addEventListener("scroll", function () {
      if (!ticking) { ticking = true; window.requestAnimationFrame(update); }
    }, { passive: true });
    window.addEventListener("resize", update);
    update();
  }

  // ---------- Barra de progreso de lectura ----------
  function progressBar() {
    const bar = document.querySelector(".progress");
    if (!bar) return;
    function upd() {
      const h = document.documentElement;
      const max = h.scrollHeight - h.clientHeight;
      bar.style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + "%";
    }
    window.addEventListener("scroll", upd, { passive: true });
    upd();
  }

  // ---------- Botones de copiar código ----------
  function copyButtons() {
    document.querySelectorAll("[data-copy]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        const card = btn.closest(".code-card");
        const code = card ? card.querySelector("pre code") : null;
        if (!code) return;
        const text = code.innerText;
        const done = function () { const o = btn.textContent; btn.textContent = "¡Copiado!"; setTimeout(function () { btn.textContent = o; }, 1400); };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text); done(); });
        } else { fallbackCopy(text); done(); }
      });
    });
  }
  function fallbackCopy(text) {
    const ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch (e) { /* nada */ }
    document.body.removeChild(ta);
  }

  // ---------- Mostrar/ocultar tabla de datos de una gráfica ----------
  function tableToggles() {
    document.querySelectorAll("[data-table-toggle]").forEach(function (b) {
      b.addEventListener("click", function () {
        const w = document.getElementById(b.getAttribute("data-table-toggle"));
        if (!w) return;
        w.classList.toggle("open");
        b.textContent = w.classList.contains("open") ? "Ocultar tabla" : "Ver tabla de datos";
      });
    });
  }

  // ---------- Matemáticas (KaTeX) y sintaxis (highlight.js) ----------
  function renderMath() {
    if (window.renderMathInElement) {
      window.renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true },
        ],
        ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
        throwOnError: false,
      });
    }
  }
  function highlight() {
    if (window.hljs) {
      document.querySelectorAll("pre code.language-python").forEach(function (el) { window.hljs.highlightElement(el); });
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    paintThemeButton();
    const tb = document.getElementById("theme-btn");
    if (tb) tb.addEventListener("click", toggleTheme);
    const tt = document.getElementById("toc-btn");
    if (tt) tt.addEventListener("click", function () { document.body.classList.toggle("toc-open"); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") document.body.classList.remove("toc-open"); });
    const items = buildToc();
    scrollSpy(items);
    progressBar();
    copyButtons();
    tableToggles();
    highlight();
    renderMath();
  });
})();

/* =========================================================
   Helper de gráficas (Chart.js) con estilo consistente
   - lee los colores de las variables CSS (modo claro/oscuro)
   - se reconstruye al cambiar de tema
   ========================================================= */
(function () {
  "use strict";
  const registry = [];

  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function palette() {
    return {
      s: [css("--s1"), css("--s2"), css("--s3"), css("--s4"), css("--s5"), css("--s6"), css("--s7"), css("--s8")],
      muted: css("--muted-series"), ink1: css("--ink-1"), ink2: css("--ink-2"), ink3: css("--ink-3"),
      grid: css("--hair"), axis: css("--axis"), surface: css("--surface-1"), surface2: css("--surface-2"),
      good: css("--good"), crit: css("--crit"), warn: css("--warn"),
    };
  }
  function alpha(hex, a) {
    const h = hex.replace("#", "");
    const n = parseInt(h.length === 3 ? h.split("").map(function (c) { return c + c; }).join("") : h, 16);
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
  }

  function baseOptions(P, extra) {
    const o = {
      responsive: true, maintainAspectRatio: false, animation: { duration: 350 },
      interaction: { mode: "nearest", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: P.surface, titleColor: P.ink1, bodyColor: P.ink2, borderColor: P.axis, borderWidth: 1,
          padding: 10, cornerRadius: 8, boxPadding: 4, usePointStyle: true,
          titleFont: { family: "system-ui", weight: "600" }, bodyFont: { family: "system-ui" },
        },
      },
      scales: {
        x: { grid: { color: P.grid, drawTicks: false }, border: { color: P.axis }, ticks: { color: P.ink3, padding: 6, font: { size: 11 } }, title: { color: P.ink2, font: { size: 12 } } },
        y: { grid: { color: P.grid, drawTicks: false }, border: { color: P.axis }, ticks: { color: P.ink3, padding: 6, font: { size: 11 } }, title: { color: P.ink2, font: { size: 12 } } },
      },
    };
    return deepMerge(o, extra || {});
  }
  function deepMerge(a, b) {
    Object.keys(b).forEach(function (k) {
      if (b[k] && typeof b[k] === "object" && !Array.isArray(b[k]) && a[k] && typeof a[k] === "object") deepMerge(a[k], b[k]);
      else a[k] = b[k];
    });
    return a;
  }

  /** Registra una gráfica: build(P) debe regresar {type, data, options} */
  function chart(canvasId, build) {
    const el = document.getElementById(canvasId);
    if (!el || !window.Chart) return null;
    const entry = { el: el, build: build, inst: null };
    entry.render = function () {
      if (entry.inst) entry.inst.destroy();
      const cfg = build(palette());
      entry.inst = new window.Chart(el, cfg);
      return entry.inst;
    };
    entry.render();
    registry.push(entry);
    return entry;
  }

  document.addEventListener("themechange", function () {
    setTimeout(function () { registry.forEach(function (e) { e.render(); }); }, 30);
  });

  /** Construye una tabla HTML simple a partir de columnas */
  function dataTable(targetId, headers, rows) {
    const t = document.getElementById(targetId);
    if (!t) return;
    let h = "<table><thead><tr>" + headers.map(function (x, i) { return "<th" + (i ? ' class="num"' : "") + ">" + x + "</th>"; }).join("") + "</tr></thead><tbody>";
    rows.forEach(function (r) { h += "<tr>" + r.map(function (c, i) { return "<td" + (i ? ' class="num"' : "") + ">" + c + "</td>"; }).join("") + "</tr>"; });
    t.innerHTML = h + "</tbody></table>";
  }

  /** Generador pseudoaleatorio con semilla (para datos sintéticos reproducibles) */
  function rng(seed) {
    let s = seed >>> 0;
    const r = function () { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    r.normal = function () { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    return r;
  }

  window.Viz = { chart: chart, baseOptions: baseOptions, palette: palette, alpha: alpha, dataTable: dataTable, rng: rng, css: css };
})();
