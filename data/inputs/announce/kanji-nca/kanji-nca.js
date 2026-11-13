/* kanji-nca.js: grown, self-healing kanji (Growing Neural Cellular Automata, Mordvintsev et al., Distill 2020).
 *
 * Plain script, no dependencies. On a page it defines window.KanjiNCA; loaded as a Worker script it runs the engine.
 *
 *   const h = KanjiNCA.attachAll(
 *     [{ canvas, weights, textEl }, ...],   // weights = the object from nca_<char>.json (format "kanji-nca/1")
 *     { workerUrl,                           // optional: this file's URL; default = a blob Worker built from it
 *       radius: 9,                           // cursor disturbance radius, in grid cells
 *       pixelated: true,                     // chunky cells like the original
 *       color: canvas => cssColor,           // ink colour (default: the canvas's computed `color`)
 *       mainBudgetMs: 4 });                  // only used if no Worker can start (main-thread fallback)
 *   h.reset(); h.destroy();
 *
 * - All glyphs on a page share one Worker. It steps them round-robin and transfers each glyph's ink as bytes; the
 *   main thread only paints (one 64x64 putImageData per changed glyph per frame) and forwards pointer input.
 * - A glyph computes only while growing (~160 steps) or healing after a disturbance (until it matches its grown
 *   self again; if it can't within 500 steps, or it collapses or balloons, it regrows from one cell), and never
 *   while off-screen or in a hidden tab. Otherwise it costs nothing.
 * - With prefers-reduced-motion, or if anything fails (no weights, bad format), nothing starts and the text stays.
 *   Otherwise the text glyph is hidden at the first grown frame (one cell), so the growth itself is visible.
 * - Pointer and touch input come from window, so the layer can be pointer-events: none. A drag disturbs; idle heals.
 *
 * The update rule matches train_nca.py exactly: per-channel [x, sobel_x, sobel_y] perception with zero padding,
 * dx = W2 relu(W1 p + b1), stochastic fire 0.5, alive = maxpool3(alpha) > 0.1 before and after the update.
 */
(function factory(root) {
  "use strict";

  function f16(b64) {
    const bin = atob(b64);
    const u16 = new Uint16Array(bin.length / 2);
    for (let i = 0; i < u16.length; i++) u16[i] = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
      out[i] = e === 0 ? s * Math.pow(2, -14) * (m / 1024)
             : e === 31 ? (m ? NaN : s * Infinity)
             : s * Math.pow(2, e - 15) * (1 + m / 1024);
    }
    return out;
  }

  class NCA {
    constructor(w) {
      if (!w || w.format !== "kanji-nca/1") throw new Error("kanji-nca: unknown weights format");
      this.H = w.height; this.W = w.width; this.C = w.channels; this.K = w.hidden; this.A = w.alpha_channel;
      this.w1 = f16(w.w1); this.b1 = f16(w.b1); this.w2 = f16(w.w2);
      // state is stored with a one-cell zero border, so the 3x3 reads never bounds-check
      this.PW = this.W + 2; this.PH = this.H + 2;
      const n = this.PH * this.PW * this.C;
      this.x = new Float32Array(n); this.y = new Float32Array(n);
      this.pre = new Uint8Array(this.PH * this.PW);
      this.p = new Float32Array(this.C * 3); this.h = new Float32Array(this.K);
      this.fire = 0.5;
      this.reset();
    }
    idx(i, j) { return ((i + 1) * this.PW + (j + 1)) * this.C; }
    reset() {
      this.x.fill(0);
      const c = this.idx(this.H >> 1, this.W >> 1);
      for (let k = this.A; k < this.C; k++) this.x[c + k] = 1;
      this.t = 0;
    }
    aliveInto(buf, out) {
      const { H, W, C, A, PW } = this, rs = PW * C;
      out.fill(0);
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        const n = ((i + 1) * PW + (j + 1)) * C + A;
        if (buf[n - rs - C] > 0.1 || buf[n - rs] > 0.1 || buf[n - rs + C] > 0.1 ||
            buf[n - C] > 0.1 || buf[n] > 0.1 || buf[n + C] > 0.1 ||
            buf[n + rs - C] > 0.1 || buf[n + rs] > 0.1 || buf[n + rs + C] > 0.1) out[(i + 1) * PW + (j + 1)] = 1;
      }
    }
    /** One update. Returns the largest |change| of the ink channel (0 when nothing moved). */
    step() {
      const { H, W, C, K, A, PW, p, h, w1, b1, w2 } = this, x = this.x, y = this.y, pre = this.pre;
      const rs = PW * C, C3 = C * 3, fire = this.fire;
      this.aliveInto(x, pre);
      y.fill(0);
      let maxd = 0;
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        const q = (i + 1) * PW + (j + 1);
        if (!pre[q]) continue;
        const n = q * C;
        if (Math.random() > fire) { for (let c = 0; c < C; c++) y[n + c] = x[n + c]; continue; }
        // perception (correlation, zero border): [x, sobel_x, sobel_y] / 8 per channel
        for (let c = 0; c < C; c++) {
          const m = n + c;
          const a = x[m - rs - C], b = x[m - rs], cc = x[m - rs + C], d = x[m - C], e = x[m], f = x[m + C],
                g = x[m + rs - C], hh = x[m + rs], ii = x[m + rs + C];
          p[3 * c] = e;
          p[3 * c + 1] = ((cc - a) + 2 * (f - d) + (ii - g)) * 0.125;
          p[3 * c + 2] = ((g - a) + 2 * (hh - b) + (ii - cc)) * 0.125;
        }
        for (let k = 0; k < K; k++) {
          let s = b1[k]; const o = k * C3;
          for (let r = 0; r < C3; r++) s += w1[o + r] * p[r];
          h[k] = s > 0 ? s : 0;
        }
        for (let c = 0; c < C; c++) {
          let dd = 0; const o = c * K;
          for (let k = 0; k < K; k++) dd += w2[o + k] * h[k];
          y[n + c] = x[n + c] + dd;
          if (c === A) { const ad = dd < 0 ? -dd : dd; if (ad > maxd) maxd = ad; }
        }
      }
      this.aliveInto(y, pre);              // post-alive (pre-dead cells are already zero in y)
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        const q = (i + 1) * PW + (j + 1);
        if (!pre[q]) { const b = q * C; for (let c = 0; c < C; c++) y[b + c] = 0; }
      }
      this.x = y; this.y = x; this.t++;
      return maxd;
    }
    /** Displace cells around grid point (cy, cx): radial push plus a drag along (vy, vx). Semi-Lagrangian. */
    disturb(cy, cx, r, push, vy, vx) {
      const { H, W, C, PW } = this, x = this.x, y = this.y;
      y.set(x);
      const i0 = Math.max(0, Math.floor(cy - r)), i1 = Math.min(H - 1, Math.ceil(cy + r));
      const j0 = Math.max(0, Math.floor(cx - r)), j1 = Math.min(W - 1, Math.ceil(cx + r));
      for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
        const dy = i - cy, dx = j - cx, dist = Math.sqrt(dy * dy + dx * dx) + 1e-6;
        if (dist >= r) continue;
        const t = dist / r, fall = (1 - t * t) * (1 - t * t);
        const sy = i - fall * (push * dy / dist + vy), sx = j - fall * (push * dx / dist + vx);
        const y0 = Math.floor(sy), x0 = Math.floor(sx), fy = sy - y0, fx = sx - x0;
        const o = ((i + 1) * PW + (j + 1)) * C;
        for (let c = 0; c < C; c++) {
          let v = 0;
          for (let a = 0; a <= 1; a++) for (let b = 0; b <= 1; b++) {
            const yy = y0 + a, xx = x0 + b;
            if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
            v += (a ? fy : 1 - fy) * (b ? fx : 1 - fx) * x[((yy + 1) * PW + (xx + 1)) * C + c];
          }
          y[o + c] = v;
        }
      }
      this.x = y; this.y = x;
    }
    alpha(out) {
      const { H, W, C, A, PW } = this, x = this.x;
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        const a = x[((i + 1) * PW + (j + 1)) * C + A]; out[i * W + j] = a < 0 ? 0 : a > 1 ? 1 : a;
      }
      return out;
    }
    /** Ink as bytes 0..255 (what the worker transfers). */
    alpha8(out) {
      const { H, W, C, A, PW } = this, x = this.x;
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        const a = x[((i + 1) * PW + (j + 1)) * C + A]; out[i * W + j] = a <= 0 ? 0 : a >= 1 ? 255 : (a * 255 + 0.5) | 0;
      }
      return out;
    }
  }


  /** Owns the NCAs and decides who steps. The same code runs in the Worker or, as a fallback, on the main thread.
   *
   * Health: when a glyph finishes growing, its ink is kept as the reference. After a disturbance it steps until the
   * ink is back within TOL of that reference for OK_STEPS steps in a row. If it hasn't healed by MAX_HEAL steps, or
   * its ink mass collapses or balloons, it regrows from one cell. The page never sits on a broken glyph. */
  function Engine(post, budgetMs, schedule) {
    const GROW = 160, TOL = 1.5e-3, OK_STEPS = 30, STALL = 220, MAX_HEAL = 500;
    const glyphs = new Map();
    let order = [], turn = 0, paused = false, pending = false;
    const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
    const busy = s => s.visible && !paused && (s.nca.t < GROW || s.healing);
    function regrow(s) { s.nca.reset(); s.healing = false; s.ref = null; s.regrows++; }
    function round() {
      pending = false;
      const t0 = now(), frames = [], transfer = [];
      let done = 0;
      while (done < order.length) {
        const id = order[turn % order.length]; turn++; done++;
        const s = glyphs.get(id);
        if (!s || !busy(s)) continue;
        s.nca.step();
        const a = s.nca.alpha8(new Uint8Array(s.nca.H * s.nca.W));
        if (!s.ref && s.nca.t >= GROW) {                       // grown: keep the reference
          s.ref = a.slice(); s.refMass = 0; for (let i = 0; i < a.length; i++) s.refMass += a[i];
        } else if (s.healing) {
          s.since++;
          let se = 0, mass = 0;
          for (let i = 0; i < a.length; i++) { const d = a[i] - s.ref[i]; se += d * d; mass += a[i]; }
          const mse = se / (a.length * 65025);
          s.ok = mse < TOL ? s.ok + 1 : 0;
          if (s.ok >= OK_STEPS) s.healing = false;
          else if (s.since > MAX_HEAL || (s.since > 60 && (mass < 0.35 * s.refMass || mass > 1.8 * s.refMass)) ||
                   (s.since > STALL && mse > 5 * TOL)) regrow(s);   // stalled far from itself: start over
        }
        frames.push({ id, a, t: s.nca.t, healing: s.healing, regrows: s.regrows }); transfer.push(a.buffer);
        if (now() - t0 > budgetMs) break;
      }
      if (frames.length) post({ type: "frames", frames }, transfer);
      if ([...glyphs.values()].some(busy)) wake();
    }
    function wake() { if (!pending) { pending = true; schedule(round); } }
    return {
      handle(m) {
        const s = glyphs.get(m.id);
        if (m.type === "init") {
          try {
            glyphs.set(m.id, { nca: new NCA(m.weights), visible: true, healing: false, since: 0, ok: 0, ref: null, refMass: 0, regrows: 0 });
            order = [...glyphs.keys()];
          } catch (e) { post({ type: "error", id: m.id, message: String(e && e.message || e) }); return; }
        } else if (m.type === "reset" && s) { regrow(s); }
        else if (m.type === "disturb" && s) {
          s.nca.disturb(m.cy, m.cx, m.r, m.push, m.vy, m.vx);
          if (s.ref) { s.healing = true; s.since = 0; s.ok = 0; }   // while still growing, growth carries on
        }
        else if (m.type === "visible" && s) { s.visible = !!m.visible; }
        else if (m.type === "pause") { paused = !!m.paused; }
        wake();
      },
    };
  }

  // ---- inside a Worker: run the engine and stop here ----
  const inWorker = typeof WorkerGlobalScope !== "undefined" && typeof self !== "undefined" && self instanceof WorkerGlobalScope;
  if (inWorker) {
    const eng = Engine((m, tr) => self.postMessage(m, tr), 12, fn => setTimeout(fn, 0));
    self.onmessage = e => eng.handle(e.data);
    return;
  }

  function parseColor(css) {
    const c = document.createElement("canvas"); c.width = c.height = 1;
    const g = c.getContext("2d"); g.fillStyle = "#000"; g.fillStyle = css; g.fillRect(0, 0, 1, 1);
    const d = g.getImageData(0, 0, 1, 1).data; return [d[0], d[1], d[2]];
  }

  function startWorker(url) {
    try {
      if (typeof Worker === "undefined") return null;
      if (url) return new Worker(url);
      const src = "(" + factory.toString() + ")(self);";
      return new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
    } catch (e) { return null; }
  }

  function attachAll(items, opts) {
    opts = opts || {};
    const noop = { reset() {}, destroy() {}, reduced: false, active: false };
    if (typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches)
      return Object.assign(noop, { reduced: true });
    const glyphs = [];
    items.forEach((it, id) => {
      const w = it && it.weights, cv = it && it.canvas;
      if (!cv || !w || w.format !== "kanji-nca/1" || !w.height || !w.width) return;
      cv.width = w.width; cv.height = w.height;
      if (opts.pixelated !== false) cv.style.imageRendering = "pixelated";
      const ctx = cv.getContext("2d"); if (!ctx) return;
      glyphs.push({ id, cv, ctx, img: ctx.createImageData(w.width, w.height), H: w.height, W: w.width, w,
                    text: it.textEl || null, latest: null, shown: false, lx: null, ly: null, dv: null, failed: false });
    });
    if (!glyphs.length) return noop;
    const byId = new Map(glyphs.map(g => [g.id, g]));
    const colorOf = opts.color || (cv => getComputedStyle(cv).color);
    const radius = opts.radius || 9;
    let dead = false, raf = 0, frameNo = 0;

    // engine: one Worker for every glyph; main-thread fallback with a small per-frame budget
    function onMsg(m) {
      if (m.type === "frames") for (const f of m.frames) { const g = byId.get(f.id); if (g) g.latest = f.a; }
      else if (m.type === "error") { const g = byId.get(m.id); if (g) { g.failed = true; if (g.text) g.text.style.opacity = ""; } }
      kick();
    }
    let worker = startWorker(opts.workerUrl), send;
    if (worker) {
      worker.onmessage = e => onMsg(e.data);
      worker.onerror = () => { if (!dead) fallback(); };
      send = (m) => worker.postMessage(m);
    } else fallback();
    function fallback() {
      if (worker) { try { worker.terminate(); } catch (e) {} worker = null; }
      const eng = Engine(m => onMsg(m), opts.mainBudgetMs || 4, fn => requestAnimationFrame(() => fn()));
      send = m => eng.handle(m);
      for (const g of glyphs) { send({ type: "init", id: g.id, weights: g.w }); if (g.dv) send(g.dv); }
    }
    if (worker) for (const g of glyphs) send({ type: "init", id: g.id, weights: g.w });

    function paint() {
      raf = 0;
      if (dead) return;
      frameNo++;
      for (const g of glyphs) {
        if (g.dv) { send(g.dv); g.dv = null; }
        if (!g.latest || g.failed) continue;
        if (!g.rgb || frameNo % 60 === 0) g.rgb = parseColor(colorOf(g.cv));
        const a = g.latest, d = g.img.data, [r, gg, b] = g.rgb;
        g.latest = null;
        for (let q = 0; q < a.length; q++) { const o = 4 * q; d[o] = r; d[o + 1] = gg; d[o + 2] = b; d[o + 3] = a[q]; }
        g.ctx.putImageData(g.img, 0, 0);
        if (!g.shown) {
          g.shown = true;
          if (g.text) { g.text.style.transition = "opacity .6s ease"; g.text.style.opacity = "0"; }
        }
      }
    }
    function kick() { if (!raf && !dead) raf = requestAnimationFrame(paint); }

    // input: pointer + touch from window; at most one disturbance per glyph per frame
    function onMove(e) {
      const pt = e.touches ? e.touches[0] : e; if (!pt) return;
      for (const g of glyphs) {
        if (!g.shown) continue;
        const r = g.cv.getBoundingClientRect(); if (!r.width) continue;
        const gx = (pt.clientX - r.left) / r.width * g.W, gy = (pt.clientY - r.top) / r.height * g.H;
        const vx = g.lx === null ? 0 : gx - g.lx, vy = g.ly === null ? 0 : gy - g.ly;
        g.lx = gx; g.ly = gy;
        if (gx < -radius || gy < -radius || gx > g.W + radius || gy > g.H + radius) continue;
        const sp = Math.min(Math.hypot(vx, vy), 6);
        if (sp < 0.05) continue;
        const cl = v => Math.max(-2.5, Math.min(2.5, v * 0.6));
        g.dv = { type: "disturb", id: g.id, cy: gy, cx: gx, r: radius, push: Math.min(0.35 * sp, 2.5), vy: cl(vy), vx: cl(vx) };
        kick();
      }
    }
    function onEnd() { for (const g of glyphs) g.lx = g.ly = null; }
    window.addEventListener("pointermove", onMove, { passive: true });
    window.addEventListener("touchmove", onMove, { passive: true });
    window.addEventListener("pointerleave", onEnd, { passive: true });
    window.addEventListener("touchend", onEnd, { passive: true });
    const onVis = () => send({ type: "pause", paused: document.hidden });
    document.addEventListener("visibilitychange", onVis);
    let io = null;
    if ("IntersectionObserver" in window) {
      io = new IntersectionObserver(es => {
        for (const e of es) { const g = glyphs.find(x => x.cv === e.target); if (g) send({ type: "visible", id: g.id, visible: e.isIntersecting }); }
      });
      glyphs.forEach(g => io.observe(g.cv));
    }
    onVis();
    return {
      active: true, reduced: false,
      get mode() { return worker ? "worker" : "main"; },
      reset() { for (const g of glyphs) send({ type: "reset", id: g.id }); },
      destroy() {
        dead = true; if (raf) cancelAnimationFrame(raf);
        if (worker) worker.terminate();
        window.removeEventListener("pointermove", onMove); window.removeEventListener("touchmove", onMove);
        window.removeEventListener("pointerleave", onEnd); window.removeEventListener("touchend", onEnd);
        document.removeEventListener("visibilitychange", onVis); if (io) io.disconnect();
        for (const g of glyphs) if (g.text) g.text.style.opacity = "";
      },
    };
  }

  const attach = (opts) => attachAll([opts], opts);
  root.KanjiNCA = { NCA, Engine, attach, attachAll, decodeF16: f16 };
})(typeof self !== "undefined" ? self : this);
