/**
 * neural-bg.js
 * Interactive neural-network canvas background for randomwalker.org
 *
 * Usage:
 *   const net = initNeuralBg('nn-canvas');
 *   // later, to cleanly remove it:
 *   net.destroy();
 *
 * The canvas must be a direct child of the element you want covered,
 * positioned absolutely with inset: 0 (see README comment at the bottom).
 */

(function (global) {
  'use strict';

  // ─── Palette (thesis cover) ──────────────────────────────────────────────
  const COLORS = [
    { r: 30,  g: 200, b: 212 },   // cosmic cyan   — primary accent
    { r: 30,  g: 200, b: 212 },   // cyan again (weighted heavier)
    { r: 155, g: 111, b: 212 },   // nebula purple
    { r: 201, g: 162, b: 39  },   // orbit gold
    { r: 224, g: 90,  b: 43  },   // catalyst orange
  ];

  // ─── Default config ──────────────────────────────────────────────────────
  const DEFAULTS = {
    nodeCount:      85,     // number of particles
    connectDist:    140,    // max px between connected nodes
    mouseRadius:    180,    // repulsion radius around cursor
    repelStrength:  0.022,  // base repulsion force
    clickStrength:  0.07,   // repulsion force while mouse button held
    speed:          0.40,   // base drift speed
    friction:       0.984,  // velocity damping per frame (< 1)
    nodeSizeMin:    1.8,
    nodeSizeMax:    3.8,
    pulseSpeed:     0.016,  // radians per frame
    edgeAlphaMax:   0.38,   // max edge opacity
    nodeAlphaBase:  0.50,   // resting node opacity
    glowRadius:     6,      // px of glow halo when excited
    bgColor:        '#0a0c10',
  };

  // ─── Helpers ─────────────────────────────────────────────────────────────
  function rgba(c, a) {
    return `rgba(${c.r},${c.g},${c.b},${+a.toFixed(3)})`;
  }

  function pickColor() {
    return COLORS[Math.floor(Math.random() * COLORS.length)];
  }

  function dist2(ax, ay, bx, by) {
    const dx = ax - bx, dy = ay - by;
    return dx * dx + dy * dy;
  }

  // ─── Main factory ────────────────────────────────────────────────────────
  function initNeuralBg(canvasId, userConfig) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
      console.warn('neural-bg: canvas #' + canvasId + ' not found.');
      return null;
    }

    const cfg    = Object.assign({}, DEFAULTS, userConfig || {});
    const ctx    = canvas.getContext('2d');
    const parent = canvas.parentElement;

    let W = 0, H = 0;
    let nodes   = [];
    let rafId   = null;
    let frame   = 0;
    let mouseX  = -9999, mouseY = -9999;
    let clicking = false;
    const connectDist2 = cfg.connectDist * cfg.connectDist;
    const mouseRadius2 = cfg.mouseRadius * cfg.mouseRadius;

    // ── DPR-aware resize ──
    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = parent.clientWidth;
      H = parent.clientHeight;
      canvas.width  = W * dpr;
      canvas.height = H * dpr;
      canvas.style.width  = W + 'px';
      canvas.style.height = H + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    // ── Node factory ──
    function makeNode() {
      const angle = Math.random() * Math.PI * 2;
      const spd   = cfg.speed * (0.5 + Math.random() * 0.5);
      return {
        x:     Math.random() * W,
        y:     Math.random() * H,
        vx:    Math.cos(angle) * spd,
        vy:    Math.sin(angle) * spd,
        size:  cfg.nodeSizeMin + Math.random() * (cfg.nodeSizeMax - cfg.nodeSizeMin),
        col:   pickColor(),
        phase: Math.random() * Math.PI * 2,   // pulse offset
        amp:   0.35 + Math.random() * 0.55,   // pulse amplitude
      };
    }

    function initNodes() {
      nodes = [];
      for (let i = 0; i < cfg.nodeCount; i++) nodes.push(makeNode());
    }

    // ── Edge drawing ──
    function drawEdge(a, b, d2) {
      const alpha = (1 - Math.sqrt(d2) / cfg.connectDist) * cfg.edgeAlphaMax;
      const grad  = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
      grad.addColorStop(0, rgba(a.col, alpha));
      grad.addColorStop(1, rgba(b.col, alpha));
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = grad;
      ctx.lineWidth   = 0.65;
      ctx.stroke();
    }

    // ── Node drawing ──
    function drawNode(n, excited) {
      const pulse = 1 + Math.sin(frame * cfg.pulseSpeed + n.phase) * 0.22 * n.amp;
      const r     = n.size * pulse;
      const alpha = cfg.nodeAlphaBase + excited * (1 - cfg.nodeAlphaBase);

      // core dot
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fillStyle = rgba(n.col, alpha);
      ctx.fill();

      // glow halo (only when cursor is near)
      if (excited > 0.08) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, r + cfg.glowRadius * excited, 0, Math.PI * 2);
        ctx.strokeStyle = rgba(n.col, excited * 0.35);
        ctx.lineWidth   = 1.2;
        ctx.stroke();
      }
    }

    // ── Main animation loop ──
    function tick() {
      frame++;
      ctx.clearRect(0, 0, W, H);

      const strength = clicking ? cfg.clickStrength : cfg.repelStrength;

      // update positions + collect edges
      for (let i = 0; i < nodes.length; i++) {
        const n  = nodes[i];

        // mouse repulsion
        const mdx = n.x - mouseX;
        const mdy = n.y - mouseY;
        const md2 = mdx * mdx + mdy * mdy;

        if (md2 < mouseRadius2 && md2 > 0.01) {
          const md  = Math.sqrt(md2);
          const f   = (1 - md / cfg.mouseRadius) * strength;
          n.vx += (mdx / md) * f * 10;
          n.vy += (mdy / md) * f * 10;
        }

        // damping
        n.vx *= cfg.friction;
        n.vy *= cfg.friction;

        // clamp max speed so nodes don't fly off
        const spd = Math.sqrt(n.vx * n.vx + n.vy * n.vy);
        if (spd > 4) { n.vx = (n.vx / spd) * 4; n.vy = (n.vy / spd) * 4; }

        n.x += n.vx;
        n.y += n.vy;

        // soft boundary bounce (not a hard clamp — feels more organic)
        const margin = 20;
        if (n.x < margin)     { n.vx += (margin - n.x)     * 0.04; }
        if (n.x > W - margin) { n.vx -= (n.x - (W-margin)) * 0.04; }
        if (n.y < margin)     { n.vy += (margin - n.y)     * 0.04; }
        if (n.y > H - margin) { n.vy -= (n.y - (H-margin)) * 0.04; }

        // draw edges to later nodes (avoids double-drawing)
        for (let j = i + 1; j < nodes.length; j++) {
          const d2 = dist2(n.x, n.y, nodes[j].x, nodes[j].y);
          if (d2 < connectDist2) drawEdge(n, nodes[j], d2);
        }
      }

      // draw nodes on top of all edges
      for (const n of nodes) {
        const d2      = dist2(n.x, n.y, mouseX, mouseY);
        const excited = d2 < mouseRadius2 ? 1 - Math.sqrt(d2) / cfg.mouseRadius : 0;
        drawNode(n, excited);
      }

      rafId = requestAnimationFrame(tick);
    }

    // ── Event listeners ──
    function onMouseMove(e) {
      const rect = canvas.getBoundingClientRect();
      mouseX = e.clientX - rect.left;
      mouseY = e.clientY - rect.top;
    }

    function onMouseLeave() {
      mouseX = -9999;
      mouseY = -9999;
    }

    function onMouseDown() { clicking = true;  }
    function onMouseUp()   { clicking = false; }

    // Touch support
    function onTouchMove(e) {
      if (e.touches.length === 0) return;
      e.preventDefault();
      const rect  = canvas.getBoundingClientRect();
      const touch = e.touches[0];
      mouseX = touch.clientX - rect.left;
      mouseY = touch.clientY - rect.top;
    }

    function onTouchEnd() {
      mouseX = -9999;
      mouseY = -9999;
    }

    parent.addEventListener('mousemove',  onMouseMove,  { passive: true });
    parent.addEventListener('mouseleave', onMouseLeave, { passive: true });
    parent.addEventListener('mousedown',  onMouseDown,  { passive: true });
    window.addEventListener('mouseup',    onMouseUp,    { passive: true });
    parent.addEventListener('touchmove',  onTouchMove,  { passive: false });
    parent.addEventListener('touchend',   onTouchEnd,   { passive: true });

    // ── ResizeObserver (handles hero height changes too) ──
    const ro = new ResizeObserver(() => resize());
    ro.observe(parent);

    // ── Boot ──
    resize();
    initNodes();
    tick();

    // ── Public API ──
    return {
      /**
       * Cleanly stops the animation and removes all event listeners.
       * Call this if you remove the canvas from the DOM (e.g. SPA routing).
       */
      destroy() {
        cancelAnimationFrame(rafId);
        ro.disconnect();
        parent.removeEventListener('mousemove',  onMouseMove);
        parent.removeEventListener('mouseleave', onMouseLeave);
        parent.removeEventListener('mousedown',  onMouseDown);
        window.removeEventListener('mouseup',    onMouseUp);
        parent.removeEventListener('touchmove',  onTouchMove);
        parent.removeEventListener('touchend',   onTouchEnd);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      },

      /**
       * Live-tweak any config value after init.
       * e.g.  net.set({ nodeCount: 120, connectDist: 160 })
       * Note: nodeCount changes take effect on next resize/reinit.
       */
      set(patch) {
        Object.assign(cfg, patch);
      },

      /** Pause / resume the animation loop. */
      pause()  { cancelAnimationFrame(rafId); rafId = null; },
      resume() { if (!rafId) tick(); },
    };
  }

  // ── Expose globally ──
  global.initNeuralBg = initNeuralBg;

}(window));

/*
 * ─── HTML setup ──────────────────────────────────────────────────────────────
 *
 * In index.html, inside .hero, add the canvas as the FIRST child:
 *
 *   <section class="hero">
 *     <canvas id="nn-canvas"></canvas>
 *     <div class="hero__orb"></div>
 *     <div class="hero__orb-secondary"></div>
 *     <div class="container">
 *       <div class="hero__content"> ... </div>
 *     </div>
 *   </section>
 *
 * ─── CSS setup (add to styles.css) ───────────────────────────────────────────
 *
 *   #nn-canvas {
 *     position: absolute;
 *     inset: 0;
 *     width: 100%;
 *     height: 100%;
 *     z-index: 0;
 *     pointer-events: none;   <- lets clicks pass through to buttons
 *   }
 *
 *   .hero__content,
 *   .hero__orb,
 *   .hero__orb-secondary {
 *     position: relative;
 *     z-index: 1;
 *   }
 *
 * ─── Script tag (bottom of <body>) ───────────────────────────────────────────
 *
 *   <script src="/neural-bg.js"></script>
 *   <script>
 *     const neuralNet = initNeuralBg('nn-canvas');
 *
 *     // Optional: pause when tab is hidden (saves CPU)
 *     document.addEventListener('visibilitychange', () => {
 *       document.hidden ? neuralNet.pause() : neuralNet.resume();
 *     });
 *   </script>
 *
 * ─── Optional: use on the CV page too ────────────────────────────────────────
 *
 *   Add a second canvas inside .cv-header, give it id="nn-canvas-cv",
 *   and call:  initNeuralBg('nn-canvas-cv', { nodeCount: 55, connectDist: 110 });
 *   The factory is multi-instance safe — each call is fully independent.
 */
