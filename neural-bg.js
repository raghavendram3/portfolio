/**
 * neural-bg.js  —  randomwalker.org
 *
 * Usage:
 *   const net = initNeuralBg('nn-canvas');
 *   net.destroy();   // clean up later if needed
 *
 * CSS you need (already in index.html <style> block):
 *   #nn-canvas { position:absolute; inset:0; width:100%; height:100%;
 *                z-index:0; pointer-events:none; display:block; }
 *   .hero__orb, .hero__orb-secondary, .hero .container
 *                { position:relative; z-index:1; }
 */

(function (global) {
  'use strict';

  /* ── Palette (PhD thesis cover) ── */
  var PALETTE = [
    { r: 30,  g: 200, b: 212 },   // cosmic cyan   (weighted 3×)
    { r: 30,  g: 200, b: 212 },
    { r: 30,  g: 200, b: 212 },
    { r: 155, g: 111, b: 212 },   // nebula purple
    { r: 201, g: 162, b: 39  },   // orbit gold
    { r: 224, g: 90,  b: 43  },   // catalyst orange
  ];

  var DEFAULTS = {
    nodeCount:     80,
    connectDist:   140,
    mouseRadius:   175,
    repelForce:    0.022,
    clickForce:    0.065,
    speed:         0.38,
    friction:      0.983,
    sizeMin:       1.8,
    sizeMax:       3.8,
    pulseSpeed:    0.016,
    edgeAlpha:     0.36,
    nodeAlpha:     0.52,
    glowSize:      6,
  };

  /* ── helpers ── */
  function rgba(c, a) {
    return 'rgba(' + c.r + ',' + c.g + ',' + c.b + ',' + a.toFixed(3) + ')';
  }
  function rndColor() { return PALETTE[Math.floor(Math.random() * PALETTE.length)]; }
  function d2(ax, ay, bx, by) { var dx=ax-bx, dy=ay-by; return dx*dx+dy*dy; }

  /* ── main factory ── */
  function initNeuralBg(canvasId, userCfg) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) { console.warn('neural-bg: #' + canvasId + ' not found'); return null; }

    var cfg = {};
    for (var k in DEFAULTS) cfg[k] = DEFAULTS[k];
    if (userCfg) for (var k in userCfg) cfg[k] = userCfg[k];

    var ctx    = canvas.getContext('2d');
    var parent = canvas.parentElement;
    var W = 0, H = 0, nodes = [], raf = null, frame = 0, clicking = false;
    var mouseX = -9999, mouseY = -9999;
    var cd2 = cfg.connectDist * cfg.connectDist;
    var mr2 = cfg.mouseRadius * cfg.mouseRadius;

    /* ── sizing ── */
    function resize() {
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = parent.clientWidth;
      H = parent.clientHeight;
      if (!W || !H) return;
      canvas.width  = W * dpr;
      canvas.height = H * dpr;
      canvas.style.width  = W + 'px';
      canvas.style.height = H + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    /* ── nodes ── */
    function makeNode() {
      var angle = Math.random() * Math.PI * 2;
      var spd   = cfg.speed * (0.5 + Math.random() * 0.5);
      return {
        x: Math.random() * W, y: Math.random() * H,
        vx: Math.cos(angle) * spd, vy: Math.sin(angle) * spd,
        size:  cfg.sizeMin + Math.random() * (cfg.sizeMax - cfg.sizeMin),
        col:   rndColor(),
        phase: Math.random() * Math.PI * 2,
        amp:   0.35 + Math.random() * 0.55,
      };
    }
    function initNodes() { nodes = []; for (var i=0; i<cfg.nodeCount; i++) nodes.push(makeNode()); }

    /* ── draw ── */
    function drawEdge(a, b, dist2val) {
      var alpha = (1 - Math.sqrt(dist2val) / cfg.connectDist) * cfg.edgeAlpha;
      var g = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
      g.addColorStop(0, rgba(a.col, alpha));
      g.addColorStop(1, rgba(b.col, alpha));
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = g; ctx.lineWidth = 0.65; ctx.stroke();
    }

    function drawNode(n, excited) {
      var pulse = 1 + Math.sin(frame * cfg.pulseSpeed + n.phase) * 0.22 * n.amp;
      var r     = n.size * pulse;
      var alpha = cfg.nodeAlpha + excited * (1 - cfg.nodeAlpha);
      ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fillStyle = rgba(n.col, alpha); ctx.fill();
      if (excited > 0.08) {
        ctx.beginPath(); ctx.arc(n.x, n.y, r + cfg.glowSize * excited, 0, Math.PI * 2);
        ctx.strokeStyle = rgba(n.col, excited * 0.32);
        ctx.lineWidth = 1.2; ctx.stroke();
      }
    }

    /* ── loop ── */
    function tick() {
      frame++;
      ctx.clearRect(0, 0, W, H);
      var force = clicking ? cfg.clickForce : cfg.repelForce;

      for (var i = 0; i < nodes.length; i++) {
        var n = nodes[i];

        /* mouse repulsion */
        var mdx = n.x - mouseX, mdy = n.y - mouseY;
        var md2 = mdx*mdx + mdy*mdy;
        if (md2 < mr2 && md2 > 0.01) {
          var md = Math.sqrt(md2);
          var f  = (1 - md / cfg.mouseRadius) * force;
          n.vx += (mdx / md) * f * 10;
          n.vy += (mdy / md) * f * 10;
        }

        /* damping + speed cap */
        n.vx *= cfg.friction; n.vy *= cfg.friction;
        var spd = Math.sqrt(n.vx*n.vx + n.vy*n.vy);
        if (spd > 4) { n.vx = n.vx/spd*4; n.vy = n.vy/spd*4; }

        n.x += n.vx; n.y += n.vy;

        /* soft boundary push */
        var m = 20;
        if (n.x < m)     n.vx += (m - n.x)     * 0.04;
        if (n.x > W - m) n.vx -= (n.x-(W-m))   * 0.04;
        if (n.y < m)     n.vy += (m - n.y)     * 0.04;
        if (n.y > H - m) n.vy -= (n.y-(H-m))   * 0.04;

        /* edges */
        for (var j = i+1; j < nodes.length; j++) {
          var dist = d2(n.x, n.y, nodes[j].x, nodes[j].y);
          if (dist < cd2) drawEdge(n, nodes[j], dist);
        }
      }

      /* nodes on top of edges */
      for (var i = 0; i < nodes.length; i++) {
        var n = nodes[i];
        var dist = d2(n.x, n.y, mouseX, mouseY);
        var excited = dist < mr2 ? 1 - Math.sqrt(dist) / cfg.mouseRadius : 0;
        drawNode(n, excited);
      }

      raf = requestAnimationFrame(tick);
    }

    /* ── mouse/touch — attached to DOCUMENT so z-index never blocks them ── */
    function getPos(e) {
      var rect = canvas.getBoundingClientRect();
      return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }

    function onMove(e) {
      var p = getPos(e);
      /* only "active" when cursor is inside the canvas bounding box */
      var rect = canvas.getBoundingClientRect();
      if (e.clientX >= rect.left && e.clientX <= rect.right &&
          e.clientY >= rect.top  && e.clientY <= rect.bottom) {
        mouseX = p.x; mouseY = p.y;
      } else {
        mouseX = -9999; mouseY = -9999;
      }
    }

    function onDown() { clicking = true;  }
    function onUp()   { clicking = false; }

    function onTouch(e) {
      if (!e.touches.length) return;
      var t    = e.touches[0];
      var rect = canvas.getBoundingClientRect();
      mouseX   = t.clientX - rect.left;
      mouseY   = t.clientY - rect.top;
    }
    function onTouchEnd() { mouseX = -9999; mouseY = -9999; }

    document.addEventListener('mousemove',  onMove,    { passive: true });
    document.addEventListener('mousedown',  onDown,    { passive: true });
    document.addEventListener('mouseup',    onUp,      { passive: true });
    document.addEventListener('touchmove',  onTouch,   { passive: true });
    document.addEventListener('touchend',   onTouchEnd,{ passive: true });

    /* ── resize ── */
    var ro = new ResizeObserver(function() { resize(); });
    ro.observe(parent);

    /* ── boot ── */
    resize();
    initNodes();
    tick();

    console.log('[neural-bg] running — ' + cfg.nodeCount + ' nodes on #' + canvasId);

    /* ── public API ── */
    return {
      destroy: function() {
        cancelAnimationFrame(raf);
        ro.disconnect();
        document.removeEventListener('mousemove',  onMove);
        document.removeEventListener('mousedown',  onDown);
        document.removeEventListener('mouseup',    onUp);
        document.removeEventListener('touchmove',  onTouch);
        document.removeEventListener('touchend',   onTouchEnd);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        console.log('[neural-bg] destroyed');
      },
      pause:  function() { cancelAnimationFrame(raf); raf = null; },
      resume: function() { if (!raf) tick(); },
      set:    function(patch) { for (var k in patch) cfg[k] = patch[k]; },
    };
  }

  global.initNeuralBg = initNeuralBg;

}(window));
