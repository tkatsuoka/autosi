/* Animated background of drifting rational-function curves.
 *
 * Each curve is a genuine rational function of x: a sum of bump terms
 * u / (1 + u^2) whose arguments drift slowly over time. All horizontal
 * scales are defined in pixels (wavelength, amplitude, drift speed), so
 * the pattern looks and moves the same at every window size instead of
 * stretching with the viewport. Runs only on pages containing the
 * #autosi-bg marker (the landing page), honors prefers-reduced-motion,
 * and adapts its colors to the theme's light/dark mode on every frame.
 */
(function () {
  "use strict";

  function init() {
    var host = document.getElementById("autosi-bg");
    if (!host) return;

    // Anchor to the document (initial containing block), not to any
    // positioned ancestor inside the article column.
    document.body.appendChild(host);

    var canvas = document.createElement("canvas");
    host.appendChild(canvas);
    var ctx = canvas.getContext("2d");

    var width = 0, height = 0, dpr = 1;

    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = host.clientWidth;
      height = host.clientHeight;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    // One curve = a base height plus layered rational bump terms: a broad
    // sweep, two mid-scale bumps, and a fine ripple. Wavelengths,
    // amplitudes, and speeds are in pixels, independent of window size.
    // The sweeps are large enough to cross neighboring curves.
    var CURVES = 13;
    var LAYERS = [
      { n: 1, wl: [450, 800], amp: [120, 220] },  // broad sweep
      { n: 3, wl: [200, 420], amp: [50, 110] },   // mid-scale bumps
      { n: 1, wl: [110, 200], amp: [16, 34] }     // fine ripple
    ];
    var curves = [];
    for (var i = 0; i < CURVES; i++) {
      var terms = [];
      for (var l = 0; l < LAYERS.length; l++) {
        var L = LAYERS[l];
        for (var k = 0; k < L.n; k++) {
          terms.push({
            wl: L.wl[0] + Math.random() * (L.wl[1] - L.wl[0]),
            amp: L.amp[0] + Math.random() * (L.amp[1] - L.amp[0]),
            c0: Math.random(),                 // initial center [fraction]
            phase: Math.random() * Math.PI * 2,    // modulation phase
            speed: (12 + Math.random() * 26)   // horizontal drift [px/s]
                   * (Math.random() < 0.5 ? -1 : 1),
            bfreq: 0.3 + Math.random() * 0.5,      // amplitude cycle [rad/s]
            wfreq: 0.15 + Math.random() * 0.3      // width wobble [rad/s]
          });
        }
      }
      curves.push({
        // One curve per band, jittered inside it: covers the full height
        base: (i + Math.random()) / CURVES,
        slope: (Math.random() * 0.7 - 0.35),       // random tilt (about +-19 deg)
        hue: 226 + Math.random() * 40,             // indigo..violet range
        terms: terms
      });
    }

    function draw(t) {
      var dark = document.documentElement.classList.contains("dark");
      ctx.clearRect(0, 0, width, height);
      ctx.lineWidth = 1.6;

      // Cap amplitudes only on very short viewports
      var ampCap = Math.max(60, height * 0.24);

      for (var i = 0; i < curves.length; i++) {
        var c = curves[i];
        ctx.strokeStyle = dark
          ? "hsla(" + c.hue + ", 75%, 72%, 0.26)"
          : "hsla(" + c.hue + ", 60%, 45%, 0.20)";
        ctx.beginPath();
        // Bump centers wrap around an extended span so that a bump
        // drifting off one edge re-enters from the other; without the
        // wrap every bump eventually leaves the screen for good and the
        // curves degenerate into straight lines.
        var MARGIN = 500;
        var span = width + 2 * MARGIN;
        for (var x = 0; x <= width; x += 4) {
          var y = c.base * height + c.slope * (x - width / 2);
          for (var k = 0; k < c.terms.length; k++) {
            var term = c.terms[k];
            // Bump width breathes by +-25% over time
            var wl = term.wl * (1 + 0.25 * Math.sin(t * term.wfreq + term.phase));
            var center = (((term.c0 * span + t * term.speed) % span) + span) % span - MARGIN;
            var u = (x - center) / wl;
            // Envelope that fades to zero at both wrap boundaries, so the
            // jump from one edge to the other never causes a visible snap
            var env = Math.sin(Math.PI * (center + MARGIN) / span);
            // Amplitude swings between -0.4x and +1.0x: bumps grow,
            // flatten, and invert, so the waveform keeps reshaping
            var amp = Math.min(term.amp, ampCap) * env
              * (0.3 + 0.7 * Math.sin(t * term.bfreq + term.phase * 2));
            y += amp * (u / (1 + u * u));
          }
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
    }

    var reduced = window.matchMedia
      && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    resize();
    window.addEventListener("resize", function () {
      resize();
      if (reduced) draw(0);
    });

    if (reduced) {
      draw(0);                                     // single static frame
    } else {
      var loop = function (now) {
        draw(now / 1000);
        window.requestAnimationFrame(loop);
      };
      window.requestAnimationFrame(loop);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
