/* Animated background of drifting rational-function curves.
 *
 * Each curve is a genuine rational function of x: a sum of bump terms
 * u / (1 + u^2) whose arguments drift slowly over time. Runs only on pages
 * containing the #autosi-bg marker (the landing page), honors
 * prefers-reduced-motion, and adapts its colors to the theme's light/dark
 * mode on every frame.
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

    // One curve = a base height plus 3 rational bump terms.
    var CURVES = 8;
    var curves = [];
    for (var i = 0; i < CURVES; i++) {
      var terms = [];
      for (var k = 0; k < 3; k++) {
        terms.push({
          amp: 0.05 + Math.random() * 0.10,        // bump height (fraction of h)
          freq: 0.8 + Math.random() * 1.6,         // horizontal frequency
          phase: Math.random() * 8 - 4,            // initial shift
          speed: (Math.random() * 0.14 + 0.04) * (Math.random() < 0.5 ? -1 : 1)
        });
      }
      curves.push({
        base: 0.12 + (i / (CURVES - 1)) * 0.72,    // vertical placement
        hue: 226 + Math.random() * 40,             // indigo..violet range
        terms: terms
      });
    }

    function draw(t) {
      var dark = document.documentElement.classList.contains("dark");
      ctx.clearRect(0, 0, width, height);
      ctx.lineWidth = 1.4;

      for (var i = 0; i < curves.length; i++) {
        var c = curves[i];
        ctx.strokeStyle = dark
          ? "hsla(" + c.hue + ", 75%, 72%, 0.20)"
          : "hsla(" + c.hue + ", 60%, 45%, 0.16)";
        ctx.beginPath();
        for (var x = 0; x <= width; x += 4) {
          var nx = (x / width) * 6 - 3;            // map to [-3, 3]
          var y = c.base * height;
          for (var k = 0; k < c.terms.length; k++) {
            var term = c.terms[k];
            var u = nx * term.freq + term.phase + t * term.speed;
            y += term.amp * height * (u / (1 + u * u));
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
