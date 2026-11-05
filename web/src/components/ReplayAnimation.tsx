import { useEffect, useRef, useState } from "react";

// Landing-page animation: what Run It Back does, in three beats.
//   1. Play one game: a lead tracker is written one play at a time.
//   2. Run it back: rewind to the start and replay the matchup a thousand times.
//   3. Count the endings: the final scores pile up into a sideways histogram.
// The games are toy random walks drawn in the browser, not model output; they only illustrate the idea.

type SportId = "nfl" | "nba";
// margins[i]: home lead after play i; at[i]: when play i happens, as a fraction of the game (plays take uneven time)
interface Game { margins: Int16Array; at: Float32Array; text?: string[] }
interface Scene { sport: SportId; games: Game[]; hist: number[]; histMax: number; homeWin: number; awayWin: number }

const N_GAMES = 1000;
// range: points of lead shown either side of a tie; bin: histogram bin width; grid: gridline leads
const SPORTS: Record<SportId, { name: string; start: string; periods: number; periodSecs: number; steps: number;
  range: number; bin: number; grid: number[] }> = {
  nfl: { name: "NFL", start: "Kickoff", periods: 4, periodSecs: 900, steps: 56, range: 32, bin: 3, grid: [-21, -7, 7, 21] },
  nba: { name: "NBA", start: "Tip-off", periods: 4, periodSecs: 720, steps: 200, range: 44, bin: 4, grid: [-30, -10, 10, 30] },
};

// Timeline (ms)
const HERO = 2800, REWIND_PAUSE = 150, REWIND_END = 3500, ENS_START = 3600;
const SLOW_N = 4, SLOW_GAP = 250, SLOW_DUR = 550, FAST_SPAN = 3200, FAST_DUR = 420;
const ENS_END = ENS_START + SLOW_N * SLOW_GAP + FAST_SPAN + FAST_DUR;
const READOUT = ENS_END + 200, FADE = READOUT + 5000, LOOP = FADE + 700;

const spawnAt = (k: number) => ENS_START + (k <= SLOW_N
  ? (k - 1) * SLOW_GAP
  : SLOW_N * SLOW_GAP + ((k - SLOW_N) / (N_GAMES - SLOW_N)) * FAST_SPAN);
const durOf = (k: number) => (k <= SLOW_N ? SLOW_DUR : FAST_DUR);

function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Cumulative play times, normalized to 0..1, so replays don't all change score at the same instants.
function timing(r: () => number, steps: number) {
  const at = new Float32Array(steps + 1);
  for (let i = 1; i <= steps; i++) at[i] = at[i - 1] + 0.4 + r() * 1.2;
  for (let i = 1; i <= steps; i++) at[i] /= at[steps];
  return at;
}

const pick = <T,>(r: () => number, xs: T[]) => xs[Math.floor(r() * xs.length)];

function nflGame(r: () => number, withText: boolean): Game {
  const { steps } = SPORTS.nfl;
  const margins = new Int16Array(steps + 1);
  const text: string[] = [];
  let m = 0;
  for (let i = 1; i <= steps; i++) {
    const team = r() < 0.57 ? 1 : -1;
    const who = team > 0 ? "Home" : "Away";
    if (r() < 0.1) {
      const td = r() < 0.62;
      m += team * (td ? 7 : 3);
      if (withText) text.push(`${who} ${td ? "touchdown!" : "field goal"}`);
    } else if (withText) {
      const y = 1 + Math.floor(r() * 18);
      text.push(`${who} ${pick(r, [`pass complete, ${y} yards`, `run, ${Math.min(y, 9)} yards`, "pass incomplete", "punt", `run, ${Math.min(y, 6)} yards`, "sacked"])}`);
    }
    margins[i] = m;
  }
  if (m === 0) { // overtime: first score wins
    const team = r() < 0.57 ? 1 : -1;
    margins[steps] = team * 3;
    if (withText) text[steps - 1] = `${team > 0 ? "Home" : "Away"} field goal in overtime!`;
  }
  return { margins, at: timing(r, steps), text: withText ? text : undefined };
}

function nbaGame(r: () => number, withText: boolean): Game {
  const { steps } = SPORTS.nba;
  const margins = new Int16Array(steps + 1);
  const text: string[] = [];
  let m = 0;
  for (let i = 1; i <= steps; i++) {
    const team = i % 2 ? 1 : -1;
    const who = team > 0 ? "Home" : "Away";
    const p2 = team > 0 ? 0.37 : 0.355, p3 = team > 0 ? 0.12 : 0.115;
    const x = r();
    if (x < p3) {
      m += team * 3;
      if (withText) text.push(`${who} three-pointer`);
    } else if (x < p3 + p2) {
      m += team * 2;
      if (withText) text.push(`${who} ${pick(r, ["layup", "dunk", "jump shot", "floater"])}`);
    } else if (withText) {
      text.push(`${who} ${pick(r, ["missed three", "missed jumper", "turnover", "shot blocked"])}`);
    }
    margins[i] = m;
  }
  if (m === 0) { // overtime, compressed into the last possession
    const team = r() < 0.53 ? 1 : -1;
    margins[steps] = team * 2;
    if (withText) text[steps - 1] = `${team > 0 ? "Home" : "Away"} wins it in overtime!`;
  }
  return { margins, at: timing(r, steps), text: withText ? text : undefined };
}

const leadChanges = (g: Game) => {
  let n = 0, last = 0;
  for (const m of g.margins) {
    const s = Math.sign(m);
    if (s && last && s !== last) n++;
    if (s) last = s;
  }
  return n;
};

const nBins = (sport: SportId) => Math.ceil(SPORTS[sport].range / SPORTS[sport].bin);
// Bin index for a final margin: 0 is a tie, ±1 is a win by 1..bin points, and so on; the end bins absorb blowouts.
const binOf = (sport: SportId, m: number) => {
  if (m === 0) return 0;
  return Math.sign(m) * Math.min(Math.ceil(Math.abs(m) / SPORTS[sport].bin), nBins(sport));
};

function makeScene(sport: SportId, seed: number): Scene {
  const gen = sport === "nfl" ? nflGame : nbaGame;
  const r = rng(seed);
  // The first replay is the one we narrate: pick a close game with some back-and-forth.
  let hero = gen(r, true);
  for (let tries = 0; tries < 200; tries++) {
    const f = hero.margins[hero.margins.length - 1];
    if (leadChanges(hero) >= 2 && f !== 0 && Math.abs(f) <= 10) break;
    hero = gen(r, true);
  }
  const games = [hero];
  for (let k = 1; k < N_GAMES; k++) games.push(gen(r, false));
  const NB = nBins(sport);
  const hist = new Array(2 * NB + 1).fill(0);
  let homeWin = 0, awayWin = 0;
  for (const g of games) {
    const f = g.margins[g.margins.length - 1];
    hist[binOf(sport, f) + NB]++;
    if (f > 0) homeWin++; else if (f < 0) awayWin++;
  }
  return { sport, games, hist, histMax: Math.max(...hist), homeWin, awayWin };
}

const clamp = (x: number, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const ease = (x: number) => 1 - (1 - x) ** 3;

interface Colors { text: string; muted: string; grid: string; ref: string; home: string; away: string; surface: string; font: string }
function readColors(el: Element): Colors {
  const cs = getComputedStyle(el);
  const v = (n: string) => cs.getPropertyValue(n).trim();
  return { text: v("--text"), muted: v("--muted"), grid: v("--grid"), ref: v("--ref"), home: v("--series-1"),
    away: v("--series-2"), surface: v("--surface"), font: cs.fontFamily };
}

function clock(sport: SportId, frac: number) {
  const { periods, periodSecs } = SPORTS[sport];
  const total = periods * periodSecs;
  const elapsed = Math.min(frac * total, total - 1);
  const q = Math.floor(elapsed / periodSecs) + 1;
  const left = Math.ceil(periodSecs - (elapsed % periodSecs));
  return `Q${q} ${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`;
}

const STEPS = [
  { h: "Play one game", p: "The model writes a game one play at a time. Each play depends on everything that happened before it." },
  { h: "Run it back", p: "Rewind to the start and play it again. Same teams, same start, different breaks. Then do it a thousand times." },
  { h: "Count the endings", p: "Stack up the final scores. How often each team wins, and by how much, is the forecast." },
];

export default function ReplayAnimation() {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const counterRef = useRef<HTMLSpanElement>(null);
  const tickerRef = useRef<HTMLSpanElement>(null);
  const readoutRef = useRef<HTMLDivElement>(null);
  const [phase, setPhase] = useState(0);
  const [sport, setSport] = useState<SportId>("nfl");
  const [paused, setPaused] = useState(false);
  const pausedRef = useRef(false);
  pausedRef.current = paused;

  useEffect(() => {
    const canvas = canvasRef.current!, wrap = wrapRef.current!;
    const ctx = canvas.getContext("2d")!;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    const baked = document.createElement("canvas");
    const bctx = baked.getContext("2d")!;

    let loop = 0;
    let scene = makeScene("nfl", 7);
    let t = 0, last = performance.now(), raf = 0, visible = true;
    let W = 0, H = 0, dpr = 1;
    let nBaked = 0; // ensemble games already drawn into `baked` (game indices 1..nBaked)
    let colorKey = "";
    let colors = readColors(wrap);
    // Relative luminance of the surface colour: works for any theme the stylesheet defines.
    const isDark = () => {
      const m = colors.surface.match(/^#([0-9a-f]{6})$/i);
      if (!m) return false;
      const n = parseInt(m[1], 16);
      return 0.2126 * (n >> 16) + 0.7152 * ((n >> 8) & 255) + 0.0722 * (n & 255) < 128;
    };
    const clearBaked = () => {
      nBaked = 0;
      bctx.setTransform(1, 0, 0, 1, 0, 0);
      bctx.clearRect(0, 0, baked.width, baked.height);
    };
    let lastText = "", lastPhase = -1, lastSport: SportId | "" = "";

    // Layout
    let x0 = 0, x1 = 0, hx = 0, hw = 0, yMid = 0, ys = 1, range = SPORTS.nfl.range;
    const layout = () => {
      const narrow = W < 520;
      x0 = narrow ? 8 : 92;
      x1 = W * (narrow ? 0.7 : 0.76);
      hx = x1 + 10;
      hw = W - hx - 8;
      yMid = H / 2;
      ys = (H / 2 - 20) / range;
    };
    const X = (f: number) => x0 + (x1 - x0) * f;
    const Y = (m: number) => yMid - clamp(m, -range, range) * ys;

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = wrap.clientWidth;
      H = Math.round(clamp(W * 0.42, 220, 330));
      canvas.width = baked.width = Math.round(W * dpr);
      canvas.height = baked.height = Math.round(H * dpr);
      canvas.style.height = `${H}px`;
      layout();
      clearBaked();
    };

    const tracePath = (c: CanvasRenderingContext2D, g: Game, upto: number) => {
      const steps = g.margins.length - 1;
      const n = Math.min(Math.floor(upto), steps);
      c.beginPath();
      c.moveTo(X(0), Y(0));
      for (let i = 1; i <= n; i++) {
        // step shape: the lead changes at the moment of the play
        c.lineTo(X(g.at[i]), Y(g.margins[i - 1]));
        c.lineTo(X(g.at[i]), Y(g.margins[i]));
      }
      const f = upto - n;
      const xe = n < steps ? X(g.at[n] + (g.at[n + 1] - g.at[n]) * f) : X(1);
      if (n < steps && f > 0) c.lineTo(xe, Y(g.margins[n]));
      return { x: xe, y: Y(g.margins[n]) };
    };

    const finalOf = (g: Game) => g.margins[g.margins.length - 1];
    const lineColor = (f: number) => (f > 0 ? colors.home : f < 0 ? colors.away : colors.ref);

    const bakeUpTo = (k: number) => {
      bctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      bctx.lineWidth = 1;
      bctx.lineJoin = "round";
      // Faint enough that a thousand overlapping lines read as a cloud, not a solid block. Thin lines
      // lose more contrast on a dark surface, so they get a little more ink there.
      const base = (W < 520 ? 0.05 : 0.06) * (isDark() ? 1.4 : 1);
      for (let i = nBaked + 1; i <= k; i++) {
        const g = scene.games[i];
        // the first few replays were watched one by one, so they stay legible
        bctx.globalAlpha = i <= SLOW_N ? 0.3 : base;
        bctx.strokeStyle = lineColor(finalOf(g));
        tracePath(bctx, g, g.margins.length - 1);
        bctx.stroke();
      }
      bctx.globalAlpha = 1;
      nBaked = Math.max(nBaked, k);
    };

    const setText = (s: string) => {
      if (s !== lastText && tickerRef.current) { tickerRef.current.textContent = s; lastText = s; }
    };

    const draw = () => {
      colors = readColors(wrap);
      const key = `${colors.text}|${colors.home}|${colors.away}`;
      if (key !== colorKey) { // theme changed: re-bake the finished games in the new palette
        colorKey = key;
        clearBaked();
      }

      const { games } = scene;
      const hero = games[0];
      const steps = hero.margins.length - 1;
      const spec = SPORTS[scene.sport];
      if (range !== spec.range) { range = spec.range; layout(); clearBaked(); }
      const NB = nBins(scene.sport), BIN = spec.bin;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const fadeOut = 1 - clamp((t - FADE) / (LOOP - FADE));
      const fadeIn = clamp(t / 300);
      ctx.globalAlpha = Math.min(fadeOut, fadeIn);

      // Axes
      ctx.font = `600 11px ${colors.font}`;
      ctx.strokeStyle = colors.grid;
      ctx.lineWidth = 1;
      for (const m of spec.grid) {
        ctx.beginPath(); ctx.moveTo(x0, Y(m)); ctx.lineTo(x1, Y(m)); ctx.stroke();
      }
      ctx.strokeStyle = colors.ref;
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(x0, yMid); ctx.lineTo(W - 4, yMid); ctx.stroke();
      ctx.setLineDash([]);
      ctx.textBaseline = "middle";
      if (x0 > 40) {
        ctx.textAlign = "right";
        ctx.fillStyle = colors.home; ctx.fillText("Home ahead ↑", x0 - 10, Y(range * 0.6));
        ctx.fillStyle = colors.muted; ctx.fillText("Tied", x0 - 10, yMid);
        ctx.fillStyle = colors.away; ctx.fillText("Away ahead ↓", x0 - 10, Y(-range * 0.6));
      } else {
        ctx.textAlign = "left";
        ctx.fillStyle = colors.home; ctx.fillText("Home ahead ↑", x0 + 2, 10);
        ctx.fillStyle = colors.away; ctx.fillText("Away ahead ↓", x0 + 2, H - 10);
      }
      ctx.fillStyle = colors.muted;
      ctx.textBaseline = "alphabetic";
      ctx.textAlign = "left";
      if (x0 > 40) ctx.fillText(spec.start, x0, H - 4);
      ctx.textAlign = "right";
      ctx.fillText("Final", x1, H - 4);
      ctx.textAlign = "left";
      ctx.fillText(W < 520 ? "Final scores" : "Final scores →", hx + 2, 12);

      // Ensemble: finished games come from the baked layer, games in flight are drawn live.
      let kDone = 0, kStarted = 0;
      if (t >= ENS_START) {
        for (let k = 1; k < N_GAMES; k++) {
          if (spawnAt(k) > t) break;
          kStarted = k;
          if (spawnAt(k) + durOf(k) <= t) kDone = k;
        }
      }
      if (kDone < nBaked) clearBaked();
      bakeUpTo(kDone);
      ctx.drawImage(baked, 0, 0, W, H);

      ctx.lineJoin = "round";
      for (let k = kDone + 1; k <= kStarted; k++) {
        const g = games[k];
        const p = ease(clamp((t - spawnAt(k)) / durOf(k)));
        ctx.globalAlpha = Math.min(fadeOut, k <= SLOW_N ? 0.85 : 0.4);
        ctx.strokeStyle = lineColor(finalOf(g));
        ctx.lineWidth = k <= SLOW_N ? 1.5 : 1;
        const tip = tracePath(ctx, g, p * (g.margins.length - 1));
        ctx.stroke();
        if (k <= SLOW_N) {
          ctx.fillStyle = ctx.strokeStyle;
          ctx.beginPath(); ctx.arc(tip.x, tip.y, 2.5, 0, Math.PI * 2); ctx.fill();
        }
      }
      ctx.globalAlpha = Math.min(fadeOut, fadeIn);

      // Histogram of final margins, fed by every game that has finished (the hero counts once it ends).
      const counts = new Array(2 * NB + 1).fill(0);
      const heroDone = t >= HERO;
      if (heroDone) counts[binOf(scene.sport, finalOf(hero)) + NB]++;
      for (let k = 1; k <= kDone; k++) counts[binOf(scene.sport, finalOf(games[k])) + NB]++;
      for (let b = -NB; b <= NB; b++) {
        const c = counts[b + NB];
        if (!c) continue;
        const top = b === 0 ? Y(0.5) : b > 0 ? Y(b * BIN + 0.5) : Y((b + 1) * BIN - 0.5);
        const bot = b === 0 ? Y(-0.5) : b > 0 ? Y((b - 1) * BIN + 0.5) : Y(b * BIN - 0.5);
        const len = Math.max(2, (c / scene.histMax) * hw);
        ctx.fillStyle = b > 0 ? colors.home : b < 0 ? colors.away : colors.ref;
        ctx.beginPath();
        ctx.roundRect(hx, top + 0.5, len, Math.max(bot - top - 1, 1), [0, 2, 2, 0]);
        ctx.fill();
      }

      // The narrated game: drawn play by play, then rewound, then left as a ghost.
      let heroUpto: number;
      if (t < HERO) heroUpto = (t / HERO) * steps;
      else heroUpto = steps * (1 - ease(clamp((t - HERO - REWIND_PAUSE) / (REWIND_END - HERO - REWIND_PAUSE))));
      // ghost of the full game once it has been played
      if (t >= HERO) {
        ctx.globalAlpha = Math.min(fadeOut, 0.22 + 0.5 * (1 - clamp((t - HERO) / 2500)));
        ctx.strokeStyle = colors.text;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        tracePath(ctx, hero, steps);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      if (heroUpto > 0 || t < HERO) {
        ctx.globalAlpha = Math.min(fadeOut, fadeIn);
        ctx.strokeStyle = colors.text;
        ctx.lineWidth = 2.25;
        const tip = tracePath(ctx, hero, heroUpto);
        ctx.stroke();
        ctx.fillStyle = colors.text;
        ctx.strokeStyle = colors.surface;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(tip.x, tip.y, 4.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      }
      // Rewind badge
      if (t >= HERO && t < ENS_START + 400) {
        const a = clamp((t - HERO) / 200) * (1 - clamp((t - ENS_START) / 400));
        ctx.globalAlpha = a * fadeOut;
        ctx.font = `700 13px ${colors.font}`;
        const label = "⟲  Run it back";
        const tw = ctx.measureText(label).width;
        const bx = x0 + 10, by = 14;
        ctx.fillStyle = colors.text;
        ctx.beginPath(); ctx.roundRect(bx, by, tw + 20, 24, 12); ctx.fill();
        ctx.fillStyle = colors.surface;
        ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText(label, bx + 10, by + 12.5);
      }
      ctx.globalAlpha = 1;

      // Text: counter, ticker, readout, step highlight
      const played = t < HERO ? 1 : t < ENS_START ? 1 : 1 + kDone;
      if (counterRef.current) {
        const s = `${spec.name} · ${t < HERO ? "Game 1" : `${played.toLocaleString()} of ${N_GAMES.toLocaleString()} games`}`;
        if (counterRef.current.textContent !== s) counterRef.current.textContent = s;
      }
      const ph = t < HERO ? 0 : t < READOUT ? 1 : 2;
      if (t < HERO) {
        const i = Math.max(1, Math.ceil(heroUpto));
        const m = hero.margins[i];
        const score = m === 0 ? "tied" : m > 0 ? `Home +${m}` : `Away +${-m}`;
        setText(`${clock(scene.sport, hero.at[i])} · ${hero.text![i - 1]} · ${score}`);
      } else if (t < ENS_START) {
        const f = finalOf(hero);
        setText(`Final: ${f > 0 ? "Home" : "Away"} wins by ${Math.abs(f)}. But was that typical?`);
      } else if (t < READOUT) {
        setText(kDone < SLOW_N ? "Same matchup, different breaks…" : "Replaying the matchup…");
      } else {
        setText("Each line is one simulated game; the bars count how they ended.");
      }
      if (readoutRef.current) {
        const on = t >= READOUT && t < FADE;
        readoutRef.current.classList.toggle("on", on);
      }
      if (ph !== lastPhase) { lastPhase = ph; setPhase(ph); }
      if (scene.sport !== lastSport) {
        lastSport = scene.sport; setSport(scene.sport);
        if (readoutRef.current) {
          const pct = (n: number) => `${Math.round((n / N_GAMES) * 100)}%`;
          readoutRef.current.querySelector("strong")!.textContent = `Home wins ${pct(scene.homeWin)}`;
          readoutRef.current.querySelector("span")!.textContent =
            `Away ${pct(scene.awayWin)}${scene.homeWin + scene.awayWin < N_GAMES ? ` · tie ${pct(N_GAMES - scene.homeWin - scene.awayWin)}` : ""} of ${N_GAMES.toLocaleString()} replays`;
        }
      }
    };

    const frame = (now: number) => {
      const dt = Math.min(now - last, 100);
      last = now;
      if (reduced.matches) {
        t = READOUT + 500; // still frame: the finished picture
      } else if (visible && !pausedRef.current) {
        t += dt;
        if (t >= LOOP) {
          t = 0;
          loop++;
          scene = makeScene(loop % 2 ? "nba" : "nfl", 7 + loop * 101);
          clearBaked();
        }
      }
      if (visible) draw();
      raf = requestAnimationFrame(frame);
    };

    resize();
    const ro = new ResizeObserver(() => { if (wrap.clientWidth !== W) resize(); });
    ro.observe(wrap);
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; });
    io.observe(wrap);
    raf = requestAnimationFrame(frame);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); io.disconnect(); };
  }, []);

  return (
    <figure className="card replay-anim">
      <div className="replay-head">
        <span ref={counterRef} className="replay-counter">NFL · Game 1</span>
        <span ref={tickerRef} className="replay-ticker" aria-hidden="true" />
        <button type="button" className="replay-pause" onClick={() => setPaused((p) => !p)}
          aria-label={paused ? "Play animation" : "Pause animation"} title={paused ? "Play" : "Pause"}>
          {paused ? (
            <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true"><path d="M4 2.5v11l9-5.5z" fill="currentColor" /></svg>
          ) : (
            <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z" fill="currentColor" /></svg>
          )}
        </button>
      </div>
      <div ref={wrapRef} className="replay-stage">
        <canvas ref={canvasRef} role="img"
          aria-label={`Animation: one simulated ${SPORTS[sport].name} game is drawn as a line showing which team leads, play by play. It is rewound and replayed a thousand times, and the final scores stack into a histogram showing how often each team wins.`} />
        <div ref={readoutRef} className="replay-readout" aria-hidden="true"><strong /><span /></div>
      </div>
      <ol className="replay-steps">
        {STEPS.map((s, i) => (
          <li key={s.h} className={phase === i ? "on" : ""}>
            <span className="num">{i + 1}</span>
            <div><h3>{s.h}</h3><p>{s.p}</p></div>
          </li>
        ))}
      </ol>
    </figure>
  );
}
