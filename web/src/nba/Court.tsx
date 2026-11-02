import type { Play, State } from "./api";
import { isMade, isShot } from "./api";

// Full court in feet: x = 0..94 (home attacks the right basket all game, away the left, as on a
// shot chart), y = 0..50. Shot spots are illustrative: the model predicts a shot's value and
// result, not where on the floor it was taken, so each shot gets a stable pseudo-random spot.
const RIM_X = 5.25;

function rand(seed: number) {
  let t = (seed + 0x6d2b79f5) | 0;
  return () => {
    t = (t + 0x6d2b79f5) | 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

// A spot relative to the left basket; mirrored for the right one.
function shotSpot(n: number, three: boolean): [number, number] {
  const r = rand(n * 7919);
  if (three) {
    if (r() < 0.27) {  // corner three
      const side = r() < 0.5 ? 1 : -1;
      return [1 + r() * 11, 25 + side * (22.4 + r() * 1.8)];
    }
    const a = (r() * 2 - 1) * 1.15;          // radians from straight on
    const d = 24 + r() * 3.5;
    return [RIM_X + Math.cos(a) * d, 25 + Math.sin(a) * d];
  }
  const u = r();
  const d = u < 0.55 ? 1 + r() * 4.5 : u < 0.8 ? 5 + r() * 9 : 14 + r() * 7.5;  // rim, paint, midrange
  const a = (r() * 2 - 1) * 1.45;
  return [Math.max(1, RIM_X + Math.cos(a) * d), Math.min(49, Math.max(1, 25 + Math.sin(a) * d))];
}

export default function Court({ s, plays }: { s: State; plays: Play[] }) {
  const toRight = (team: string) => team === s.home;
  const place = (p: Play): [number, number] => {
    const [x, y] = p.outcome.play_type === "free_throw" ? [19, 25] : shotSpot(p.n, p.outcome.play_type === "shot3");
    return toRight(p.before.pos) ? [94 - x, 50 - y] : [x, y];
  };
  const shots = plays.filter((p) => isShot(p.outcome.play_type));
  const last = plays[plays.length - 1];
  const lastShot = last && (isShot(last.outcome.play_type) || last.outcome.play_type === "free_throw") ? last : null;
  const attackRight = toRight(s.pos);
  const ballX = s.phase === "tip" ? 47 : s.phase === "free_throw" ? (attackRight ? 75 : 19) : attackRight ? 62 : 32;
  const halfCourt = (flip: boolean) => {
    const t = (x: number) => (flip ? 94 - x : x);
    const rim = t(RIM_X);
    return (
      <g className="court-lines">
        <rect x={Math.min(t(0), t(19))} y={17} width={19} height={16} className="paint" />
        <circle cx={t(19)} cy={25} r={6} />
        <path d={`M${t(0)},3 L${t(14)},3 A23.75,23.75 0 0 ${flip ? 0 : 1} ${t(14)},47 L${t(0)},47`} />
        <line x1={t(4)} x2={t(4)} y1={22} y2={28} className="backboard" />
        <circle cx={rim} cy={25} r={0.75} className="rim" />
        <path d={`M${t(RIM_X)},21 A4,4 0 0 ${flip ? 0 : 1} ${t(RIM_X)},29`} />
      </g>
    );
  };
  return (
    <figure className="court-wrap">
      <svg className="court" viewBox="-1 -1 96 52" role="img"
        aria-label={`Court: ${s.away} shoot at the left basket, ${s.home} at the right. ${shots.length} shots so far.`}>
        <rect x="0" y="0" width="94" height="50" className="floor" />
        <g className="court-lines">
          <rect x="0" y="0" width="94" height="50" />
          <line x1="47" x2="47" y1="0" y2="50" />
          <circle cx="47" cy="25" r="6" />
        </g>
        {halfCourt(false)}
        {halfCourt(true)}
        <text x="23" y="48.4" className="court-label" textAnchor="middle">{s.away} shoot ◀</text>
        <text x="71" y="48.4" className="court-label" textAnchor="middle">▶ {s.home} shoot</text>
        {shots.map((p) => {
          const [x, y] = place(p);
          const made = isMade(p.outcome.result);
          const cls = `shot ${p.before.pos === s.home ? "home" : "away"} ${made ? "made" : "missed"}`;
          return made ? <circle key={p.n} cx={x} cy={y} r={0.9} className={cls} />
            : <path key={p.n} d={`M${x - 0.7},${y - 0.7}L${x + 0.7},${y + 0.7}M${x + 0.7},${y - 0.7}L${x - 0.7},${y + 0.7}`} className={cls} />;
        })}
        {lastShot && (() => {
          const [x, y] = place(lastShot);
          return <circle cx={x} cy={y} r={2.2} className={`last-shot ${isMade(lastShot.outcome.result) ? "made" : "missed"}`} />;
        })()}
        {!s.over && (
          <g style={{ transform: `translateX(${ballX}px)`, transition: "transform .35s ease" }}>
            <circle cx={0} cy={25} r={1.3} className="ball" />
            {s.phase !== "tip" && <text x={0} y={22} className="dir" textAnchor="middle">{attackRight ? "▶" : "◀"}</text>}
          </g>
        )}
      </svg>
      <figcaption className="muted court-note">
        <span><span className="key made" /> made</span> <span><span className="key missed">×</span> missed</span>
        <span>Shot spots are illustrative; the model predicts 2 vs 3 and the result, not the location.</span>
      </figcaption>
    </figure>
  );
}
