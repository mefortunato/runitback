import type { State } from "./api";

// Field drawn in yards: x=0..120 incl. end zones (home defends the left goal), y=0..53.3.
export default function Field({ s, prev }: { s: State; prev?: State }) {
  const ballX = (st: State) => {
    const home = st.pos === st.home;
    if (st.phase === "kickoff") return 10 + (home ? 35 : 65);
    if (st.phase === "pat") return 10 + (home ? 85 : 15);
    return 10 + (home ? 100 - st.yl : st.yl);
  };
  const x = ballX(s);
  const homeDir = s.pos === s.home ? 1 : -1;
  const fd = s.phase === "scrimmage" && s.ytg < s.yl ? x + homeDir * s.ytg : null;
  const px = prev && !prev.over ? ballX(prev) : null;
  const lines = Array.from({ length: 21 }, (_, i) => 10 + i * 5);
  return (
    <svg className="field" viewBox="0 0 120 53.3" role="img"
      aria-label={`Ball at ${s.yl} yards from the ${s.pos} offense's goal`}>
      <rect x="0" y="0" width="120" height="53.3" className="turf" />
      <rect x="0" y="0" width="10" height="53.3" className="ez" />
      <rect x="110" y="0" width="10" height="53.3" className="ez" />
      <text x="5" y="26.6" className="ez-label" transform="rotate(-90 5 26.6)">{s.home}</text>
      <text x="115" y="26.6" className="ez-label" transform="rotate(90 115 26.6)">{s.away}</text>
      {lines.map((lx) => <line key={lx} x1={lx} x2={lx} y1="0" y2="53.3" className="yardline" />)}
      {[10, 20, 30, 40, 50, 40, 30, 20, 10].map((n, i) => (
        <text key={i} x={20 + i * 10} y="8" className="num">{n}</text>
      ))}
      {px !== null && px !== x && (
        <line x1={px} x2={x} y1="26.6" y2="26.6" className="trail" />
      )}
      {s.phase === "scrimmage" && <line x1={x} x2={x} y1="0" y2="53.3" className="los" />}
      {fd !== null && <line x1={fd} x2={fd} y1="0" y2="53.3" className="fd" />}
      <ellipse cx={x} cy="26.6" rx="1.6" ry="1" className="ball" style={{ transition: "cx 0.35s ease" }} />
      <text x={x} y="33" className="dir">{homeDir > 0 ? "▶" : "◀"}</text>
    </svg>
  );
}
