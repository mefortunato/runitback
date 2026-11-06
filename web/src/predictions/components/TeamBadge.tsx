import { useState } from "react";
import type { Team } from "../lib/types";
import { cx } from "../cx";
import s from "./TeamBadge.module.css";

// chip text: the second team color where it reads on the first (contrast 3:1 or more), else
// white or black, whichever reads better
function luminance(hex: string): number | null {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return null;
  const [r, g, b] = [0, 2, 4].map((i) => {
    const c = parseInt(m[1].slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
const contrast = (a: number, b: number) => (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);

function chipText(team: Team): string {
  const bg = team.color ? luminance(team.color) : null;
  if (bg === null) return "#fff";
  const second = team.color2 ? luminance(team.color2) : null;
  if (second !== null && contrast(bg, second) >= 3) return team.color2!;
  return contrast(bg, 1) >= contrast(bg, 0) ? "#fff" : "#000";
}

export default function TeamBadge({ team, size = 28, label = "abbr" }: {
  team: Team; size?: number; label?: "abbr" | "name" | "nickname" | "none";
}) {
  // the team-colored chip shows until the logo has loaded, and stays if it fails (keyed by URL)
  const [loadedLogo, setLoaded] = useState<string>();
  const [failedLogo, setFailed] = useState<string>();
  const loaded = !!team.logo && loadedLogo === team.logo;
  const failed = !!team.logo && failedLogo === team.logo;
  const text = label === "name" ? team.name : label === "nickname" ? (team.nickname ?? team.name) : (team.abbr ?? team.team);
  return (
    <span className={cx(s.root, s.badge, "team-badge")} title={team.name}>
      <span className={s.mark} style={{ width: size, height: size }}>
        {!loaded && (
          <span className={s.chip} style={{ background: team.color ?? "var(--muted)", color: chipText(team),
            fontSize: Math.max(9, size * 0.32) }}>{team.abbr ?? team.team}</span>
        )}
        {team.logo && !failed && (
          <img src={team.logo} alt="" width={size} height={size} loading="lazy" className={cx(loaded && s.loaded)}
            onLoad={() => setLoaded(team.logo!)} onError={() => setFailed(team.logo!)} />
        )}
      </span>
      {label !== "none" && <span className={s.label}>{text}</span>}
    </span>
  );
}
