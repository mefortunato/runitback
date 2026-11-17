import { useEffect, useRef, useSyncExternalStore } from "react";
import type { Team } from "./lib/types";

// The header search and the page under it. A page that can show a team registers with
// `useTeamPick`; a team picked for another sport or page is left pending for the page the search
// then goes to.

/** What the page on show does with a team picked in the header search. */
export interface TeamTarget {
  sport: string;
  /** What picking does, shown beside the team ("Highlight"). */
  verb: string;
  pick: (team: Team) => void;
}

let target: TeamTarget | null = null;
let pending: { sport: string; team: Team } | null = null;
const listeners = new Set<() => void>();

function setTarget(next: TeamTarget | null) {
  target = next;
  listeners.forEach((l) => l());
}

const subscribe = (l: () => void) => { listeners.add(l); return () => { listeners.delete(l); }; };

/** The page that can show a picked team right where the reader is, if any. */
export function useTeamTarget(): TeamTarget | null {
  return useSyncExternalStore(subscribe, () => target);
}

/** Hand `team` to the next page of `sport` that registers (the one the search goes to). */
export function setPendingTeam(sport: string, team: Team) {
  pending = { sport, team };
}

/** Let the header search hand this page its sport's teams while the calling component is mounted,
 * including one picked on the way here. Call it after the state `pick` changes is set up (and
 * after any effect that resets it), since a pending team is picked in an effect. */
export function useTeamPick(sport: string, verb: string, pick: (team: Team) => void) {
  const latest = useRef(pick);
  useEffect(() => { latest.current = pick; });
  useEffect(() => {
    const mine: TeamTarget = { sport, verb, pick: (team) => latest.current(team) };
    setTarget(mine);
    const waiting = pending;
    if (waiting?.sport === sport) {
      pending = null;
      mine.pick(waiting.team);
    }
    return () => { if (target === mine) setTarget(null); };
  }, [sport, verb]);
}
