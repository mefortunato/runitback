import { lazy, Suspense, type ReactNode } from "react";
import { Link, navigate } from "../router";
import { api } from "./lib/api";
import { links } from "./lib/links";
import type { SportInfo } from "./lib/types";
import { useData } from "./useData";
import { cx } from "./cx";
import "./predictions.css";

const Upcoming = lazy(() => import("./pages/Upcoming"));
const Game = lazy(() => import("./pages/Game"));
const Recent = lazy(() => import("./pages/Recent"));
const Rankings = lazy(() => import("./pages/Rankings"));
const Performance = lazy(() => import("./pages/Performance"));
const Players = lazy(() => import("./pages/Players"));
const TeamMapPage = lazy(() => import("./pages/Map"));
const Challenge = lazy(() => import("./pages/Challenge"));

type Route =
  | { page: "upcoming"; season?: number; date?: number }
  | { page: "recent"; season?: number; date?: number }
  | { page: "game"; gameId: string }
  | { page: "rankings" | "players" | "performance" | "map" | "challenge" | "generator" }
  | { page: "missing" };

const int = (s?: string) => (s !== undefined && /^\d+$/.test(s) ? Number(s) : undefined);

/** /<sport>/... -> the page to show. */
export function parseRoute(rest: string[]): Route {
  const [head, a, b] = rest;
  if (head === undefined) return { page: "upcoming" };
  if (int(head) !== undefined) return rest.length <= 2 && (a === undefined || int(a) !== undefined)
    ? { page: "upcoming", season: int(head), date: int(a) } : { page: "missing" };
  if (head === "recent") return { page: "recent", season: int(a), date: int(b) };
  if (head === "games" && a) return { page: "game", gameId: decodeURIComponent(a) };
  if (["rankings", "players", "performance", "map", "challenge", "generator"].includes(head) && a === undefined)
    return { page: head as "rankings" };
  return { page: "missing" };
}

/** A sport's pages: the predictions (precomputed, from the wager models) and, for a sport with a
 * simulator, the play-by-play generator (live, from the server). A sport without predictions yet
 * shows the generator only, or says the predictions aren't published yet. */
export default function SportPage({ sport, label, rest, generator }: {
  sport: string; label: string; rest: string[]; generator?: () => ReactNode;
}) {
  const { data: sports, error } = useData(() => api.sports().catch(() => [] as SportInfo[]), []);
  const route = parseRoute(rest);
  if (!sports && !error) return <p className="muted">Loading…</p>;
  const info = sports?.find((s) => s.sport === sport);
  const page = generator && (route.page === "generator" || (!info && route.page !== "missing")) ? "generator"
    : route.page === "generator" ? "missing" : route.page;

  const tabs = info ? [
    { label: "Upcoming", to: links.upcoming(sport), active: page === "upcoming" },
    { label: "Recent", to: links.recent(sport), active: page === "recent" },
    { label: "Rankings", to: links.rankings(sport), active: page === "rankings" },
    ...(info.has_players ? [{ label: "Players", to: links.players(sport), active: page === "players" }] : []),
    ...(info.has_team_map ? [{ label: "Team map", to: links.teamMap(sport), active: page === "map" }] : []),
    ...(info.challenge_games ? [{ label: "Performance", to: links.performance(sport), active: page === "performance" }] : []),
    ...(info.challenge_games && (info.has_spread || info.markets?.length)
      ? [{ label: "Beat the model", to: links.challenge(sport), active: page === "challenge" }] : []),
  ] : [];
  const gen = generator && { label: "Play-by-play generator", to: links.generator(sport), active: page === "generator" };
  const current = [...tabs, ...(gen ? [gen] : [])].find((t) => t.active);

  return (
    <div className="predictions">
      <nav className="sport-nav" aria-label={`${label} pages`}>
        {tabs.map((t) => (
          <Link key={t.label} to={t.to} className={cx("page", t.active && "active")} aria-current={t.active ? "page" : undefined}>
            {t.label}
          </Link>
        ))}
        {tabs.length > 0 && (
          <select aria-label="Page" value={current && current !== gen ? current.to : ""} onChange={(e) => navigate(e.target.value)}>
            {(!current || current === gen) && <option value="" disabled>Predictions</option>}
            {tabs.map((t) => <option key={t.label} value={t.to}>{t.label}</option>)}
          </select>
        )}
        {gen && tabs.length > 0 && <span className="sep" aria-hidden="true" />}
        {gen && <Link to={gen.to} className={cx(gen.active && "active")} aria-current={gen.active ? "page" : undefined}>{gen.label}</Link>}
      </nav>
      <Suspense fallback={<p className="muted">Loading…</p>}>
        {page === "generator" && generator ? generator()
          : !info && page !== "missing" ? (
            <section className="card">
              <h1>{label} predictions</h1>
              <p>The {label} predictions aren't published yet. Try the <Link to="/">home page</Link>.</p>
            </section>
          )
          : !info || page === "missing" ? (
            <section className="card">
              <h1>Page not found</h1>
              <p>
                Try the {info && <Link to={links.upcoming(sport)}>{label} predictions</Link>}
                {info && gen && " or the "}
                {gen && <Link to={gen.to}>generator</Link>}
                {!info && !gen && <Link to="/">home page</Link>}.
              </p>
            </section>
          )
          : route.page === "upcoming" ? <Upcoming info={info} season={route.season} date={route.date} />
          : route.page === "recent" ? <Recent info={info} season={route.season} date={route.date} />
          : route.page === "game" ? <Game info={info} gameId={route.gameId} />
          : route.page === "rankings" ? <Rankings info={info} />
          : route.page === "players" ? <Players info={info} />
          : route.page === "performance" ? <Performance info={info} />
          : route.page === "map" ? <TeamMapPage info={info} />
          : <Challenge info={info} />}
      </Suspense>
    </div>
  );
}
