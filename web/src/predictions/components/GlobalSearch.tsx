import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { navigate } from "../../router";
import { navSports, sportById } from "../../sports";
import { api, type NextGame } from "../lib/api";
import { links, sportPages } from "../lib/links";
import { globalSearch, type PageResult, type TeamResult } from "../lib/search";
import type { SportInfo } from "../lib/types";
import { setPendingTeam, useTeamTarget } from "../headerSearch";
import { useData } from "../useData";
import { cx } from "../cx";
import TeamBadge from "./TeamBadge";
import s from "./GlobalSearch.module.css";

interface TeamIndex {
  teams: TeamResult[];
  /** Each team's next game date, by `${sport}/${team}`, for those with one. */
  next: Map<string, NextGame>;
}

type Item = { kind: "page"; page: PageResult } | { kind: "team"; team: TeamResult };

const ID = "site-search";

/** The header's search: every sport's pages and teams, one pick away. Picking a team shows it on
 * the current page when that page can (Upcoming/Recent filter to its games, Rankings/Team map
 * highlight it); otherwise it goes to the team's next game date, or its sport's rankings. */
export default function GlobalSearch({ current }: { current?: string }) {
  const { data: sports } = useData(() => api.sports().catch(() => [] as SportInfo[]), []);
  const input = useRef<HTMLInputElement>(null);
  const loading = useRef<Promise<void> | null>(null);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [index, setIndex] = useState<TeamIndex | null>(null);
  const target = useTeamTarget();

  // every sport in the nav: its predictions pages, and its generator where it has one
  const pages = useMemo(() => navSports(sports?.map((i) => i.sport)).flatMap((def) => {
    const info = sports?.find((i) => i.sport === def.id);
    const page = (label: string, href: string): PageResult => ({ sport: def.id, sportLabel: def.label, label, href });
    return [
      ...(info ? sportPages(info).map((p) => page(p.label, p.to)) : []),
      ...(def.generator ? [page("Play-by-play generator", links.generator(def.id))] : []),
    ];
  }), [sports]);
  const results = useMemo(() => globalSearch(query, pages, index?.teams ?? [], current), [query, pages, index, current]);
  const items: Item[] = [
    ...results.pages.map((page) => ({ kind: "page" as const, page })),
    ...results.teams.map((team) => ({ kind: "team" as const, team })),
  ];
  const typed = query.trim() !== "";
  const showList = open && typed;

  // the teams load on first use: every sport's rankings and upcoming files
  function loadTeams() {
    if (!sports || loading.current) return;
    loading.current = Promise.all(sports.map(async (info) => {
      const { teams, next } = await api.searchTeams(info.sport)
        .catch(() => ({ teams: [], next: new Map<string, NextGame>() }));
      return { info, teams, next };
    })).then((all) => setIndex({
      teams: all.flatMap(({ info, teams }) => teams.map((team) => ({
        sport: info.sport, sportLabel: sportById(info.sport)?.label ?? info.label, team }))),
      next: new Map(all.flatMap(({ info, next }) => [...next].map(([team, when]) => [`${info.sport}/${team}`, when] as const))),
    }));
  }
  // sports.json may still be loading when the box gets focus
  useEffect(() => { if (open) loadTeams(); });

  /** Where picking `t` takes the reader, and what it does there. */
  function teamAction(t: TeamResult): { verb: string; run: () => void } {
    if (target?.sport === t.sport) return { verb: target.verb, run: () => target.pick(t.team) };
    const next = index?.next.get(`${t.sport}/${t.team.team}`);
    const href = next ? links.upcoming(t.sport, next.season, next.date) : links.rankings(t.sport);
    return {
      verb: next ? "Upcoming games" : "Rankings",
      run: () => { setPendingTeam(t.sport, t.team); navigate(href); },
    };
  }

  function choose(item: Item) {
    setQuery("");
    setOpen(false);
    input.current?.blur();
    if (item.kind === "page") navigate(item.page.href);
    else teamAction(item.team).run();
  }

  function keydown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Escape") {
      if (showList) setOpen(false);
      else { setQuery(""); input.current?.blur(); }
      event.preventDefault();
      return;
    }
    if (!items.length) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      setOpen(true);
      const step = event.key === "ArrowDown" ? 1 : -1;
      setActive((a) => (Math.min(a, items.length - 1) + step + items.length) % items.length);
    } else if (event.key === "Enter" && showList) {
      event.preventDefault();
      choose(items[Math.min(active, items.length - 1)]);
    }
  }

  // "/" anywhere outside a form field jumps to the search
  useEffect(() => {
    const shortcut = (event: globalThis.KeyboardEvent) => {
      if (event.key !== "/" || event.metaKey || event.ctrlKey || event.altKey) return;
      const el = event.target as HTMLElement | null;
      if (el?.closest?.("input, textarea, select, [contenteditable='true']")) return;
      event.preventDefault();
      input.current?.focus();
    };
    window.addEventListener("keydown", shortcut);
    return () => window.removeEventListener("keydown", shortcut);
  }, []);

  const shown = Math.min(active, Math.max(items.length - 1, 0));
  return (
    <div className={s.search}>
      <label className="sr-only" htmlFor={ID}>Search teams and pages</label>
      <svg className={s.icon} viewBox="0 0 16 16" aria-hidden="true">
        <circle cx="7" cy="7" r="4.5" fill="none" stroke="currentColor" strokeWidth="1.6" />
        <path d="M10.5 10.5 14 14" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
      <input
        ref={input}
        id={ID}
        className={s.input}
        type="search"
        autoComplete="off"
        spellCheck={false}
        placeholder="Search teams and pages"
        value={query}
        role="combobox"
        aria-expanded={showList}
        aria-controls={`${ID}-list`}
        aria-autocomplete="list"
        aria-activedescendant={showList && items.length ? `${ID}-${shown}` : undefined}
        onChange={(e) => { setQuery(e.target.value); setActive(0); setOpen(true); }}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={keydown}
      />
      {!typed && <kbd className={s.hint} aria-hidden="true">/</kbd>}
      <ul id={`${ID}-list`} className={s.list} role="listbox" aria-label="Search results" hidden={!showList}>
        {items.map((item, i) => {
          const key = item.kind === "page" ? `p ${item.page.href}` : `t ${item.team.sport} ${item.team.team.team}`;
          const group = i === 0 && item.kind === "page" ? "Pages"
            : item.kind === "team" && (i === 0 || items[i - 1].kind === "page") ? "Teams" : null;
          return [
            group && <li key={`g ${group}`} className={s.group} role="presentation">{group}</li>,
            // mousedown, not click: the input's blur would close the list first
            <li key={key} id={`${ID}-${i}`} role="option" aria-selected={i === shown} className={cx(s.option, i === shown && s.active)}
              onMouseDown={(e) => { e.preventDefault(); choose(item); }} onMouseEnter={() => setActive(i)}>
              {item.kind === "page" ? (
                <>
                  <span className={s.sport}>{item.page.sportLabel}</span>
                  <span className={s.name}>{item.page.label}</span>
                </>
              ) : (
                <>
                  <TeamBadge team={item.team.team} label="none" size={20} />
                  <span className={s.name}>{item.team.team.name}</span>
                  <span className={s.sport}>{item.team.sportLabel}</span>
                  <span className={cx(s.verb, "muted small")}>{teamAction(item.team).verb}</span>
                </>
              )}
            </li>,
          ];
        })}
        {!index ? <li className={cx(s.note, "muted small")} role="presentation">Loading teams…</li>
          : !items.length ? <li className={cx(s.note, "muted small")} role="presentation">No team or page matches “{query.trim()}”.</li>
          : null}
      </ul>
    </div>
  );
}
