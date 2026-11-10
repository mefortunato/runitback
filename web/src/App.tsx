import { useEffect } from "react";
import Home from "./Home";
import NbaApp from "./nba/NbaApp";
import NflApp from "./nfl/NflApp";
import { Link, usePath } from "./router";
import { ThemeToggle } from "./theme";
import SportPage from "./predictions/SportPage";
import { api } from "./predictions/lib/api";
import { useData } from "./predictions/useData";
import { navSports, sportById } from "./sports";

// each sport: /<sport>/... = predictions pages, /<sport>/generator = the play-by-play generator
// (only the sports with a simulator: see ./sports.ts)
const GENERATORS: Record<string, () => React.ReactNode> = {
  nfl: () => <NflApp />,
  nba: () => <NbaApp />,
};

function route(path: string): { title: string; render: () => React.ReactNode } | undefined {
  if (path === "/") return { title: "Run It Back", render: () => <Home /> };
  const [sport, ...rest] = path.split("/").filter(Boolean);
  const s = sportById(sport);
  if (!s) return undefined;
  const page = rest[0] === "generator" ? "Generator" : rest.length ? rest[0][0].toUpperCase() + rest[0].slice(1) : "Predictions";
  return {
    title: `${s.label} ${/^\d+$/.test(page) ? "Predictions" : page} · Run It Back`,
    render: () => <SportPage key={sport} sport={sport} label={s.label} rest={rest} generator={GENERATORS[sport]} />,
  };
}

export default function App() {
  const raw = usePath();
  const path = raw.length > 1 ? raw.replace(/\/+$/, "") : raw;
  const page = route(path);
  // the nav lists a sport once it has predictions (or a generator); sports.json is shared with
  // the predictions pages (loaded once)
  const { data: exported } = useData(() => api.sports().then((s) => s.map((i) => i.sport)).catch(() => []), []);
  const current = sportById(path.split("/")[1]);
  const nav = navSports(exported);
  if (current && !nav.includes(current)) nav.push(current);  // a direct link still shows its tab
  useEffect(() => { document.title = page?.title ?? "Not found · Run It Back"; }, [page?.title]);
  const tab = (to: string, label: string) => {
    const on = to === "/" ? path === "/" : path === to || path.startsWith(to + "/");
    return <Link key={to} to={to} className={on ? "selected" : ""} aria-current={on ? "page" : undefined}>{label}</Link>;
  };
  return (
    <div className="app">
      <header className="site-header">
        <Link to="/" className="brand">Run It Back</Link>
        <div className="header-actions">
          <nav aria-label="Sports">{tab("/", "Home")}{nav.map((s) => tab(`/${s.id}`, s.label))}</nav>
          <ThemeToggle />
        </div>
      </header>
      <main>
        {page ? page.render() : (
          <section className="card not-found">
            <h1>Page not found</h1>
            <p>
              Try the <Link to="/">home page</Link>
              {nav.map((s, i) => <span key={s.id}>{i === nav.length - 1 ? " or the " : ", the "}<Link to={`/${s.id}`}>{s.label}</Link></span>)}.
            </p>
          </section>
        )}
      </main>
    </div>
  );
}
