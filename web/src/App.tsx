import { useEffect } from "react";
import Home from "./Home";
import NbaApp from "./nba/NbaApp";
import NflApp from "./nfl/NflApp";
import { Link, usePath } from "./router";
import { ThemeToggle } from "./theme";
import SportPage from "./predictions/SportPage";

// each sport: /<sport>/... = predictions pages, /<sport>/generator = the play-by-play generator
const SPORTS: Record<string, { label: string; generator: () => React.ReactNode }> = {
  nfl: { label: "NFL", generator: () => <NflApp /> },
  nba: { label: "NBA", generator: () => <NbaApp /> },
};

function route(path: string): { title: string; render: () => React.ReactNode } | undefined {
  if (path === "/") return { title: "Run It Back", render: () => <Home /> };
  const [sport, ...rest] = path.split("/").filter(Boolean);
  const s = SPORTS[sport];
  if (!s) return undefined;
  const page = rest[0] === "generator" ? "Generator" : rest.length ? rest[0][0].toUpperCase() + rest[0].slice(1) : "Predictions";
  return {
    title: `${s.label} ${/^\d+$/.test(page) ? "Predictions" : page} · Run It Back`,
    render: () => <SportPage key={sport} sport={sport} label={s.label} rest={rest} generator={s.generator} />,
  };
}

export default function App() {
  const raw = usePath();
  const path = raw.length > 1 ? raw.replace(/\/+$/, "") : raw;
  const page = route(path);
  useEffect(() => { document.title = page?.title ?? "Not found · Run It Back"; }, [page?.title]);
  const tab = (to: string, label: string) => {
    const on = to === "/" ? path === "/" : path === to || path.startsWith(to + "/");
    return <Link to={to} className={on ? "selected" : ""} aria-current={on ? "page" : undefined}>{label}</Link>;
  };
  return (
    <div className="app">
      <header className="site-header">
        <Link to="/" className="brand">Run It Back</Link>
        <div className="header-actions">
          <nav aria-label="Sports">{tab("/", "Home")}{tab("/nfl", "NFL")}{tab("/nba", "NBA")}</nav>
          <ThemeToggle />
        </div>
      </header>
      <main>
        {page ? page.render() : (
          <section className="card not-found">
            <h1>Page not found</h1>
            <p>Try the <Link to="/">home page</Link>, the <Link to="/nfl">NFL</Link> or the <Link to="/nba">NBA</Link>.</p>
          </section>
        )}
      </main>
    </div>
  );
}
