import { useEffect } from "react";
import Home from "./Home";
import NbaApp from "./nba/NbaApp";
import NflApp from "./nfl/NflApp";
import { Link, usePath } from "./router";

const PAGES: Record<string, { title: string; render: () => React.ReactNode }> = {
  "/": { title: "Run It Back", render: () => <Home /> },
  "/nfl": { title: "NFL · Run It Back", render: () => <NflApp /> },
  "/nba": { title: "NBA · Run It Back", render: () => <NbaApp /> },
};

export default function App() {
  const raw = usePath();
  const path = raw.length > 1 ? raw.replace(/\/+$/, "") : raw;
  const page = PAGES[path];
  useEffect(() => { document.title = page?.title ?? "Not found · Run It Back"; }, [page]);
  const tab = (to: string, label: string) => (
    <Link to={to} className={path === to ? "selected" : ""} aria-current={path === to ? "page" : undefined}>{label}</Link>
  );
  return (
    <div className="app">
      <header className="site-header">
        <Link to="/" className="brand">Run It Back</Link>
        <nav aria-label="Sports">{tab("/", "Home")}{tab("/nfl", "NFL")}{tab("/nba", "NBA")}</nav>
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
