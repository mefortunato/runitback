import { useEffect, useState } from "react";

/** Load async page data; re-runs when `deps` change. */
export function useData<T>(load: () => Promise<T>, deps: unknown[]): { data?: T; error?: string } {
  const [state, setState] = useState<{ data?: T; error?: string }>({});
  useEffect(() => {
    let live = true;
    setState({});
    load().then((data) => live && setState({ data }), (e) => live && setState({ error: String(e?.message ?? e) }));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

/** A remembered per-viewer choice (wrapped: storage can be blocked). */
export function remembered<T extends string>(key: string, fallback: T): T {
  try { return (localStorage.getItem(key) as T | null) ?? fallback; } catch { return fallback; }
}
export function remember(key: string, value: string) {
  try { localStorage.setItem(key, value); } catch { /* only a convenience */ }
}
