// Shared fetch helper: JSON in, JSON out, FastAPI error details surfaced as Error messages.
export async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, {
    ...init, headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

export const errorText = (e: unknown) => String((e as Error)?.message || e);
