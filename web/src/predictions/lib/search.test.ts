import { describe, expect, it } from "vitest";
import { globalSearch, normalize, teamScore, type PageResult, type SearchResults, type TeamResult } from "./search";
import type { Team } from "./types";

const lions: Team = { team: "DET", name: "Detroit Lions", nickname: "Lions" };
const saoPaulo: Team = { team: "SAO", name: "São Paulo FC", abbr: "SPFC" };

describe("team search", () => {
  it("normalizes case, accents and punctuation", () => {
    expect(normalize("  São-Paulo  F.C. ")).toBe("sao paulo f c");
  });

  it("ranks exact names above prefixes, word prefixes and substrings", () => {
    expect(teamScore(lions, "det")).toBe(4);
    expect(teamScore(lions, "detr")).toBe(3);
    expect(teamScore(lions, "lio det")).toBe(2);
    expect(teamScore(lions, "troit")).toBe(1);
    expect(teamScore(lions, "bears")).toBeNull();
    expect(teamScore(saoPaulo, "sao paulo")).toBe(3);
    expect(teamScore(saoPaulo, "spfc")).toBe(4);
  });
});

describe("global search", () => {
  const page = (sport: string, label: string): PageResult => ({ sport, sportLabel: sport.toUpperCase(), label, href: `/${sport}/${label}` });
  const pages = ["nfl", "nba"].flatMap((s) => ["Upcoming", "Rankings", "Team map"].map((l) => page(s, l)));
  const team = (sport: string, t: Team): TeamResult => ({ sport, sportLabel: sport.toUpperCase(), team: t });
  const knicks: Team = { team: "NY", name: "New York Knicks", nickname: "Knicks" };
  const jets: Team = { team: "NYJ", name: "New York Jets", nickname: "Jets" };
  const teams = [team("nfl", lions), team("nfl", jets), team("nba", knicks)];
  const labels = (r: SearchResults) => [...r.pages.map((p) => `${p.sport} ${p.label}`), ...r.teams.map((t) => t.team.name)];

  it("finds nothing for an empty query", () => {
    expect(globalSearch("  ", pages, teams)).toEqual({ pages: [], teams: [] });
  });

  it("lists a sport's pages for its name", () => {
    expect(labels(globalSearch("nba", pages, teams))).toEqual(["nba Upcoming", "nba Rankings", "nba Team map"]);
  });

  it("finds pages by name, the current sport first", () => {
    expect(labels(globalSearch("rank", pages, teams, "nba"))).toEqual(["nba Rankings", "nfl Rankings"]);
    expect(labels(globalSearch("nfl map", pages, teams))).toEqual(["nfl Team map"]);
  });

  it("finds teams in every sport, the current sport first", () => {
    expect(labels(globalSearch("new york", pages, teams))).toEqual(["New York Jets", "New York Knicks"]);
    expect(labels(globalSearch("new york", pages, teams, "nba"))).toEqual(["New York Knicks", "New York Jets"]);
    expect(labels(globalSearch("nba new york", pages, teams))).toEqual(["New York Knicks"]);
  });
});
