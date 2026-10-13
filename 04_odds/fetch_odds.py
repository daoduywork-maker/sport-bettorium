"""
fetch_odds.py  -  download historical LoL series odds from OddsPapi (run in WSL)

Needs only the Python standard library.  API key: oddspapi_key.txt in this folder.

Step 1 (about 1 min):  python fetch_odds.py --explore
    prints your quota, the LoL sport id, our 6 leagues' tournament ids, the LoL markets,
    and the raw odds of ONE finished match  ->  saved in odds_raw/explore_*.json
    Send me the printed output so I can check the market ids before the full download.

Step 2 (about 35 min):  python fetch_odds.py --market <id>
    downloads every finished match since --start (default 2026-06-01) and writes
    odds_series.csv  (fixture, start time, teams, closing odds of each bookmaker)
    Raw responses are cached in odds_raw/, so you can stop and rerun to resume.
"""
import argparse
import csv
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

BASE = "https://api.oddspapi.io/v4"
LEAGUE_WORDS = ["LCK", "LPL", "LEC", "LCS", "CBLOL", "LCP"]
# picked from the --explore list: LCS, LEC, LCK, LPL (Tencent + TJ Sports), CBLOL, LCP
TOURNAMENT_IDS = {2450, 2452, 2454, 15490, 39985, 26698, 45589}
BOOKMAKERS = "pinnacle,bet365,1xbet"     # max 3 per request
COOLDOWN = {"historical-odds": 5.1, "fixtures": 1.1, "default": 1.1}
RAW = "odds_raw"


def key():
    with open("oddspapi_key.txt") as fh:
        return fh.read().strip()


def get(endpoint, **params):
    params["apiKey"] = key()
    url = f"{BASE}/{endpoint}?{urllib.parse.urlencode(params)}"
    for attempt in range(8):
        try:
            # Cloudflare blocks Python's default User-Agent (error 1010), so send a normal one
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) lol-odds-research/1.0",
                "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode())
            time.sleep(COOLDOWN.get(endpoint, COOLDOWN["default"]))
            return data
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            if e.code == 429 and "REQUEST_LIMIT" in body:
                raise SystemExit(f"Quota used up: {body}")
            if e.code == 404 and "NOT_FOUND" in body:
                time.sleep(COOLDOWN.get(endpoint, COOLDOWN["default"]))
                return []          # nothing for this filter (e.g. an old, inactive tournament)
            if e.code in (401, 403):
                raise SystemExit(f"HTTP {e.code} on {endpoint}: {body}  (no retry)")
            wait = 10 * (attempt + 1)
            print(f"  HTTP {e.code} on {endpoint}: {body} -> retry in {wait}s")
            time.sleep(wait)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            # dropped connection (unstable network / VPN): wait and try again
            wait = 10 * (attempt + 1)
            print(f"  network error on {endpoint}: {e} -> retry in {wait}s")
            time.sleep(wait)
    raise SystemExit(f"Giving up on {endpoint} {params}")


def save(name, data):
    os.makedirs(RAW, exist_ok=True)
    with open(os.path.join(RAW, name), "w") as fh:
        json.dump(data, fh, indent=1)


def lol_sport_id():
    sports = get("sports")
    save("explore_sports.json", sports)
    hits = [s for s in sports if "legends" in json.dumps(s).lower()]
    if not hits:
        raise SystemExit("No League of Legends sport found; see odds_raw/explore_sports.json")
    return hits[0]["sportId"], hits[0]


def our_tournaments(sport_id):
    ts = get("tournaments", sportId=sport_id)
    save("explore_tournaments.json", ts)
    keep = []
    for t in ts:
        name = f"{t.get('tournamentName', '')} {t.get('categoryName', '')}".upper()
        if t.get("tournamentId") in TOURNAMENT_IDS:
            keep.append(t)
    return keep


def fixtures(tournament_id, start):
    """Finished fixtures of one tournament since `start`."""
    data = get("fixtures", tournamentId=tournament_id, statusId=2,
               **{"from": f"{start}T00:00:00Z"})
    return data if isinstance(data, list) else data.get("data", [])


def closing_prices(hist, market, start_time):
    """Last price at or before the start time, per bookmaker and outcome."""
    out = {}
    for bk, bdata in (hist.get("bookmakers") or {}).items():
        mk = (bdata.get("markets") or {}).get(str(market))
        if not mk:
            continue
        for oid, odata in mk.get("outcomes", {}).items():
            entries = []
            for plist in (odata.get("players") or {}).values():
                entries += plist if isinstance(plist, list) else [plist]
            entries = [e for e in entries if e.get("price") and e.get("createdAt", "") <= start_time]
            if entries:
                last = max(entries, key=lambda e: e["createdAt"])
                out[(bk, oid)] = last["price"]
    return out


def explore(args):
    print("connecting...", flush=True)
    acc = get("account")
    # print only quota-related fields (the account response may contain the key itself)
    safe = {k: v for k, v in (acc if isinstance(acc, dict) else {}).items()
            if any(w in k.lower() for w in ("limit", "used", "request", "plan", "remaining"))}
    print("ACCOUNT (quota):", safe)
    sid, sport = lol_sport_id()
    print("\nLoL SPORT:", sport)
    ts = our_tournaments(sid)
    print(f"\nOUR TOURNAMENTS ({len(ts)}):")
    for t in ts:
        print(f"  {t['tournamentId']:>8}  {t.get('categoryName', '')} | {t.get('tournamentName', '')}")
    markets = [m for m in get("markets") if m.get("sportId") == sid]
    save("explore_markets.json", markets)
    print(f"\nLoL MARKETS ({len(markets)}), first 25:")
    for m in markets[:25]:
        outs = ", ".join(f"{o['outcomeId']}={o['outcomeName']}" for o in m.get("outcomes", []))
        print(f"  {m['marketId']:>6}  {m.get('marketName')}  [{outs}]")
    for t in ts:
        fx = fixtures(t["tournamentId"], args.start)
        if fx:
            f0 = fx[0]
            print(f"\nSAMPLE FIXTURE: {json.dumps(f0)[:500]}")
            hist = get("historical-odds", fixtureId=f0["fixtureId"], bookmakers=BOOKMAKERS)
            save("explore_historical.json", hist)
            for bk, b in (hist.get("bookmakers") or {}).items():
                print(f"  bookmaker {bk}: markets {list((b.get('markets') or {}).keys())[:15]}")
            break
    print("\nDone. Send this output to Claude.")


def download(args):
    sid, _ = lol_sport_id()
    ts = our_tournaments(sid)
    rows = []
    for t in ts:
        fx = fixtures(t["tournamentId"], args.start)
        print(f"{t.get('tournamentName')}: {len(fx)} finished fixtures")
        for i, f in enumerate(fx):
            path = os.path.join(RAW, f"hist_{f['fixtureId']}.json")
            if os.path.exists(path):
                hist = json.load(open(path))
            else:
                hist = get("historical-odds", fixtureId=f["fixtureId"], bookmakers=BOOKMAKERS)
                save(f"hist_{f['fixtureId']}.json", hist)
            start = f.get("trueStartTime") or f.get("startTime") or ""
            for market in str(args.market).split(","):
                prices = closing_prices(hist, market, start.replace("Z", "+00:00"))
                for bk in sorted({bk for bk, _ in prices}):
                    o = {oid: p for (b, oid), p in prices.items() if b == bk}
                    if len(o) == 2:
                        (o1, p1), (o2, p2) = sorted(o.items())
                        rows.append({"fixtureId": f["fixtureId"], "tournament": t.get("tournamentName"),
                                     "start": start, "team1": f.get("participant1Name"),
                                     "team2": f.get("participant2Name"), "market": market,
                                     "bookmaker": bk, "outcome1": o1, "odds1": p1,
                                     "outcome2": o2, "odds2": p2})
            print(f"  {i + 1}/{len(fx)} {f.get('participant1Name')} vs {f.get('participant2Name')}")
    with open("odds_series.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["empty"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved odds_series.csv: {len(rows)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--market", help="market ids, comma-separated: 181 = series winner, 1847 = map 1 winner")
    ap.add_argument("--start", default="2026-06-01")
    args = ap.parse_args()
    if args.explore:
        explore(args)
    elif args.market:
        download(args)
    else:
        ap.print_help()
