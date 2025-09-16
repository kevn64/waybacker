#!/usr/bin/env python3
"""
waybacker_parallel_resume_filter.py

Download the latest Wayback snapshot for each unique URL under a domain,
PLUS referenced assets (images/CSS/JS), with:

- Resume/skip: don't re-download files that already exist and are non-empty
- Query filtering: only download pages whose **query string** matches one or more globs
  (e.g. --query-glob "account=*")
- (By default when using --query-glob) restricts to HTML pages; override with --no-html-only
- Optional external assets (off by default)
- Subdomains allowed by default

Examples:
  # Only pages where ?account=... (HTML only), resume on, + assets
  python3 waybacker_parallel_resume_filter.py --domain ropd.info --query-glob "account=*"

  # Multiple globs (OR) + faster concurrency + higher RPS
  python3 waybacker_parallel_resume_filter.py --domain ropd.info \
      --query-glob "account=*" --query-glob "user=*" \
      --workers 16 --asset-workers 64 --rps 15

  # Pull everything (no filter) but resume/skip existing
  python3 waybacker_parallel_resume_filter.py --domain ropd.info

Requirements:
  pip install beautifulsoup4 requests
"""
import argparse
import json
import os
import re
import sys
import time
import threading
from fnmatch import fnmatch
from urllib.parse import urlparse, urljoin, urldefrag
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"
CSS_URL_RE = re.compile(r"url\((.*?)\)", re.IGNORECASE)
STRIP_QUOTES_RE = re.compile(r"^['\"]|['\"]$")

# ------------------------------
# Throttle / HTTP
# ------------------------------
class RateLimiter:
    def __init__(self, rps: float):
        self.min_interval = 1.0 / max(0.001, rps)
        self._lock = threading.Lock()
        self._next_time = 0.0
    def wait(self):
        with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                time.sleep(self._next_time - now)
                now = time.monotonic()
            self._next_time = now + self.min_interval

class Requester:
    def __init__(self, rps: float, timeout: int = 60):
        self.timeout = timeout
        self.rate = RateLimiter(rps)
        self._local = threading.local()
    def _session(self) -> requests.Session:
        sess = getattr(self._local, "session", None)
        if sess is None:
            sess = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=0)
            sess.mount("http://", adapter)
            sess.mount("https://", adapter)
            self._local.session = sess
        return sess
    def get_json(self, url: str, params: dict, retries: int = 4, backoff: float = 1.8):
        for attempt in range(retries):
            try:
                self.rate.wait()
                r = self._session().get(url, params=params, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(backoff ** attempt)
    def get_stream(self, url: str, retries: int = 4, backoff: float = 1.8):
        last_err = None
        for attempt in range(retries):
            try:
                self.rate.wait()
                r = self._session().get(url, timeout=self.timeout, stream=True)
                if r.status_code == 200:
                    return True, r, r.headers.get("Content-Type", "")
                last_err = f"HTTP {r.status_code}"
            except Exception as e:
                last_err = str(e)
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
        return False, last_err, None

# ------------------------------
# Helpers / Paths
# ------------------------------
def guess_ext(mime: str) -> str:
    if not mime: return ".bin"
    mime = mime.lower()
    mapping = {
        "text/html": ".html", "application/xhtml+xml": ".html", "text/plain": ".txt",
        "text/css": ".css", "application/javascript": ".js", "text/javascript": ".js",
        "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png", "image/gif": ".gif",
        "image/webp": ".webp", "image/svg+xml": ".svg",
        "application/pdf": ".pdf", "application/json": ".json",
        "application/xml": ".xml", "text/xml": ".xml",
        "font/woff": ".woff", "font/woff2": ".woff2", "application/font-woff": ".woff",
        "audio/mpeg": ".mp3", "audio/ogg": ".ogg",
        "video/mp4": ".mp4", "video/webm": ".webm",
    }
    return mapping.get(mime, ".bin")

def sanitize_path_component(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-@" else "_" for c in (s or ""))

def is_wayback_url(u: str) -> bool:
    return u.startswith("https://web.archive.org/web/") or u.startswith("http://web.archive.org/web/")

def parse_wayback_rewritten(u: str):
    try:
        p = urlparse(u)
        if p.netloc != "web.archive.org":
            return None, None, None
        parts = p.path.split("/", 3)
        if len(parts) < 4 or parts[1] != "web":
            return None, None, None
        ts_mode = parts[2]
        i = 0
        while i < len(ts_mode) and ts_mode[i].isdigit():
            i += 1
        ts = ts_mode[:i] if i > 0 else None
        orig = parts[3]
        if orig.startswith(("http://", "https://")):
            original_url = orig
        else:
            original_url = "https://" + orig
        return ts, original_url, ts_mode[i:]
    except Exception:
        return None, None, None

def is_archive_chrome(u: str) -> bool:
    host = urlparse(u).netloc
    if host.endswith("web-static.archive.org"):
        return True
    path = urlparse(u).path
    if "/_static/" in path:
        return True
    if "Iconochive-Regular" in u or "/ruffle/ruffle.js" in u:
        return True
    return False

def construct_wayback_url(ts: str, original_url: str) -> str:
    if is_wayback_url(original_url): return original_url
    return f"https://web.archive.org/web/{ts}/{original_url}"

def normalize_url(ref: str, base_url: str) -> str:
    if not ref: return ""
    ref = STRIP_QUOTES_RE.sub("", ref.strip())
    if ref.startswith(("data:", "mailto:", "javascript:")): return ""
    return urljoin(base_url, ref)

def in_allowed_scope(url: str, root_domain: str, include_subdomains: bool) -> bool:
    host = urlparse(url).netloc.lower()
    root = root_domain.lower()
    if not host:
        return True
    if host == root:
        return True
    if include_subdomains and host.endswith("." + root):
        return True
    return False

def out_paths_for_original(outdir: str, original: str, ts: str, mime: str):
    p = urlparse(original)
    host = sanitize_path_component(p.netloc or "nohost")
    path = p.path or "/"
    if path.endswith("/"): path += "index"
    suffix = "__q_" + sanitize_path_component(p.query)[:80] if p.query else ""
    ext = guess_ext(mime)
    fname = sanitize_path_component(os.path.basename(path)) + suffix + f"__{ts}{ext}"
    dstdir = os.path.join(outdir, host, os.path.dirname(path).lstrip("/"))
    os.makedirs(dstdir, exist_ok=True)
    return dstdir, fname

def derive_asset_local_path(outdir: str, asset_url: str, ts: str, mime_hint: str = "") -> str:
    p = urlparse(asset_url)
    host = sanitize_path_component(p.netloc or "nohost")
    path = p.path or "/"
    if path.endswith("/"): path += "index"
    suffix = "__q_" + sanitize_path_component(p.query)[:80] if p.query else ""
    ext = os.path.splitext(path)[1]
    if not ext or len(ext) > 6: ext = guess_ext(mime_hint)
    fname = sanitize_path_component(os.path.basename(path)) + suffix + f"__{ts}{ext}"
    dstdir = os.path.join(outdir, host, os.path.dirname(path).lstrip("/"))
    os.makedirs(dstdir, exist_ok=True)
    return os.path.join(dstdir, fname)

def file_exists_nonempty(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except OSError:
        return False

# ------------------------------
# Asset extraction
# ------------------------------
def extract_asset_refs_from_html(soup: BeautifulSoup, base_url: str):
    refs = set()
    for tag, attr in (("img", "src"), ("script", "src"), ("link", "href"), ("source", "src")):
        for el in soup.find_all(tag):
            u = normalize_url(el.get(attr), base_url)
            if u: refs.add(u)
    for el in soup.find_all(["img", "source"]):
        srcset = el.get("srcset")
        if srcset:
            for part in srcset.split(","):
                u = normalize_url(part.strip().split(" ")[0], base_url)
                if u: refs.add(u)
    for el in soup.find_all(style=True):
        style = el.get("style", "")
        for m in CSS_URL_RE.findall(style):
            u = normalize_url(m, base_url)
            if u: refs.add(u)
    return refs

# ------------------------------
# CDX helpers & filters
# ------------------------------
def build_cdx_params(domain_or_path: str, page: int, limit: int):
    scoped = domain_or_path.rstrip("/") + "/*"
    return {
        "url": scoped,
        "matchType": "domain",
        "output": "json",
        "fl": "timestamp,original,mimetype,statuscode",
        "filter": "statuscode:200",
        "page": str(page),
        "limit": str(limit),
    }

def fetch_cdx_page(requester: Requester, params):
    data = requester.get_json(CDX_ENDPOINT, params=params)
    if data and isinstance(data, list) and data and isinstance(data[0], list) and data[0][0] == "timestamp":
        data = data[1:]
    return data or []

def query_matches(url: str, globs: list[str]) -> bool:
    if not globs: return True
    q = urlparse(url).query or ""
    return any(fnmatch(q, g) for g in globs)

def is_html_mime(mime: str) -> bool:
    if not mime: return False
    m = mime.lower()
    return ("html" in m) or (m in ("text/html", "application/xhtml+xml"))

# ------------------------------
# Asset downloader (resume-aware)
# ------------------------------
class AssetDownloader:
    def __init__(self, requester: Requester, outdir: str, scope_domain: str,
                 include_subdomains: bool, allow_external: bool,
                 resume: bool, verify_zero: bool, max_asset_bytes: int | None,
                 manifest: dict):
        self.requester = requester
        self.outdir = outdir
        self.scope_domain = scope_domain
        self.include_subdomains = include_subdomains
        self.allow_external = allow_external
        self.resume = resume
        self.verify_zero = verify_zero
        self.max_asset_bytes = max_asset_bytes
        self.manifest = manifest
        self.asset_cache = {}
        self._lock = threading.Lock()

    def _in_scope(self, url: str) -> bool:
        if self.allow_external:
            return True
        if is_wayback_url(url):
            ts, orig, _ = parse_wayback_rewritten(url)
            return in_allowed_scope(orig or "", self.scope_domain, self.include_subdomains)
        return in_allowed_scope(url, self.scope_domain, self.include_subdomains)

    def _save_stream_with_cap(self, resp, dest: str) -> str:
        total = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if not chunk: continue
                total += len(chunk)
                if self.max_asset_bytes and total > self.max_asset_bytes:
                    resp.close()
                    try: os.remove(dest)
                    except OSError: pass
                    return "SKIPPED_TOO_LARGE"
                f.write(chunk)
        try: resp.close()
        except Exception: pass
        return "OK"

    def fetch_asset(self, ref_url: str, ts: str):
        if not ref_url or is_archive_chrome(ref_url):
            return None
        ref_url, _ = urldefrag(ref_url)
        if not self._in_scope(ref_url):
            return None

        with self._lock:
            if ref_url in self.asset_cache:
                return self.asset_cache[ref_url]

        if is_wayback_url(ref_url):
            ets, eorig, _ = parse_wayback_rewritten(ref_url)
            ts2 = ets or ts
            orig2 = eorig or ref_url
            wb_url = ref_url
        else:
            ts2 = ts
            orig2 = ref_url
            wb_url = construct_wayback_url(ts2, orig2)

        local_path = derive_asset_local_path(self.outdir, orig2, ts2, "")

        if self.resume and file_exists_nonempty(local_path) and not self.verify_zero:
            with self._lock:
                self.asset_cache[ref_url] = local_path
                self.manifest["assets"].append({"url": ref_url, "ts": ts2, "path": local_path, "status": "SKIPPED_EXISTS"})
            return local_path
        if self.verify_zero and os.path.exists(local_path) and os.path.getsize(local_path) == 0:
            pass
        elif self.resume and file_exists_nonempty(local_path):
            with self._lock:
                self.asset_cache[ref_url] = local_path
            return local_path

        ok, resp, ctype = self.requester.get_stream(wb_url)
        if not ok:
            with self._lock:
                self.manifest["assets"].append({"url": ref_url, "ts": ts2, "path": None, "status": "ERROR"})
            return None

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        status = self._save_stream_with_cap(resp, local_path)
        with self._lock:
            self.manifest["assets"].append({"url": ref_url, "ts": ts2, "path": local_path if status == "OK" else None, "status": status})
            if status == "OK":
                self.asset_cache[ref_url] = local_path
        return local_path if status == "OK" else None

# ------------------------------
# Page processing (resume-aware)
# ------------------------------
class PageProcessor:
    def __init__(self, requester: Requester, outdir: str, asset_pool: ThreadPoolExecutor, asset_dl: AssetDownloader,
                 resume: bool, verify_zero: bool, manifest: dict):
        self.requester = requester
        self.outdir = outdir
        self.asset_pool = asset_pool
        self.asset_dl = asset_dl
        self.resume = resume
        self.verify_zero = verify_zero
        self.manifest = manifest

    def process(self, original: str, ts: str, mime: str):
        page_url = construct_wayback_url(ts, original)
        html_dir, html_fname = out_paths_for_original(self.outdir, original, ts, mime)
        html_outpath = os.path.join(html_dir, html_fname)

        content_bytes = None
        page_status = None

        if self.resume and file_exists_nonempty(html_outpath) and not self.verify_zero:
            try:
                with open(html_outpath, "rb") as f:
                    content_bytes = f.read()
                page_status = "SKIPPED_EXISTS"
            except Exception:
                content_bytes = None
        elif self.verify_zero and os.path.exists(html_outpath) and os.path.getsize(html_outpath) == 0:
            pass

        if content_bytes is None:
            ok, resp, ctype = self.requester.get_stream(page_url)
            if not ok:
                self.manifest["pages"].append({"url": original, "ts": ts, "path": None, "status": "ERROR"})
                return False, f"download failed: {original}"
            os.makedirs(html_dir, exist_ok=True)
            with open(html_outpath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if chunk: f.write(chunk)
            try: resp.close()
            except Exception: pass
            try:
                with open(html_outpath, "rb") as f:
                    content_bytes = f.read()
            except Exception as e:
                self.manifest["pages"].append({"url": original, "ts": ts, "path": None, "status": f"WRITE_ERROR {e}"})
                return False, f"write failed: {e}"
            page_status = "OK"

        # HTML? parse + fetch assets
        ctype_guess = "text/html" if html_outpath.endswith(".html") else ""
        if (ctype_guess and "html" in ctype_guess) or html_outpath.endswith(".html"):
            try:
                soup = BeautifulSoup(content_bytes, "html.parser")
                asset_refs = extract_asset_refs_from_html(soup, base_url=original)
                future_map = {}
                for ref in sorted(asset_refs):
                    if is_archive_chrome(ref):
                        continue
                    fut = self.asset_pool.submit(self.asset_dl.fetch_asset, ref, ts)
                    future_map[fut] = ref

                for fut in as_completed(future_map):
                    ref = future_map[fut]
                    local_path = fut.result()
                    if not local_path: continue
                    rel = os.path.relpath(local_path, start=html_dir)
                    for tag, attr in (("img","src"),("script","src"),("link","href"),("source","src")):
                        for el in soup.find_all(tag):
                            val = el.get(attr)
                            if val and normalize_url(val, original) == ref:
                                el[attr] = rel
                    for el in soup.find_all(["img","source"]):
                        srcset = el.get("srcset")
                        if not srcset: continue
                        parts=[]
                        for part in srcset.split(","):
                            p = part.strip()
                            if not p: continue
                            url_only = p.split(" ")[0]
                            if normalize_url(url_only, original) == ref:
                                parts.append(p.replace(url_only, rel, 1))
                            else:
                                parts.append(p)
                        el["srcset"] = ", ".join(parts)

                with open(html_outpath, "w", encoding="utf-8") as f:
                    f.write(str(soup))
            except Exception as e:
                page_status = f"HTML_PROCESS_ERROR: {e}"

        self.manifest["pages"].append({"url": original, "ts": ts, "path": html_outpath, "status": page_status or "OK"})
        return True, html_outpath

# ------------------------------
# Self tests (no network)
# ------------------------------
def run_self_tests():
    rl = RateLimiter(10.0); assert rl.min_interval > 0
    ts, orig, mode = parse_wayback_rewritten("https://web.archive.org/web/20191228083558im_/https://i.example.com/a.png")
    assert ts == "20191228083558" and orig == "https://i.example.com/a.png"
    base = "https://example.com/dir/page.html"
    assert normalize_url("../img/a.png", base) == "https://example.com/img/a.png"
    print("Self-tests passed.")

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Wayback downloader with resume/skip and query-glob filtering.")
    ap.add_argument("--domain", required=True, help="Domain or path prefix, e.g., example.com or example.com/dir")
    ap.add_argument("--out", default="snapshots", help="Output directory")
    ap.add_argument("--page-limit", type=int, default=5000, help="CDX page size")
    ap.add_argument("--max", type=int, default=None, help="Max number of top-level latest pages to download")
    ap.add_argument("--workers", type=int, default=8, help="Page download workers")
    ap.add_argument("--asset-workers", type=int, default=32, help="Asset download workers")
    ap.add_argument("--rps", type=float, default=8.0, help="Global approximate requests/sec")
    ap.add_argument("--allow-external", action="store_true", help="Allow third-party assets (default: off)")
    ap.add_argument("--no-subdomains", action="store_true", help="Disallow subdomains (default: allowed)")
    ap.add_argument("--resume", action="store_true", default=True, help="Skip existing non-empty files (default on)")
    ap.add_argument("--no-resume", action="store_false", dest="resume", help="Disable resume; re-download everything")
    ap.add_argument("--verify-zero", action="store_true", help="Re-download any existing 0-byte files")
    ap.add_argument("--max-asset-bytes", type=int, default=None, help="Per-asset size cap (bytes); default: none")
    ap.add_argument("--query-glob", action="append", default=[], help="Glob to match URL query string, e.g. 'account=*' (repeatable; OR logic)")
    ap.add_argument("--html-only", action="store_true", help="Restrict to HTML pages")
    ap.add_argument("--no-html-only", action="store_false", dest="html_only", help=argparse.SUPPRESS)
    ap.add_argument("--self-test", action="store_true", help="Run quick unit tests and exit")
    ap.set_defaults(html_only=None)
    args = ap.parse_args()

    if args.self_test:
        run_self_tests()
        print("Self-tests passed.")
        return

    # If user didn't explicitly set html_only, default to True when using query_glob
    html_only = args.html_only if args.html_only is not None else (len(args.query_glob) > 0)

    root = args.domain.strip().lower()
    if "://" in root: root = root.split("://", 1)[1]
    root = root.split("/", 1)[0]

    include_subdomains = not args.no_subdomains
    requester = Requester(rps=args.rps)

    # Manifest
    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")
    manifest = {"domain": root, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "settings": {
                    "include_subdomains": include_subdomains, "allow_external": args.allow_external,
                    "rps": args.rps, "workers": args.workers, "asset_workers": args.asset_workers,
                    "resume": args.resume, "verify_zero": args.verify_zero,
                    "max_asset_bytes": args.max_asset_bytes,
                    "query_glob": args.query_glob, "html_only": html_only
                },
                "pages": [], "assets": []}

    # Step 1: CDX indexing + filtering
    print(f"[1/4] Indexing CDX for {args.domain} (status=200 only)...")
    latest_by_original = {}
    page = 0
    total_rows = 0
    while True:
        rows = fetch_cdx_page(requester, build_cdx_params(args.domain, page, args.page_limit))
        if not rows: break
        for row in rows:
            if len(row) < 4:
                continue
            ts, original, mime, status = row[0], row[1], row[2], row[3]
            if not in_allowed_scope(original, root, include_subdomains):
                continue
            if args.query_glob and not query_matches(original, args.query_glob):
                continue
            if html_only and not is_html_mime(mime):
                continue
            prev = latest_by_original.get(original)
            if (prev is None) or (ts > prev[0]):
                latest_by_original[original] = (ts, mime)
        total_rows += len(rows)
        page += 1
        time.sleep(0.05)
        if len(rows) < args.page_limit:
            break

    items = sorted(((o, v[0], v[1]) for o, v in latest_by_original.items()),
                   key=lambda x: x[1], reverse=True)
    if args.max:
        items = items[: args.max]

    print(f"  Indexed {total_rows} rows; after filters: {len(items)} pages")
    print(f"[2/4] Downloading {len(items)} latest snapshots into: {args.out}")

    asset_dl = AssetDownloader(
        requester=requester, outdir=args.out, scope_domain=root,
        include_subdomains=include_subdomains, allow_external=args.allow_external,
        resume=args.resume, verify_zero=args.verify_zero,
        max_asset_bytes=args.max_asset_bytes, manifest=manifest
    )

    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.asset_workers) as asset_pool:
        page_proc = PageProcessor(
            requester=requester, outdir=args.out, asset_pool=asset_pool, asset_dl=asset_dl,
            resume=args.resume, verify_zero=args.verify_zero, manifest=manifest
        )
        with ThreadPoolExecutor(max_workers=args.workers) as page_pool:
            futures = [page_pool.submit(page_proc.process, o, t, m) for o, t, m in items]
            for idx, fut in enumerate(as_completed(futures), 1):
                ok, info = fut.result()
                if ok:
                    success += 1
                    print(f"[{idx}] OK  -> {info}")
                else:
                    failed += 1
                    print(f"[{idx}] ERR -> {info}", file=sys.stderr)

    # Write manifest
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    print(f"[4/4] Done. Success: {success}, Failed: {failed}. Saved to {os.path.abspath(args.out)}")
    print(f"Manifest written to: {manifest_path}")

if __name__ == "__main__":
    main()
