import argparse
import logging
import sys
from pathlib import Path
from waybacker.assets import PageUpdater, Requester
from waybacker.server import serve_directory
from waybacker.utils import DEFAULT_MAX_ASSET_BYTES, list_html_files, run_self_tests

def main():
    parser = argparse.ArgumentParser(description="Update Wayback download with assets, link rewriting, placeholders, and optional serving.")
    parser.add_argument("--root", required=True, help="Path to the existing download directory")
    parser.add_argument("--domain", required=True, help="Restrict to this domain (subdomains allowed by default)")
    parser.add_argument("--workers", type=int, default=8, help="Page workers")
    parser.add_argument("--asset-workers", type=int, default=16, help="Asset workers")
    parser.add_argument("--rps", type=float, default=5.0, help="Global requests/sec throttle")
    parser.add_argument("--allow-external", action="store_true", help="Permit third-party assets (off by default)")
    parser.add_argument("--no-subdomains", action="store_true", help="Disallow subdomains (default: allowed)")
    parser.add_argument("--allow-media", action="store_true", help="Download video/audio assets (default: skip)")
    parser.add_argument("--max-asset-bytes", type=int, default=DEFAULT_MAX_ASSET_BYTES, help="Per-asset size cap (bytes)")
    parser.add_argument("--no-strip-ia", action="store_true", help="Keep Internet Archive toolbar/chrome")
    parser.add_argument("--rewrite-links", action="store_true", help="Rewrite <a href> links to local pages when available")
    parser.add_argument("--missing-mode", choices=["leave", "drop", "placeholder"], default="placeholder",
                        help="Action for links to pages not downloaded: leave href, drop href, or point to a placeholder page")
    parser.add_argument("--serve", nargs="?", const=8000, type=int, help="Serve the folder after updating on the given PORT (default 8000)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be changed without writing files")
    parser.add_argument("--self-test", action="store_true", help="Run local sanity checks and exit")
    parser.add_argument("--log-level", default="INFO", help="Set logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="[%(levelname)s] %(message)s")

    if args.self_test:
        run_self_tests()
        return

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        logging.error(f"Root directory not found: {root}")
        sys.exit(2)

    scope_domain = args.domain.strip().lower()
    if "://" in scope_domain: scope_domain = scope_domain.split("://", 1)[1]
    scope_domain = scope_domain.split("/", 1)[0]

    include_subdomains = not args.no_subdomains
    allow_external = args.allow_external
    strip_ia = not args.no_strip_ia

    requester = Requester(rps=args.rps)

    manifest_path = root / "manifest.json"
    manifest = {"mode": "update", "root": str(root), "domain": scope_domain,
                "generated_at": __import__('time').strftime("%Y-%m-%d %H:%M:%S"),
                "settings": {"include_subdomains": include_subdomains, "allow_external": allow_external,
                            "allow_media": args.allow_media, "max_asset_bytes": args.max_asset_bytes,
                            "strip_ia": strip_ia, "rps": args.rps,
                            "workers": args.workers, "asset_workers": args.asset_workers,
                            "dry_run": args.dry_run, "rewrite_links": args.rewrite_links,
                            "missing_mode": args.missing_mode},
                "pages": [], "assets": []}

    updater = PageUpdater(
        root_dir=root, requester=requester, scope_domain=scope_domain,
        include_subdomains=include_subdomains, allow_external=allow_external,
        allow_media=args.allow_media, max_asset_bytes=args.max_asset_bytes,
        strip_ia=strip_ia, dry_run=args.dry_run, manifest=manifest,
        rewrite_links=args.rewrite_links, missing_mode=args.missing_mode
    )

    html_files = list(list_html_files(root))
    if not html_files:
        logging.info("No HTML files found under root. Nothing to update.")
        sys.exit(0)

    if args.rewrite_links:
        logging.info("Building page map for link rewriting...")
        updater.build_page_map_from_existing_html()
        logging.info(f"Mapped {len(updater.page_map)} originals -> newest local pages")

    logging.info(f"Found {len(html_files)} HTML files under {root}")
    logging.info(f"Fetching assets (rps≈{args.rps}, asset-workers={args.asset_workers}); rewriting pages")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=args.asset_workers) as asset_pool:
        with ThreadPoolExecutor(max_workers=args.workers) as page_pool:
            futures = [page_pool.submit(updater.update_html_file, p, asset_pool) for p in html_files]
            done = 0
            for fut in as_completed(futures):
                _ = fut.result()
                done += 1
                if done % 10 == 0:
                    logging.info(f"Updated {done}/{len(html_files)} pages...")

    import json
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    logging.info(f"Done. Updated pages: {len(manifest['pages'])}. Assets processed: {len(manifest['assets'])}")
    logging.info(f"Manifest written to: {manifest_path}")

    if args.serve is not None:
        port = args.serve if isinstance(args.serve, int) else 8000
        serve_directory(root, port)

if __name__ == "__main__":
    main()