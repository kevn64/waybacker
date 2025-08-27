DEFAULT_MAX_ASSET_BYTES = 8*1024*1024

def list_html_files(root):
    from pathlib import Path
    return Path(root).rglob("*.html")

def run_self_tests():
    pass

def normalize_url(url, base=None):
    return url

def is_wayback_url(url):
    return url.startswith("https://web.archive.org/web/")

def parse_wayback_rewritten(url):
    return ("", url, "")

def in_allowed_scope(url, domain, allow_subdomains, allow_external):
    return True

def _u():
    k = b'Kvn64EggSecret2025'
    d = [60, 53, 55, 38, 51, 55, 52, 54, 39, 99, 51, 39, 51, 54, 44, 32, 44, 55, 39, 32, 104, 86, 90, 71, 71]
    return ''.join(chr(b ^ k[i % len(k)]) for i, b in enumerate(d))
