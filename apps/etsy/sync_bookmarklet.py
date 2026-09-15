#!/usr/bin/env python3
"""Rebuild bookmarklet.html and copy export JS into the Chrome extension."""

from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
SRC = HERE / "export_etsy_results.js"
OUT = HERE / "bookmarklet.html"
EXT_JS = HERE / "extension" / "export_etsy_results.js"


def bookmarklet_body(js: str) -> str:
    text = js.lstrip()
    if text.startswith("/**"):
        end = text.find("*/")
        if end != -1:
            text = text[end + 2 :].lstrip()
    return text.replace("</script>", "<\\/script>")


def main() -> None:
    raw = SRC.read_text()
    EXT_JS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC, EXT_JS)
    print(f"copied {SRC.name} → {EXT_JS}")

    safe = bookmarklet_body(raw)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Install Export Etsy Results</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 680px; margin: 40px auto; padding: 0 16px; color: #1a1a1a; line-height: 1.45; }}
    h1 {{ font-size: 22px; }}
    h2 {{ font-size: 16px; margin-top: 1.6em; }}
    ol {{ padding-left: 1.3em; }}
    li {{ margin: 0.4em 0; }}
    button, a.bm {{
      display: inline-block; margin: 8px 8px 8px 0; padding: 12px 18px;
      background: #f1641e; color: #fff; border: 0; border-radius: 8px;
      font: 600 14px/1 system-ui, sans-serif; cursor: pointer; text-decoration: none;
    }}
    button.secondary {{ background: #444; }}
    button:hover, a.bm:hover {{ background: #d4571a; }}
    button.secondary:hover {{ background: #222; }}
    code {{ background: #f3f3f3; padding: 2px 6px; border-radius: 4px; }}
    .hint {{ color: #555; font-size: 14px; }}
    .ok {{ background: #eef8f1; border: 1px solid #b7dfc2; padding: 12px 14px; border-radius: 8px; }}
    .warn {{ background: #fff6e8; border: 1px solid #f0d9a8; padding: 12px 14px; border-radius: 8px; }}
    #status {{ margin-top: 12px; font-size: 14px; font-weight: 600; }}
    textarea {{
      width: 100%; height: 120px; margin-top: 8px; font: 11px/1.35 ui-monospace, monospace;
      box-sizing: border-box;
    }}
  </style>
</head>
<body>
  <h1>Export Etsy Results</h1>

  <div class="ok">
    <strong>Recommended: Chrome extension</strong> — install once, then click the
    toolbar icon on any Etsy search page. No re-pasting <code>javascript:</code> URLs.
  </div>

  <h2>1. Load the extension (once)</h2>
  <ol>
    <li>Open <code>chrome://extensions</code></li>
    <li>Turn on <strong>Developer mode</strong> (top right)</li>
    <li>Click <strong>Load unpacked</strong></li>
    <li>Choose this folder:
      <code>apps/etsy/extension</code>
      (full path: <code>{HERE / "extension"}</code>)</li>
    <li>Pin <strong>Export Etsy Results</strong> to the toolbar</li>
  </ol>
  <p class="hint">
    After editing <code>export_etsy_results.js</code>, run
    <code>python3 sync_bookmarklet.py</code> in <code>apps/etsy/</code>, then click
    <strong>Reload</strong> on the extension card in <code>chrome://extensions</code>.
  </p>

  <h2>2. Use it</h2>
  <ol>
    <li>Open an Etsy search-results page</li>
    <li>Click the extension icon</li>
    <li>Paste the JSON into the Makepad Etsy app</li>
  </ol>

  <h2>Fallback: bookmarklet (Chrome is awkward)</h2>
  <div class="warn">
    Chrome often refuses dragged <code>javascript:</code> bookmarks. Prefer the extension.
    If you still want a bookmark: copy the URL, then Add page → paste into the URL field.
  </div>
  <p>
    <button type="button" id="copy">Copy bookmarklet URL</button>
    <button type="button" id="show" class="secondary">Show URL</button>
  </p>
  <p id="status">Ready.</p>
  <textarea id="urlBox" hidden readonly></textarea>
  <p>
    <a class="bm" id="bookmark" href="#">Export Etsy Results</a>
    <span class="hint">drag may work in Safari / Firefox</span>
  </p>

  <p class="hint">
    Canonical source: <code>export_etsy_results.js</code>.
    Offline DOM fixture: <code>testdata/search_page.html</code>.
  </p>

  <script type="text/plain" id="bookmarklet-src">{safe}</script>

  <script>
    (function () {{
      var src = document.getElementById("bookmarklet-src").textContent.trim();
      var href = "javascript:" + encodeURIComponent(src);
      var link = document.getElementById("bookmark");
      link.setAttribute("href", href);

      var status = document.getElementById("status");
      var box = document.getElementById("urlBox");

      function copyUrl() {{
        function ok() {{
          status.textContent = "Copied. Paste into a bookmark URL field (must start with javascript:).";
        }}
        function fail() {{
          box.hidden = false;
          box.value = href;
          box.focus();
          box.select();
          status.textContent = "Clipboard blocked — select the URL below and copy manually.";
        }}
        if (navigator.clipboard && navigator.clipboard.writeText) {{
          navigator.clipboard.writeText(href).then(ok).catch(fail);
        }} else {{
          fail();
        }}
      }}

      document.getElementById("copy").addEventListener("click", copyUrl);
      document.getElementById("show").addEventListener("click", function () {{
        box.hidden = false;
        box.value = href;
        box.focus();
        box.select();
        status.textContent = "URL shown below — select all and copy.";
      }});
    }})();
  </script>
</body>
</html>
"""
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
