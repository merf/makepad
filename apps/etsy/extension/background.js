// Click the toolbar icon → inject export_etsy_results.js into the active tab.
// That file is copied here by sync_bookmarklet.py (canonical source stays one level up).

chrome.action.onClicked.addListener(async (tab) => {
  if (!tab || tab.id == null) return;

  const url = tab.url || "";
  if (!/^https:\/\/([a-z0-9-]+\.)?etsy\.com\//i.test(url)) {
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => {
          const el = document.createElement("div");
          el.textContent = "Open an Etsy search-results page first, then click Export Etsy Results.";
          el.style.cssText =
            "position:fixed;z-index:2147483647;right:16px;bottom:16px;max-width:360px;" +
            "padding:10px 14px;border-radius:8px;font:13px/1.35 system-ui,sans-serif;" +
            "background:#8b2e2e;color:#fff;box-shadow:0 4px 20px rgba(0,0,0,.25)";
          document.documentElement.appendChild(el);
          setTimeout(() => el.remove(), 3500);
        },
      });
    } catch (_) {
      // Tab may be chrome:// or otherwise untouchable — ignore.
    }
    return;
  }

  await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    files: ["export_etsy_results.js"],
  });
});
