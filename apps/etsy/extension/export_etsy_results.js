/**
 * Export Etsy Results — readable source for the bookmarklet / Chrome extension.
 *
 * Preferred install: load apps/etsy/extension as an unpacked Chrome extension
 * (chrome://extensions → Developer mode → Load unpacked). Click the toolbar
 * icon on an Etsy search page. After editing this file, run:
 *   python3 sync_bookmarklet.py
 * then Reload the extension.
 *
 * Bookmarklet fallback: see bookmarklet.html (Chrome often blocks javascript: URLs).
 *
 * All Etsy-specific selectors live in SELECTORS below — update that object
 * when the live DOM changes. Fixture: testdata/search_page.html
 *
 * Shop review counts on search cards are shop popularity, not listing reviews.
 * Never treat "X out of 5 stars" aria as a review count.
 *
 * Only exports listing cards that are outermost for their listing id AND have
 * real layout (non-zero size, not display:none / aria-hidden). Nested
 * data-listing-id clones, hidden SEO/template cards, and promoted Ads
 * (Ad・By …) are skipped by default.
 */
(function () {
  // Set true to include promoted "Ad・" cards in the export.
  var INCLUDE_ADS = false;

  // ========== SELECTORS (update here when Etsy DOM changes) ==========
  var SELECTORS = {
    resultsRoots: [
      "[data-search-results]",
      "[data-appears-component-type='search_results']",
      ".search-listings-group",
      "#content .wt-grid",
      "#content",
      "main",
      "body"
    ],
    listingCards: [
      "[data-listing-id]",
      "li[data-appears-component-type='listing']",
      ".js-merch-stash-check-listing",
      ".etsy-fixture-card"
    ],
    adMarkers: [
      "[data-is-ad]",
      "[data-ad-listing]",
      ".ad-listing",
      "[aria-label*='Ad by']",
      "[aria-label*='Sponsored']"
    ],
    title: [
      "h3",
      "[data-listing-title]",
      ".v2-listing-card__title",
      "a.listing-link",
      ".etsy-fixture-title"
    ],
    link: [
      "a[href*='/listing/']",
      "a.listing-link",
      "a[data-listing-link]"
    ],
    price: [
      ".currency-value",
      "[data-buy-box-region] .currency-value",
      ".lc-price .currency-value",
      ".etsy-fixture-price"
    ],
    currencyCode: [
      ".currency-symbol",
      "[data-currency-code]",
      ".etsy-fixture-currency"
    ],
    postage: [
      "[data-shipping]",
      ".etsy-fixture-postage"
    ],
    // Hint selectors only — live extraction also walks all aria-labels
    // case-insensitively (CSS *= is case-sensitive; Etsy uses "Reviews").
    reviewCount: [
      ".etsy-fixture-reviews",
      "[aria-label*='review']",
      "[aria-label*='Review']",
      "[aria-label*='Bewertung']"
    ],
    shop: [
      "[data-shop-name]",
      ".etsy-fixture-shop",
      "p a[href*='/shop/']",
      "a[href*='/shop/']"
    ],
    image: [
      "img[data-listing-card-listing-image]",
      "img[src*='etsystatic']",
      "img[data-src]",
      "img",
      ".etsy-fixture-image"
    ]
  };
  // ========== end SELECTORS ==========

  function qs(root, list) {
    for (var i = 0; i < list.length; i++) {
      var el = root.querySelector(list[i]);
      if (el) return el;
    }
    return null;
  }

  function qsa(root, list) {
    for (var i = 0; i < list.length; i++) {
      var found = root.querySelectorAll(list[i]);
      if (found && found.length) return Array.prototype.slice.call(found);
    }
    return [];
  }

  function resultsRoot() {
    for (var i = 0; i < SELECTORS.resultsRoots.length; i++) {
      var el = document.querySelector(SELECTORS.resultsRoots[i]);
      if (el) return el;
    }
    return document.body;
  }

  function textOf(el) {
    return el ? (el.textContent || "").replace(/\s+/g, " ").trim() : "";
  }

  function attr(el, name) {
    return el && el.getAttribute ? (el.getAttribute(name) || "") : "";
  }

  function absUrl(href) {
    if (!href) return "";
    try {
      return new URL(href, location.href).href;
    } catch (e) {
      return href;
    }
  }

  function canonicalListingUrl(href) {
    var absolute = absUrl(href);
    if (!absolute) return "";
    try {
      var u = new URL(absolute);
      var m = u.pathname.match(/\/listing\/(\d+)(?:\/[^/]*)?/);
      if (!m) return absolute.split("?")[0].split("#")[0];
      return u.origin + "/listing/" + m[1];
    } catch (e) {
      return absolute.split("?")[0].split("#")[0];
    }
  }

  function listingIdFrom(card, url) {
    var id = attr(card, "data-listing-id") || attr(card, "data-palette-listing-id");
    if (id) return id;
    var m = (url || "").match(/\/listing\/(\d+)/);
    return m ? m[1] : "";
  }

  function parsePrice(text) {
    if (!text) return null;
    var m = String(text).replace(/,/g, "").match(/(\d+(?:\.\d+)?)/);
    return m ? parseFloat(m[1]) : null;
  }

  function detectCurrency(priceText, card) {
    var codeEl = qs(card, SELECTORS.currencyCode);
    var code = attr(codeEl, "data-currency-code") || textOf(codeEl);
    if (/gbp|£/i.test(code) || /£/.test(priceText)) return "GBP";
    if (/usd|\$/i.test(code) || /\$/.test(priceText)) return "USD";
    if (/eur|€/i.test(code) || /€/.test(priceText)) return "EUR";
    if (code && code.length <= 3) return code.toUpperCase();
    return null;
  }

  function queryFromUrl(url) {
    try {
      var u = new URL(url);
      return u.searchParams.get("q") || u.searchParams.get("search_query") || "";
    } catch (e) {
      return "";
    }
  }

  function pageFromUrl(url) {
    try {
      var u = new URL(url);
      var p = u.searchParams.get("page") || u.searchParams.get("ref") || "";
      // Etsy uses ?page=2; ignore non-numeric refs.
      if (/^\d+$/.test(p)) {
        var n = parseInt(p, 10);
        return n > 0 ? n : 1;
      }
      return 1;
    } catch (e) {
      return 1;
    }
  }

  /** Parse "1,234", "1.2k", "12k", "1.5m" into an integer count. */
  function parseCountToken(token) {
    if (!token) return null;
    var s = String(token).replace(/,/g, "").replace(/\s+/g, "").toLowerCase();
    if (/k$/.test(s)) {
      var k = parseFloat(s);
      return isFinite(k) ? Math.round(k * 1000) : null;
    }
    if (/m$/.test(s)) {
      var m = parseFloat(s);
      return isFinite(m) ? Math.round(m * 1000000) : null;
    }
    if (!/^\d+$/.test(s)) return null;
    var n = parseInt(s, 10);
    return isFinite(n) ? n : null;
  }

  /**
   * Extract shop review count from text. Never treats the star *rating*
   * ("4.8 out of 5 stars") as the count — that digit is 1–5.
   */
  function parseShopReviewCount(text, shopName) {
    if (!text) return null;
    var t = String(text);

    // Explicit count words (EN / DE). Prefer these even inside star aria-labels.
    var withWord =
      t.match(/(?:from\s+)?([\d,]+(?:\.\d+)?\s*[kKmM]?)\s*reviews?\b/i) ||
      t.match(/([\d.]+(?:\.\d+)?\s*[kKmM]?)\s*Bewertungen\b/i) ||
      t.match(/bei\s+([\d.]+)\s*Bewertungen\b/i);
    if (withWord) {
      var token = withWord[1];
      // German thousands: "1.234" (dot as separator, no k/m).
      if (/^\d{1,3}(\.\d{3})+$/.test(token)) {
        token = token.replace(/\./g, "");
      }
      var fromWord = parseCountToken(token);
      if (fromWord != null) return fromWord;
    }

    // Strip rating phrases so leftover "(1,284)" / "1.2k" can be read safely.
    var stripped = t
      .replace(/[\d.]+\s*(?:out of|\/)\s*5(?:\s*stars?)?/gi, " ")
      .replace(/\bstar\s*seller\b/gi, " ")
      .replace(/\b\d(?:\.\d)?\s*-\s*Sterne\b/gi, " ");

    // Parenthetical with k/m.
    var km = stripped.match(/\(([\d,]+(?:\.\d+)?\s*[kKmM])\)/i);
    if (km) {
      var fromKm = parseCountToken(km[1]);
      if (fromKm != null) return fromKm;
    }

    // Parenthetical near shop name: "CardCraftUK (1,284)".
    if (shopName) {
      var escaped = shopName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      var nearShop = stripped.match(
        new RegExp(
          escaped + "\\s*\\(([\\d,]+(?:\\.\\d+)?\\s*[kKmM]?)\\)",
          "i"
        )
      );
      if (nearShop) {
        var near = parseCountToken(nearShop[1]);
        if (near != null) return near;
      }
    }

    // Any remaining "(1,284)" / "(412)" after rating strip — skip "% off".
    var all = stripped.match(/\(([\d,]+(?:\.\d+)?\s*[kKmM]?)\)/gi) || [];
    for (var i = 0; i < all.length; i++) {
      var inner = all[i].replace(/^\(|\)$/g, "");
      if (/%/.test(all[i])) continue;
      var n = parseCountToken(inner);
      // Skip lone (1)–(5) that might still be rating residue; allow ≥6.
      if (n != null && n >= 6) return n;
    }

    // "4.8 out of 5 stars 1.2k" without parentheses.
    var bareK = stripped.match(/\b([\d]+(?:\.\d+)?\s*[kKmM])\b/i);
    if (bareK) {
      var bk = parseCountToken(bareK[1]);
      if (bk != null && bk >= 100) return bk;
    }

    return null;
  }

  // Expose for Node unit tests (testdata/review_parse_cases.json).
  if (typeof globalThis !== "undefined") {
    globalThis.__etsyParseShopReviewCount = parseShopReviewCount;
    globalThis.__etsyParseCountToken = parseCountToken;
  }

  function toast(msg, ok) {
    var el = document.createElement("div");
    el.setAttribute("role", "status");
    el.textContent = msg;
    el.style.cssText =
      "position:fixed;z-index:2147483647;right:16px;bottom:16px;" +
      "max-width:420px;padding:10px 14px;border-radius:8px;font:13px/1.35 system-ui,sans-serif;" +
      "box-shadow:0 4px 20px rgba(0,0,0,.25);color:#fff;" +
      (ok ? "background:#1b7a4a;" : "background:#8b2e2e;");
    document.documentElement.appendChild(el);
    setTimeout(function () {
      if (el.parentNode) el.parentNode.removeChild(el);
    }, 4000);
  }

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.style.cssText = "position:fixed;left:-9999px;top:0";
      document.body.appendChild(ta);
      ta.select();
      try {
        if (!document.execCommand("copy")) throw new Error("copy failed");
        resolve();
      } catch (e) {
        reject(e);
      } finally {
        document.body.removeChild(ta);
      }
    });
  }

  function isLaidOut(el) {
    if (!el || el.nodeType !== 1) return false;
    var cur = el;
    while (cur && cur.nodeType === 1) {
      if (cur.hidden) return false;
      if (attr(cur, "aria-hidden") === "true") return false;
      if (cur.hasAttribute && cur.hasAttribute("inert")) return false;
      var st = window.getComputedStyle(cur);
      if (!st || st.display === "none" || st.visibility === "hidden") return false;
      if (parseFloat(st.opacity || "1") === 0) return false;
      cur = cur.parentElement;
    }
    var r = el.getBoundingClientRect();
    if (!r || r.width < 4 || r.height < 4) return false;
    return true;
  }

  function isOutermostListingNode(el) {
    var id = attr(el, "data-listing-id");
    if (!id) return true;
    var p = el.parentElement;
    while (p) {
      if (attr(p, "data-listing-id") === id) return false;
      p = p.parentElement;
    }
    return true;
  }

  function cardFromNode(node) {
    var withId =
      (node.closest && node.closest("[data-listing-id]")) ||
      (node.getAttribute && node.getAttribute("data-listing-id") ? node : null);
    if (withId) return withId;
    if (node.matches && node.matches("a[href*='/listing/']")) {
      return (
        node.closest("li") ||
        node.closest("article") ||
        node.closest("div") ||
        node
      );
    }
    return node;
  }

  function isAdCard(card, raw) {
    if (qs(card, SELECTORS.adMarkers)) return true;
    var t = raw || textOf(card);
    if (/\bAd\s*[・·•]\s*By\b/i.test(t)) return true;
    if (/\bAd\s+from\s+shop\b/i.test(t)) return true;
    if (/\bSponsored\b/i.test(t) && /\bAd\b/i.test(t)) return true;
    return false;
  }

  function shopFromRaw(raw) {
    if (!raw) return "";
    var m =
      raw.match(/\bAd\s*[・·•]\s*By\s+([^\s|]+)/i) ||
      raw.match(/\bFrom\s+shop\s+([^\s|]+)/i) ||
      raw.match(/\bBy\s+([A-Za-z0-9_-]+)\s+From\s+shop/i) ||
      raw.match(/\bBy\s+([A-Za-z0-9_-]+)\b/);
    return m ? m[1].trim() : "";
  }

  function postageFromCard(card, raw) {
    var postageEl = qs(card, SELECTORS.postage);
    if (postageEl) {
      var pt = textOf(postageEl);
      if (isPostagePhrase(pt)) return pt;
    }
    if (!raw) return "";
    var free = raw.match(/\bFree(?:\s+UK)?\s+(?:delivery|shipping)\b/i);
    if (free) return free[0];
    var incl = raw.match(/£\s*[\d,.]+(?:\.\d+)?\s*incl\.?\s*postage/i);
    if (incl) return incl[0];
    var paid = raw.match(/£\s*[\d,.]+(?:\.\d+)?\s+(?:delivery|shipping|postage)\b/i);
    if (paid) return paid[0];
    return "";
  }

  function isPostagePhrase(pt) {
    if (!pt) return false;
    if (/original\s*price|sale\s*price|\(\s*\d+\s*%\s*off\s*\)/i.test(pt)) {
      return false;
    }
    return /free|delivery|shipping|postage|incl\.?\s*postage/i.test(pt);
  }

  function shopReviewCountFromCard(card, raw, shopName) {
    // 1) Walk every aria-label (case-insensitive). Live Etsy often puts the
    // count only in aria, e.g. "4.8 out of 5 stars, 1,284 Reviews".
    var labeled = card.querySelectorAll ? card.querySelectorAll("[aria-label]") : [];
    for (var i = 0; i < labeled.length; i++) {
      var aria = attr(labeled[i], "aria-label");
      if (!aria) continue;
      if (!/review|bewertung|star|rating|out of\s*5/i.test(aria)) continue;
      var fromAria = parseShopReviewCount(aria, shopName);
      if (fromAria != null) return fromAria;
    }

    // 2) Hint selectors + short parenthetical nodes next to stars.
    var candidates = qsa(card, SELECTORS.reviewCount);
    for (var j = 0; j < candidates.length; j++) {
      var el = candidates[j];
      var fromVis = parseShopReviewCount(
        attr(el, "aria-label") || textOf(el),
        shopName
      );
      if (fromVis != null) return fromVis;
    }
    var shorts = card.querySelectorAll
      ? card.querySelectorAll("p, span, div")
      : [];
    for (var k = 0; k < shorts.length; k++) {
      var t = textOf(shorts[k]);
      if (!t || t.length > 24) continue;
      if (!/^\([\d,.\s]*[kKmM]?\)$/i.test(t)) continue;
      var fromShort = parseShopReviewCount(t, shopName);
      if (fromShort != null) return fromShort;
    }

    return parseShopReviewCount(raw, shopName);
  }

  function extractListing(card) {
    var linkEl =
      qs(card, SELECTORS.link) ||
      (card.matches && card.matches("a[href*='/listing/']") ? card : null);
    var hrefRaw = linkEl ? linkEl.getAttribute("href") : "";
    var href = canonicalListingUrl(hrefRaw);
    var titleEl = qs(card, SELECTORS.title);
    var title = textOf(titleEl);
    if (!title && linkEl) title = attr(linkEl, "title") || textOf(linkEl);

    var priceEl = qs(card, SELECTORS.price);
    var priceText = textOf(priceEl);
    if (priceEl && !/£|\$|€/.test(priceText)) {
      var sym = qs(card, SELECTORS.currencyCode);
      var symText = textOf(sym);
      if (symText) priceText = (symText + priceText).trim();
    }
    var raw = textOf(card);
    var sale = raw.match(/Sale\s+Price\s*(£|\$|€)\s*([\d,.]+(?:\.\d+)?)/i);
    if (sale) {
      priceText = sale[1] + sale[2];
    }
    var price = parsePrice(priceText);
    var currency = detectCurrency(priceText, card);

    var postageText = postageFromCard(card, raw);

    var shopEl = qs(card, SELECTORS.shop);
    var shopName = shopEl ? textOf(shopEl) : "";
    if (!shopName) shopName = shopFromRaw(raw);
    shopName = shopName.replace(/^By\s+/i, "").trim();

    var shopUrl = "";
    if (shopEl) {
      var shopHref = attr(shopEl, "href");
      if (/\/shop\//i.test(shopHref)) shopUrl = absUrl(shopHref);
    }
    if (!shopUrl) {
      var anyShop = card.querySelector && card.querySelector("a[href*='/shop/']");
      if (anyShop) shopUrl = absUrl(attr(anyShop, "href"));
    }

    var reviewCount = shopReviewCountFromCard(card, raw, shopName);

    var imgEl = qs(card, SELECTORS.image);
    var imageUrl = "";
    if (imgEl) {
      imageUrl = absUrl(attr(imgEl, "src") || attr(imgEl, "data-src") || "");
    }

    var listingId = listingIdFrom(card, href);
    var ad = isAdCard(card, raw);

    var out = {
      title: title || null,
      url: href || absUrl(hrefRaw) || null,
      price_text: priceText || null,
      price: price,
      currency: currency,
      postage_text: postageText || null,
      shop_review_count: reviewCount,
      review_count: reviewCount, // alias for older importers
      shop_name: shopName || null,
      shop_url: shopUrl || null,
      image_url: imageUrl || null,
      raw_text: raw || null,
      is_ad: ad ? true : null
    };
    if (listingId) out.listing_id = listingId;

    Object.keys(out).forEach(function (k) {
      if (out[k] === null || out[k] === "") delete out[k];
    });
    return out;
  }

  function dedupeListings(listings) {
    var seen = {};
    var out = [];
    for (var i = 0; i < listings.length; i++) {
      var L = listings[i];
      var key =
        L.listing_id ||
        canonicalListingUrl(L.url || "") ||
        "t:" + (L.title || "") + ":" + i;
      if (seen[key]) continue;
      seen[key] = 1;
      out.push(L);
    }
    return out;
  }

  try {
    var root = resultsRoot();
    var nodes = qsa(root, SELECTORS.listingCards);
    var skippedNested = 0;
    var skippedHidden = 0;
    var skippedAds = 0;
    var cards = [];
    var seenCard = {};

    for (var i = 0; i < nodes.length; i++) {
      var card = cardFromNode(nodes[i]);
      if (!card) continue;
      if (!isOutermostListingNode(card)) {
        skippedNested++;
        continue;
      }
      if (!isLaidOut(card)) {
        skippedHidden++;
        continue;
      }
      var mark =
        (card.getAttribute && card.getAttribute("data-listing-id")) ||
        card;
      if (seenCard[mark] && typeof mark === "string") {
        continue;
      }
      if (typeof mark === "string") seenCard[mark] = 1;
      else {
        if (seenCard._els && seenCard._els.indexOf(card) >= 0) continue;
        seenCard._els = seenCard._els || [];
        seenCard._els.push(card);
      }
      cards.push(card);
    }

    var listings = dedupeListings(
      cards
        .map(extractListing)
        .filter(function (L) {
          if (!(L.title || L.url)) return false;
          if (!INCLUDE_ADS && L.is_ad) {
            skippedAds++;
            return false;
          }
          return true;
        })
    );

    if (!listings.length) {
      toast(
        "0 listings — DOM hits " +
          nodes.length +
          " (nested " +
          skippedNested +
          ", hidden " +
          skippedHidden +
          ", ads " +
          skippedAds +
          ").",
        false
      );
      return;
    }

    var href = location.href;
    var payload = {
      schema_version: 2,
      source: "etsy",
      source_url: href,
      query: queryFromUrl(href),
      page: pageFromUrl(href),
      exported_at: new Date().toISOString(),
      listings: listings
    };

    var json = JSON.stringify(payload, null, 2);
    var parts = [];
    if (skippedNested) parts.push(skippedNested + " nested");
    if (skippedHidden) parts.push(skippedHidden + " hidden");
    if (skippedAds) parts.push(skippedAds + " ads");
    var skipNote = parts.length ? " (skipped " + parts.join(", ") + ")" : "";
    var withReviews = 0;
    for (var r = 0; r < listings.length; r++) {
      if (listings[r].shop_review_count != null || listings[r].review_count != null) {
        withReviews++;
      }
    }
    var reviewNote = " · " + withReviews + " with shop reviews";
    copyText(json)
      .then(function () {
        toast(
          "Copied " + listings.length + " listings" + reviewNote + skipNote,
          true
        );
      })
      .catch(function () {
        toast(
          "Found " + listings.length + " listings but clipboard copy failed",
          false
        );
      });
  } catch (err) {
    toast("Export failed: " + (err && err.message ? err.message : String(err)), false);
  }
})();
