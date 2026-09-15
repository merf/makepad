# Etsy Greeting Card Market Research Tool

## Goal

Build a small local market-research tool for comparing UK Etsy greeting-card listings.

There are two parts:

1. A browser bookmarklet called **Export Etsy Results**.
2. A small **Makepad** desktop app that accepts the exported data, stores it locally, and provides useful pricing/market analysis.

The purpose is not to build a generic Etsy scraper. The user will navigate Etsy normally, run a search, and explicitly click the bookmarklet on the current search-results page. The bookmarklet should only extract data from the page currently visible/loaded in the browser and copy the structured results to the clipboard.

Keep the implementation simple, robust and easy to understand.

---

# Part 1 — Etsy Export Bookmarklet

## User workflow

The intended workflow is:

1. Open Etsy UK in a normal browser.
2. Search for something such as:
   - funny birthday card
   - birthday card wife
   - cute birthday card
   - handmade greeting card
   - Christmas card
3. Optionally apply normal Etsy filters.
4. Let the current results page load.
5. Click a bookmark called **Export Etsy Results**.
6. The bookmarklet extracts the listing cards from the current page.
7. It copies a JSON payload to the clipboard.
8. The user pastes that payload into the Makepad app.

Do NOT automate navigation, login, pagination, requests, or repeated downloading of Etsy pages. This is an explicit user-initiated export of the page they are already viewing.

## Bookmarklet requirements

Create a small JavaScript bookmarklet.

It should:

- Run entirely in the current browser page.
- Find Etsy search-result listing cards from the DOM.
- Extract as many of these fields as can be reliably identified:
  - listing title
  - listing URL
  - displayed price
  - currency
  - displayed postage/delivery text
  - shop review count, if visible on the search card (this is shop popularity, not listing reviews — never confuse star-rating aria like "4.8 out of 5 stars" with the count)
  - shop name and shop URL, if visible
  - image URL, if readily available
  - search/query context and page number if obtainable from the current URL
- Preserve the raw displayed text for each listing where useful.
- Include a timestamp.
- Include the source URL.
- Copy JSON to the clipboard.
- Show a small unobtrusive success/failure message in the page.

Prefer stable semantic selectors and defensive extraction over brittle nth-child selectors.

The Etsy DOM may change. Put all Etsy-specific selectors in one clearly marked section so they are easy to update.

If the DOM structure cannot be reliably determined, fail gracefully and tell the user how many listings were found rather than producing misleading data.

## Data format

Use a versioned JSON format (current: schema_version 2; importers accept 1 or 2):

```json
{
  "schema_version": 2,
  "source": "etsy",
  "source_url": "...",
  "query": "...",
  "page": 1,
  "exported_at": "...",
  "listings": [
    {
      "title": "...",
      "url": "...",
      "price_text": "...",
      "price": 3.50,
      "currency": "GBP",
      "postage_text": "...",
      "shop_review_count": 1234,
      "review_count": 1234,
      "shop_name": "...",
      "shop_url": "...",
      "image_url": "...",
      "raw_text": "..."
    }
  ]
}
```

`shop_review_count` is the authoritative field; `review_count` is kept as an alias for older clipboard JSON. Search-page counts are **shop** reviews.
Do not assume every field exists.

The Makepad app should accept the JSON even if some fields are null/missing.

---

# Part 2 — Makepad Application

## General direction

Build a small desktop application using Makepad.

Use Makepad's existing UI capabilities rather than inventing a large framework.

The application should have a simple structure:

```text
Etsy Market Research
├── Import
├── Listings
├── Analysis
└── Searches
```

Use a local database/storage mechanism supported naturally by the Makepad project/environment. Keep persistence local and simple.

Avoid unnecessary networking.

---

# Import screen

The primary screen should have:

- Large multiline text box.
- Button: **Import Etsy Export**
- Button: **Clear**
- Import status.
- Number of listings imported.
- Search/query detected.
- Export timestamp.

The user should be able to paste the bookmarklet's JSON directly.

Also provide a useful error message if invalid JSON is pasted.

On import:

1. Validate schema version.
2. Parse listings.
3. Normalise numeric values.
4. Store the original/raw data.
5. Deduplicate obvious duplicates.
6. Record which search/export the listing came from.

Do not silently discard data. Where parsing is uncertain, retain the raw text and mark the field as unknown.

---

# Database model

Keep the schema relatively small.

Suggested entities:

## Search

```text
Search
- id
- source
- query
- source_url
- exported_at
```

## Listing

```text
Listing
- id
- search_id
- source_listing_id or URL
- title
- url
- shop_name
- price
- currency
- postage
- delivered_price
- review_count
- pack_size
- price_per_card
- personalised
- digital
- category
- style
- raw_text
- image_url
- first_seen
- last_seen
```

Some fields may initially be unknown.

Use URL as the primary deduplication key where possible.

---

# Normalisation

This is important.

A greeting-card market dataset is only useful if obvious non-comparable products are removed or classified.

Implement simple normalisation first.

## Price

Convert displayed GBP values to a numeric price.

Examples:

```text
£3.50 -> 3.50
£12.99 -> 12.99
```

Keep the original `price_text`.

## Postage

Try to identify:

- free delivery
- £X delivery
- unknown

Store:

```text
postage = 0
```

for explicit free delivery.

Do NOT treat unknown postage as zero.

Use a separate boolean/status such as:

```text
postage_known
```

or an enum.

## Delivered price

Where both price and postage are known:

```text
delivered_price = price + postage
```

Otherwise leave it unknown.

## Pack size

Try to infer quantities from titles/raw text:

```text
single -> 1
pack of 4 -> 4
set of 6 -> 6
6 cards -> 6
10 greeting cards -> 10
```

Do not guess aggressively.

If uncertain, leave pack size unknown.

## Price per card

If price and pack size are known:

```text
price_per_card = price / pack_size
```

Also calculate:

```text
delivered_price_per_card
```

where delivered price is known.

---

# Classification

Create simple flags/categories to make analysis useful.

At minimum:

```text
physical
digital
personalised
bundle
single
pack
```

Try to detect obvious digital downloads and exclude them from default physical-card statistics.

Likewise flag highly personalised products separately.

Do not pretend this classification is perfect.

The UI should allow the user to toggle filters.

---

# Analysis screen

This is the main purpose of the app.

Provide a compact dashboard.

## Top-level metrics

Show:

- Number of listings
- Number of physical listings
- Number of shops
- Median item price
- Median delivered price
- Median postage
- Median price per card
- Median delivered price per card
- Median review count

Prefer median over mean for prices because Etsy data will contain outliers.

Also show P25 and P75 where useful.

Example:

```text
Physical listings: 184

Item price
P25     £2.75
Median  £3.50
P75     £4.25

Delivered price
P25     £4.25
Median  £5.00
P75     £6.00

Postage
P25     £0.00
Median  £1.50
P75     £1.80
```

---

# Price distribution

Show a histogram or equivalent visualisation of:

- item price
- delivered price
- price per card

Allow switching between these.

This should make the market's price bands obvious.

---

# Singles vs multipacks

This is particularly important.

Group listings by pack size:

```text
Pack size | Listings | Median price | Median delivered | Median/card
1         | 82       | £3.50        | £5.00             | £3.50
4         | 31       | £9.95        | £11.50            | £2.49
6         | 38       | £12.95       | £14.45            | £2.16
8         | 27       | £14.95       | £16.45            | £1.87
10        | 22       | £17.95       | £19.45            | £1.80
```

Use whatever actual data exists; the above is illustrative only.

This view should help answer:

> How much discount are sellers giving buyers for multipacks?

---

# Postage analysis

Show:

- percentage with free postage
- median paid postage
- distribution of postage prices
- delivered price vs item price

Useful table:

```text
                         Free postage    Median postage
Singles                     32%              £1.80
4-packs                     55%              £0.00
6-packs                     61%              £0.00
8-packs                     70%              £0.00
```

Again, calculate from actual data.

The UI should make it obvious that:

> £3.00 + £1.80 postage

is a £4.80 customer purchase.

---

# Popularity analysis

Reviews are NOT sales, so do not call them sales.

Use review count as a crude popularity/market-presence indicator.

Show:

- median reviews
- P25/P75 reviews
- top listings by review count
- price vs review count

A scatter plot of:

```text
X = delivered price
Y = review count
```

would be particularly useful.

Consider also a log-scale Y axis if Makepad's charting makes that easy.

Do not imply causality.

---

# "Successful price bands"

Create a useful heuristic view.

For example, group listings into delivered-price bands:

```text
£3–£4
£4–£5
£5–£6
£6–£7
£7+
```

For each band show:

- listing count
- median reviews
- median review count
- maximum review count
- percentage of listings with free postage

This lets the user see whether expensive cards appear to have a meaningful popularity disadvantage.

Again, explicitly label this as a heuristic, not a measure of sales.

---

# Shop-level analysis

Show a table of shops:

```text
Shop | Listings | Median price | Median reviews | Max reviews
```

Useful for spotting strong competitors.

Allow sorting by:

- listings
- median price
- median reviews
- max reviews

Do not overcomplicate this initially.

---

# Search comparison

Because the user will export multiple Etsy searches, retain the search context.

Allow analysis by query:

```text
funny birthday card
cute birthday card
birthday card wife
birthday card husband
Christmas card
```

For each search show:

- listing count
- median price
- median delivered price
- median postage
- median reviews
- median price/card

This is important because different card niches can have significantly different pricing.

---

# Product filtering

Provide simple filters:

```text
[ ] Physical only
[ ] Exclude personalised
[ ] Exclude digital
[ ] Singles only
[ ] Packs only

Pack size: [Any]
Search:    [Any]
Price:     [Any]
```

Filters should update the analysis.

---

# Recommended-price helper

Eventually add a section called:

## Pricing benchmark

This should NOT claim to calculate the correct price.

Instead provide market benchmarks such as:

```text
Comparable physical cards:

25th percentile delivered price: £X
Median delivered price:           £Y
75th percentile delivered price:  £Z

Recommended benchmark range:
£Y–£Z
```

For multipacks:

```text
Single:
Median delivered: £X

Pack of 4:
Median delivered: £Y
Median/card: £Z

Pack of 6:
Median delivered: £Y
Median/card: £Z
```

The user can then manually decide where their own product belongs based on quality, design, packaging, personalisation, etc.

Do not automatically recommend a price without explaining that product quality/brand/design affects positioning.

---

# Data quality

Add a small data-quality indicator.

For example:

```text
Dataset
184 listings
92% have price
81% have postage
76% have review count
64% have inferred pack size
```

This prevents the dashboard from looking more precise than the underlying data.

Also show warnings for:

- listings with unknown postage
- listings with unknown pack size
- suspiciously parsed prices
- likely digital products
- likely personalised products

---

# Importing repeated searches

The app should make repeated collection easy.

Typical session:

```text
Search Etsy: funny birthday card
Export bookmarklet
Paste JSON
Import

Search Etsy: cute birthday card
Export bookmarklet
Paste JSON
Import

Search Etsy: birthday card wife
Export bookmarklet
Paste JSON
Import
```

The app should retain all of these.

Avoid overwriting previous imports.

If the same listing appears in multiple searches, store one listing with multiple search associations if practical, or otherwise deduplicate it while retaining search membership.

---

# UI design

Keep the UI clean and utilitarian.

Suggested layout:

```text
┌─────────────────────────────────────────────────────┐
│ Etsy Greeting Card Market Research                  │
├────────────┬────────────────────────────────────────┤
│ Import     │                                        │
│ Listings   │            Main Content                │
│ Analysis   │                                        │
│ Searches   │                                        │
├────────────┴────────────────────────────────────────┤
│                                                    │
└─────────────────────────────────────────────────────┘
```

Use Makepad's existing widgets and styling conventions.

Do not build a complicated navigation framework.

Prioritise readability of numbers and tables.

---

# Implementation order

Do this incrementally.

## Phase 1

Build bookmarklet:

- detect Etsy result cards
- extract title/url/price/postage/reviews
- copy JSON
- success/error feedback

## Phase 2

Build Makepad app:

- basic window
- import screen
- JSON parser
- local persistence
- listings table

## Phase 3

Normalisation:

- price
- postage
- delivered price
- pack size
- price/card
- digital/personalised flags

## Phase 4

Analysis:

- summary metrics
- price distributions
- singles vs multipacks
- postage analysis
- popularity analysis

## Phase 5

Search/shop comparison:

- search breakdown
- shop breakdown
- filters

## Phase 6

Pricing benchmark view.

Do not attempt all phases at once.

---

# Important engineering preferences

The user is an experienced C++ gameplay programmer and prefers:

- simple, direct code
- easy-to-follow implementations
- minimal abstraction
- C++11/14-era style where practical
- custom/simple containers where the existing Makepad codebase provides them
- no unnecessary bleeding-edge language features
- avoid architecture astronautics

Follow the conventions of the existing Makepad project rather than imposing a foreign architecture.

Before implementing anything substantial, inspect the existing project structure and identify:

- how Makepad applications are structured
- existing widgets suitable for tables/charts
- existing persistence/database support
- existing clipboard support
- existing JSON parsing
- existing plotting/chart components if present

Reuse those facilities.

---

# Testing

Create representative fixture data rather than relying on live Etsy data during development.

Include test cases for:

1. Single card with free postage.
2. Single card with £1.80 postage.
3. Pack of 4.
4. Pack of 6.
5. Pack of 10.
6. Digital download.
7. Personalised card.
8. Missing review count.
9. Unknown postage.
10. Currency other than GBP.
11. Duplicate listing appearing in two searches.
12. Malformed/incomplete JSON.
13. Etsy DOM extraction finding zero listings.
14. Etsy DOM structure changing slightly.

Do not make tests depend on the live Etsy website.

---

# Important scope constraint

This is a **market research helper**, not a production scraping service.

Do not add:

- automatic Etsy login
- automatic browsing
- automatic pagination
- proxy rotation
- CAPTCHA handling
- anti-bot evasion
- high-frequency requests
- background Etsy crawling

The bookmarklet should only operate when the user explicitly clicks it on a page they are viewing.

---

# First task for Cursor

Start by inspecting the current Makepad project.

Then:

1. Explain briefly how the existing project handles UI, JSON, persistence and clipboard.
2. Identify the simplest existing components that can implement the requested UI.
3. Implement Phase 1 bookmarklet and a minimal Makepad import screen.
4. Use fixture JSON to demonstrate the import path before worrying about live Etsy DOM extraction.
5. Keep the implementation small and easy to modify.

Do not build the entire application in one pass.

After Phase 1/2 works, iterate on the analysis UI.
