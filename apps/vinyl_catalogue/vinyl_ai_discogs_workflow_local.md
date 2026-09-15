# Local-First Vinyl Collection Catalogue

## Purpose

Build a small personal tool for rapidly cataloguing a collection of late-1990s / early-2000s house and techno records.

The physical workflow should be:

**pick up record → speak its details → put it down → next record**

The software should then:

1. transcribe the spoken notes locally
2. turn the transcription into structured records using a local LLM
3. search Discogs for likely releases
4. let the user confirm the correct pressing
5. provide a useful indication of value
6. flag records worth separating from the bulk collection
7. maintain a local catalogue and export CSV

The primary objective is **finding the valuable / interesting records**, not producing a perfect retail valuation for every £2 record.

---

# Core principle: 100% local except Discogs

The application should be designed so that **audio, transcription, LLM processing and the catalogue database never need to leave the computer**.

The one unavoidable external dependency is Discogs, because we want to query its catalogue.

The resulting architecture is:

```text
                 LOCAL
┌──────────────────────────────────────────────┐
│                                              │
│  Microphone                                  │
│      ↓                                       │
│  Local speech recognition                    │
│      ↓                                       │
│  Raw transcript                              │
│      ↓                                       │
│  Local LLM                                   │
│      ↓                                       │
│  Structured record                           │
│      ↓                                       │
│  Discogs API ────────────────────────┐       │
│      ↓                               │       │
│  Candidate releases                  │       │
│      ↓                               │       │
│  Human confirmation                  │       │
│      ↓                               │       │
│  Local SQLite database ←─────────────┘       │
│                                              │
└──────────────────────────────────────────────┘
```

Discogs is therefore the only network service required for the core workflow.

---

# 1. Speech recognition: faster-whisper

## Recommended choice

Use **faster-whisper** for local speech-to-text.

faster-whisper is an efficient reimplementation of OpenAI Whisper using CTranslate2. Its project reports substantially faster inference and lower memory use than the original Whisper implementation, with further gains possible through quantisation. It can run entirely locally. 

Repository:

https://github.com/SYSTRAN/faster-whisper

Installation:

```text
pip install faster-whisper
```

The project currently supports Python 3.9+.

## Model

Start with:

```text
large-v3
```

If that proves unnecessarily slow on the machine, try:

```text
medium
```

or:

```text
small
```

The vinyl application is an unusually good use case for Whisper because the speech is likely to be:

- short
- fairly clear
- repetitive
- mostly proper names
- separated by a known word such as "next"

The main challenge is not normal English speech recognition. It is getting unusual names such as:

```text
Surgeon
Tresor
Chain Reaction
Underground Resistance
Rephlex
Peacefrog
Jeff Mills
Basic Channel
```

correct.

Therefore test a few models against real vinyl-related speech rather than assuming the largest model is automatically necessary.

## Why local Whisper rather than the OpenAI API?

Advantages:

- no per-minute API cost
- no audio leaves the machine
- works without an internet connection
- easy to process long batches
- no dependency on an external speech service

The trade-off is that the initial setup is slightly more involved and inference speed depends on the computer.

---

# 2. Audio capture

There are two reasonable approaches.

## Option A: browser microphone

Use a tiny local web application.

The browser handles microphone permission and records audio using `MediaRecorder`.

```text
Browser
  ↓
MediaRecorder
  ↓
audio blob
  ↓
Python backend
  ↓
faster-whisper
```

This is probably the easiest first implementation.

Potentially useful browser API:

```text
MediaRecorder
```

No special audio library is required for microphone capture.

## Option B: Python microphone capture

Use:

```text
sounddevice
```

and record directly from Python.

This gives more control but adds another layer of platform-specific audio handling.

### Recommendation

Start with **browser microphone capture**.

The browser UI can have:

```text
[ RECORD ]

Transcript:
Dave Angel, Arpeggio, Rotation Records, R-001.
Next.
Jeff Mills, The Bells, Purpose Maker, 1996.
Next.

[ STOP ]
```

The actual transcription can happen in batches.

---

# 3. Do not start with realtime transcription

It is tempting to build:

```text
microphone
→ streaming speech recognition
→ words appearing live
```

but that adds complexity without much benefit initially.

Instead:

```text
Record 30–120 seconds
        ↓
send audio to faster-whisper
        ↓
receive transcript
```

The user can keep flipping through records while recording.

The system can then process the batch after stopping.

This separates the **physical cataloguing workflow** from the **computer processing workflow**.

---

# 4. Spoken input convention

Keep the convention simple:

```text
Artist — Title — Label — Catalogue Number — Notes — Next
```

For example:

```text
Dave Angel, Arpeggio, Rotation Records, R-001, next.

Jeff Mills, The Bells, Purpose Maker, PM-001, 1996, next.

Underground Resistance, Jupiter Jazz, red sleeve, next.
```

Missing information is completely acceptable.

The user should not have to remember an exact syntax.

The important word is:

```text
next
```

because it gives the local LLM a strong boundary between records.

---

# 5. Local LLM for structuring the transcript

The second AI stage should also be local.

## Recommended tool: Ollama

Use **Ollama** as the local model runner.

It provides a simple local API and supports structured JSON output using JSON schemas. That makes it particularly suitable for turning messy speech transcripts into reliable records. citeturn0search0turn0search5

Website:

https://ollama.com/

Python package:

```text
ollama
```

The application can send the transcript to Ollama running on:

```text
localhost
```

No transcript needs to leave the machine.

## What the LLM does

The LLM should have one very narrow job:

**convert speech into structured record data.**

Input:

```text
Dave Angel Arpeggio Rotation R-001 next
Jeff Mills Bells Purpose Maker 1996 next
Underground Resistance Jupiter Jazz red sleeve next
```

Output:

```json
[
  {
    "artist": "Dave Angel",
    "title": "Arpeggio",
    "label": "Rotation Records",
    "catalogue_number": "R-001",
    "notes": ""
  },
  {
    "artist": "Jeff Mills",
    "title": "The Bells",
    "label": "Purpose Maker",
    "catalogue_number": "",
    "notes": "1996"
  },
  {
    "artist": "Underground Resistance",
    "title": "Jupiter Jazz",
    "label": "",
    "catalogue_number": "",
    "notes": "red sleeve"
  }
]
```

Ollama supports passing a JSON schema directly, so the application can validate that the model returns the expected structure rather than relying on free-form text. citeturn0search0turn0search2

## Model choice

Do not overthink the local LLM model initially.

The task is relatively constrained and does not require sophisticated reasoning.

Use a current small/medium general-purpose instruct model available through Ollama and test it against actual transcripts.

The important thing is that it should be good at:

- extracting fields
- preserving proper nouns
- not inventing missing information
- recognising "next" as a record boundary
- returning valid structured output

The LLM should **never invent a catalogue number or release identification**.

---

# 6. Discogs identification

Once records are structured, search Discogs.

This is the one network-dependent part of the application.

Use the official Discogs API rather than scraping Discogs pages for database metadata.

Search progressively:

```text
1. catalogue number + label

2. artist + title + label

3. artist + title

4. artist + title + notes
```

Catalogue number is particularly valuable for old dance 12"s.

Example:

```text
Artist:
Jeff Mills

Title:
The Bells

Label:
Purpose Maker

Catalogue:
PM-001
```

should produce much better candidate matches than simply searching:

```text
Jeff Mills The Bells
```

---

# 7. Human confirmation is essential

Do not automatically select the first Discogs result.

The exact pressing can matter enormously.

The application should show something like:

```text
Jeff Mills – The Bells

1. Purpose Maker – PM-001
   1996 / US / 12"

2. Purpose Maker – PM-001
   1998 reissue

3. Purpose Maker – PM-001
   White label / promo

[ 1 ] [ 2 ] [ 3 ] [ UNRESOLVED ]
```

Ideally show enough information to distinguish the releases:

- year
- country
- label
- catalogue number
- format
- release notes
- artwork where available
- identifiers
- Discogs release link

The user picks the correct pressing.

This is deliberately **human-in-the-loop**.

The application should never turn:

```text
AI guessed the release
```

into:

```text
therefore this record is worth £75
```

without confirmation.

---

# 8. Discogs valuation

## Important distinction

Discogs contains two broad classes of useful information.

### Database information

Useful for automated identification:

- artist
- title
- label
- catalogue number
- release year
- country
- format
- track listing
- barcodes
- other identifiers
- release information

### Marketplace information

This includes things such as:

- current Marketplace listings
- prices
- sales history
- pricing suggestions

Discogs treats Marketplace information as restricted data under its API terms.

Therefore the application should **not be designed as a bulk Marketplace-price scraper**.

Instead, make the identification system independent of valuation.

After confirming the release, the application can provide a link to the Discogs release page, where the user can inspect available Marketplace/sales information.

If Discogs provides an explicitly permitted API mechanism for a particular pricing field, it can be incorporated later.

The key point is:

> **The application remains useful even if automated price retrieval changes or is unavailable.**

---

# 9. What we actually need from valuation

We do not need perfect prices.

We need to answer:

> "Is this record worth pulling out before I sell the rest as a job lot?"

Therefore the useful categories are:

```text
BULK
CHECK
VALUABLE
HIGH_VALUE
UNRESOLVED
```

Initial thresholds:

```text
£0–5       BULK
£5–15      CHECK
£15–40     VALUABLE
£40+       HIGH_VALUE
```

Make the thresholds configurable.

The value should be explicitly labelled as something like:

```text
Approximate Discogs value
```

rather than:

```text
Record value
```

---

# 10. Local database

Use **SQLite**.

There is no reason to introduce PostgreSQL or another server database.

One local file is ideal:

```text
vinyl.db
```

Suggested main table:

```text
records
-------
id
record_number

raw_transcript

artist
title
label
catalogue_number
notes

discogs_release_id
discogs_url
discogs_artist
discogs_title
discogs_label
discogs_catalogue_number
discogs_year
discogs_country
discogs_format

value_low
value_median
value_high
value_source

match_confidence
status

created_at
updated_at
```

Keep the raw transcript.

It will be useful when investigating a bad transcription or an unresolved record.

---

# 11. Suggested UI

Keep the UI extremely simple.

## Capture screen

```text
┌─────────────────────────────────────────────────────┐
│ VINYL CATALOGUE                                     │
│                                                     │
│ Batch: 003                                          │
│                                                     │
│                 [ RECORD ]                          │
│                                                     │
│ "Surgeon, Magneze, Tresor, T-179, next..."          │
│                                                     │
│                 [ STOP ]                            │
│                                                     │
│ 27 records captured                                 │
└─────────────────────────────────────────────────────┘
```

After transcription:

```text
┌────────────────────────────────────────────────────────┐
│ CAPTURED RECORDS                                       │
│                                                        │
│ 001  Surgeon       Magneze        Tresor     T-179     │
│ 002  Dave Clarke   Red 2          Bush       BUSH 001  │
│ 003  Robert Hood   Internal Empire Axis                │
│                                                        │
│                  [ RESOLVE WITH DISCOGS ]              │
└────────────────────────────────────────────────────────┘
```

Then:

```text
┌────────────────────────────────────────────────────────┐
│ RECORD 002                                              │
│                                                        │
│ Dave Clarke – Red 2                                   │
│ Bush – BUSH 001                                       │
│                                                        │
│ 1  1994 UK 12"                                        │
│    [ SELECT ]                                         │
│                                                        │
│ 2  1995 reissue                                       │
│    [ SELECT ]                                         │
│                                                        │
│ 3  Promo                                               │
│    [ SELECT ]                                         │
│                                                        │
│             [ UNRESOLVED ]                            │
└────────────────────────────────────────────────────────┘
```

The emphasis should be on **keyboard shortcuts and speed**.

For example:

```text
1 = select candidate 1
2 = select candidate 2
3 = select candidate 3
U = unresolved
N = next record
E = edit
```

This could make processing hundreds of records much quicker.

---

# 12. Batch processing

Do not wait for Discogs after every record.

Use batches.

```text
Physical session
──────────────────────────────────

record
record
record
record
...
record
        ↓
30–50 records
        ↓
transcribe
        ↓
structure
        ↓
Discogs searches
        ↓
confirm candidates
```

This keeps the physical activity uninterrupted.

The user can process another crate while the computer works through a previous batch if desired.

---

# 13. Optional photographs

Do not make photography part of the initial workflow.

Add it later for:

- unresolved records
- unusual pressings
- white labels
- promos
- potentially valuable records

Useful photos:

```text
front sleeve
back sleeve
label
runout / matrix
```

The photograph can then help distinguish between otherwise identical Discogs releases.

A later local vision model could potentially extract text from labels and sleeves, but this is an enhancement rather than an MVP requirement.

---

# 14. Recommended project structure

Something like:

```text
vinyl-catalogue/
│
├── app/
│   ├── main.py
│   ├── audio.py
│   ├── transcribe.py
│   ├── extract.py
│   ├── discogs.py
│   ├── database.py
│   └── valuation.py
│
├── web/
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── data/
│   └── vinyl.db
│
├── recordings/
│
├── exports/
│
├── requirements.txt
└── README.md
```

This is deliberately boring.

There is no need for a complicated architecture.

---

# 15. Python dependencies

Likely starting set:

```text
faster-whisper
ollama
flask
requests
```

Potentially:

```text
sounddevice
numpy
```

if using Python-side microphone capture.

SQLite is built into Python.

CSV handling is also built in.

The browser provides microphone recording if using the recommended browser approach.

---

# 16. First implementation

Build this in stages.

## Stage 1 — Whisper test

Before writing the application:

1. Install faster-whisper.
2. Record yourself reading 30–50 records.
3. Transcribe the recording.
4. Examine how it handles artist and label names.

Example test:

```text
Surgeon, Magneze, Tresor, T-179, next
Jeff Mills, The Bells, Purpose Maker, PM-001, next
Underground Resistance, Jupiter Jazz, next
Basic Channel, Phylyps Trak, Basic Channel, next
```

This establishes whether `large-v3` is worthwhile on the machine.

## Stage 2 — local LLM extraction

Feed the transcript to Ollama.

Make sure it reliably produces:

```text
artist
title
label
catalogue_number
notes
```

as structured JSON.

Do not involve Discogs yet.

## Stage 3 — Discogs search

Take a manually created list of ~20 records.

Implement:

```text
record
→ Discogs search
→ candidate releases
→ display candidates
```

## Stage 4 — confirmation UI

Add:

```text
select candidate
```

and store the Discogs release ID.

## Stage 5 — database + export

Add SQLite and CSV export.

## Stage 6 — valuation

Only once identification is working should valuation be added.

This prevents the project from becoming tangled around Marketplace access.

---

# 17. Important performance target

The key metric is:

> **How many records can I process per hour while standing at the record boxes?**

Aim for roughly:

```text
10–20 seconds per record
```

for the capture phase.

A 500-record collection would then be a manageable few hours of physical cataloguing rather than a major manual data-entry project.

The Discogs confirmation phase can be done afterwards.

---

# 18. The desired end result

After processing a collection, the application should be able to produce something like:

```text
500 records

BULK
────
412 records

CHECK
─────
57 records

VALUABLE
────────
26 records

HIGH_VALUE
──────────
5 records

UNRESOLVED
──────────
0 records
```

Then sort the interesting records:

```text
HIGH_VALUE

£85   Underground Resistance – XXXXX
£62   Jeff Mills – XXXXX
£54   Surgeon – XXXXX
£47   Robert Hood – XXXXX
£43   Regis – XXXXX
```

Those five records get physically separated.

The other 495 can potentially be offered to a dealer as a collection.

That is the actual purpose of the application.

---

# 19. Why this architecture is preferable

The important separation is:

```text
LOCAL AI
    ↓
"What did Neil say?"
    ↓
"What record does that correspond to?"
    ↓
"What Discogs release is this?"
```

versus:

```text
Discogs
    ↓
"What is this record worth?"
```

The first three stages can be made completely local and controlled.

The final valuation stage depends on external marketplace information and is inherently less stable.

That means the project remains useful even if Discogs changes its Marketplace API rules.

---

# 20. Future possibilities

Once the basic tool works, several useful extensions become possible.

### OCR

Take a photo of the label and extract:

```text
label
catalogue number
artist
title
```

This could be especially useful for obscure records.

### Runout recognition

For genuinely valuable records, capture the runout/matrix inscription.

This may distinguish pressings that look identical.

### Local vision model

Use a local vision-capable LLM to help identify labels, artwork or printed information.

Ollama supports vision models and structured outputs, so this can remain local as well. citeturn0search0

### Duplicate detection

Detect multiple copies of the same release.

### Condition

Add:

```text
Media condition
Sleeve condition
```

Only when needed.

### Dealer estimate

Eventually create a second value:

```text
Discogs retail indication: £25

Likely bulk dealer value: £7–10
```

This is probably more useful for the eventual decision about whether to sell individually or as a collection.

---

# 21. Recommended starting stack

For the first version:

| Component | Choice |
|---|---|
| Language | Python |
| UI | Plain HTML + JavaScript |
| Web server | Flask |
| Speech-to-text | faster-whisper |
| Whisper model | large-v3 initially |
| Local LLM runner | Ollama |
| LLM output | JSON Schema |
| Database | SQLite |
| Discogs access | Official API |
| Export | CSV |
| Audio capture | Browser MediaRecorder |
| Network dependency | Discogs only |

This gives a genuinely local AI pipeline:

```text
MIC
 ↓
faster-whisper
 ↓
Ollama
 ↓
Discogs API
 ↓
SQLite
```

with no OpenAI API requirement.

---

# References

- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- Ollama: https://ollama.com/
- Ollama structured outputs: https://docs.ollama.com/capabilities/structured-outputs
- Discogs API Terms of Use: https://support.discogs.com/hc/en-us/articles/360009334593-API-Terms-of-Use
- Discogs Collection feature: https://support.discogs.com/hc/en-us/articles/360007331534-How-Does-The-Collection-Feature-Work

The Discogs API terms should be checked again before implementing automated Marketplace-price collection or turning the tool into anything public/commercial.
