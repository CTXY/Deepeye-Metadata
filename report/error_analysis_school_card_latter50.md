# DeepEye-SQL Error Analysis Report
## Dataset: BIRD (california_schools + card_games), latter 50 — Context Graph Memory Augmentation

**Generated from:** `workspace/sql_selection/bird/school_and_card_latter50_context_graph_incorrect.json`  
**Total incorrect:** 32 / (evaluated range)  
**Databases:** `california_schools` (Q44–Q88) · `card_games` (Q435–Q530)

---

## 1. Error Category Overview

Each incorrect SQL was classified into one or more of four error categories:

| # | Category | Count | % of 32 cases |
|---|---|---|---|
| A | **Operation / Logic Errors** | 30 | 93.8% |
| B | **Wrong Output Columns / Fields Selected** | 18 | 56.3% |
| C | **Wrong Field / Semantic Matching** | 8 | 25.0% |
| D | **Wrong JOIN Keys / Path Connection** | 6 | 18.8% |

> Many questions carry multiple error categories; percentages sum to >100%.

---

## 2. Detailed Per-Question Analysis

---

### Q49 · california_schools
**Question:** "Which different county has the most number of closed schools? Please provide the name of each school as well as the closure date."

```sql
-- PRED
SELECT s."School", s."ClosedDate"
FROM "schools" AS s
WHERE s."StatusType" = 'Closed'
  AND s."County" = (SELECT "County" FROM "schools" WHERE "StatusType" = 'Closed'
                    GROUP BY "County" ORDER BY COUNT(*) DESC LIMIT 1)

-- GOLD
SELECT DISTINCT County, School, ClosedDate
FROM schools
WHERE County = (SELECT County FROM schools WHERE StatusType = 'Closed'
                GROUP BY County ORDER BY COUNT(School) DESC LIMIT 1)
  AND StatusType = 'Closed' AND school IS NOT NULL
```

**Errors:** [B] Wrong Output Columns · [A] Operation/Logic

| Issue | Detail |
|---|---|
| Missing `County` in SELECT | The question asks "which county" — it must appear in the output |
| Missing `DISTINCT` | Duplicate school rows may appear |
| Missing `school IS NOT NULL` filter | NULL school rows are included, polluting results |
| Inner subquery: `COUNT(*)` vs `COUNT(School)` | Gold excludes NULL school names from the ranking count |

---

### Q51 · california_schools
**Question:** "In which mailing street address can you find the school that has the lowest average score in reading?"

```sql
-- PRED
SELECT s.MailStreet, s.School FROM satscores sc
INNER JOIN schools s ON sc.cds = s.CDSCode
ORDER BY sc.AvgScrRead ASC LIMIT 1

-- GOLD
SELECT T2.MailStreet, T2.School FROM satscores AS T1
INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode
WHERE T1.AvgScrRead IS NOT NULL
ORDER BY T1.AvgScrRead ASC LIMIT 1
```

**Errors:** [A] Operation/Logic

| Issue | Detail |
|---|---|
| Missing `AvgScrRead IS NOT NULL` | In SQLite, NULL sorts before any value in ASC order — the query returns a school with no reading score instead of the actual minimum |

---

### Q53 · california_schools
**Question:** "How many test takers are there at the school/s whose mailing city address is in Fresno?"

```sql
-- PRED
SELECT SUM(satscores.NumTstTakr) AS total_test_takers
FROM schools INNER JOIN satscores ON schools.CDSCode = satscores.cds
WHERE schools.MailCity = 'Fresno'

-- GOLD
SELECT T1.NumTstTakr FROM satscores AS T1
INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode
WHERE T2.MailCity = 'Fresno'
```

**Errors:** [A] Operation/Logic

| Issue | Detail |
|---|---|
| Unnecessary `SUM()` aggregation | Gold returns per-school row values; the question's "how many" refers to each school's count, not a grand total |

---

### Q56 · california_schools
**Question:** "Of all the schools with a mailing state address in California, how many are active in San Joaquin city?"

```sql
-- PRED
SELECT COUNT("schools"."cdscode") FROM "schools"
WHERE "schools"."mailstate" = 'CA' AND "schools"."statustype" = 'Active'
  AND "schools"."mailcity" = 'San Joaquin'

-- GOLD
SELECT COUNT(CDSCode) FROM schools
WHERE City = 'San Joaquin' AND MailState = 'CA' AND StatusType = 'Active'
```

**Errors:** [C] Wrong Field / Semantic Matching

| Issue | Detail |
|---|---|
| `MailCity` used instead of `City` | The model over-anchored on "mailing" from the first clause. "Active in San Joaquin city" refers to the school's physical `City`, not its mailing address city |

---

### Q65 · california_schools
**Question:** "What is the ratio in percentage of Santa Clara County schools that are locally funded compared to all other types of charter school funding?"

```sql
-- PRED
SELECT CAST(SUM(CASE WHEN FundingType = 'Locally funded' THEN 1 ELSE 0 END) AS FLOAT) * 100
       / COUNT(*) ...

-- GOLD
SELECT CAST(SUM(CASE WHEN FundingType = 'Locally funded' THEN 1 ELSE 0 END) AS REAL) * 100
       / SUM(CASE WHEN FundingType != 'Locally funded' THEN 1 ELSE 0 END) ...
```

**Errors:** [A] Operation/Logic

| Issue | Detail |
|---|---|
| Wrong denominator | PRED: divides by `COUNT(*)` (total schools) → percentage of all charter schools. GOLD: divides by count of non-locally-funded schools → ratio "compared to all other types". The keyword "compared to" demands the latter |

---

### Q72 · california_schools
**Question:** "How many students from the ages of 5 to 17 are enrolled at the State Special School school in Fremont for the 2014-2015 academic year?"

```sql
-- PRED
SELECT SUM("frpm"."Enrollment (Ages 5-17)") ... WHERE "frpm"."Academic Year" = '2014-2015' ...

-- GOLD
SELECT T1.`Enrollment (Ages 5-17)` ... WHERE T1.`Academic Year` BETWEEN 2014 AND 2015
```

**Errors:** [A] Operation/Logic

| Issue | Detail |
|---|---|
| Unnecessary `SUM()` | Gold returns the raw per-row enrollment value for a specific school |
| Academic year filter type mismatch | PRED uses string `'2014-2015'`; GOLD uses integer `BETWEEN 2014 AND 2015` — different storage types may yield zero results |

---

### Q75 · california_schools
**Question:** "What is the educational level name for the schools with Breakfast Provision 2 in county code 37? Indicate the name of the school."

```sql
-- PRED
SELECT "schools"."school" FROM "frpm" INNER JOIN "schools" ...
WHERE "frpm"."NSLP Provision Status" = 'Breakfast Provision 2'
  AND "frpm"."County Code" = '37'

-- GOLD
SELECT T2.EILName, T2.School FROM frpm AS T1 INNER JOIN schools AS T2 ...
WHERE T1.`NSLP Provision Status` = 'Breakfast Provision 2' AND T1.`County Code` = 37
```

**Errors:** [B] Wrong Output Columns · [A] Operation/Logic

| Issue | Detail |
|---|---|
| Missing `EILName` column | The question explicitly asks for "educational level name" — `EILName` is the required column |
| County Code type mismatch | String `'37'` vs integer `37`; may return no rows if the column stores integers |

---

### Q83 · california_schools
**Question:** "Of the schools that offers a magnet program serving a grade span of Kindergarten to 8th grade, how many offers Multiple Provision Types? List the number of cities … indicate how many schools …"

```sql
-- PRED (highly over-complex)
SELECT (subquery count), COUNT(DISTINCT CASE ...), s."City", COUNT(CASE ...) ... WHERE s."GSserved" = 'K-8'

-- GOLD
SELECT T2.City, COUNT(T2.CDSCode) FROM frpm AS T1 INNER JOIN schools AS T2 ...
WHERE T2.Magnet = 1 AND T2.GSoffered = 'K-8' AND T1.`NSLP Provision Status` = 'Multiple Provision Types'
GROUP BY T2.City
```

**Errors:** [C] Wrong Field/Semantic · [B] Wrong Output Columns · [A] Operation/Logic · [D] Wrong JOIN

| Issue | Detail |
|---|---|
| `GSserved` vs `GSoffered` | Question says "offers a grade span" → `GSoffered` is the correct column |
| Misplaced filters | `Magnet = 1` and `Multiple Provision Types` are removed from the outer WHERE, breaking per-city counts |
| LEFT JOIN instead of INNER JOIN | Includes schools with no FRPM records |
| Over-complex output | Returns 4 columns (including a correlated subquery) vs gold's simple City + count |

---

### Q84 · california_schools
**Question:** "What are the two most common first names among the school administrators? Indicate the district to which they administer."

```sql
-- PRED: UNION ALL of AdmFName1, AdmFName2, AdmFName3
-- GOLD: Only considers AdmFName1
SELECT DISTINCT T1.AdmFName1, T1.District FROM schools AS T1
INNER JOIN (SELECT admfname1 FROM schools GROUP BY admfname1 ORDER BY COUNT(admfname1) DESC LIMIT 2) AS T2
ON T1.AdmFName1 = T2.admfname1
```

**Errors:** [C] Wrong Field/Semantic · [A] Operation/Logic

| Issue | Detail |
|---|---|
| All 3 name fields pooled | PRED treats AdmFName1/2/3 as equivalent; gold only ranks by AdmFName1 (primary administrator). Pooling distorts frequency counts |

---

### Q85 · california_schools
**Question:** "What is the Percent (%) Eligible Free (K-12)… List the district code of the school."

```sql
-- PRED
SELECT "frpm"."Percent (%) Eligible Free (K-12)", "frpm"."District Code" ...

-- GOLD
SELECT T1.`Free Meal Count (K-12)` * 100 / T1.`Enrollment (K-12)`, T1.`District Code` ...
```

**Errors:** [C] Wrong Field/Semantic

| Issue | Detail |
|---|---|
| Pre-computed column vs derived formula | PRED selects the stored `Percent` column; GOLD computes it as `Count*100/Enrollment`. The derived computation is the expected output and may differ from the stored (potentially rounded/stale) value |

---

### Q437 · card_games
**Question:** "Among black card borders, which card has full artwork?"

```sql
-- PRED: SELECT name FROM cards WHERE borderColor = 'black' AND isFullArt = 1
-- GOLD: SELECT id   FROM cards WHERE borderColor = 'black' AND isFullArt = 1
```

**Errors:** [B] Wrong Output Columns

| Issue | Detail |
|---|---|
| `name` vs `id` | Gold expects `id` as the identifier; PRED returns `name` — intuitively more useful but wrong per gold standard |

---

### Q442 · card_games
**Question:** "Mention the base set size and set code of the set that was in block named 'Masques' and 'Mirage'."

```sql
-- PRED: SELECT baseSetSize, code FROM sets WHERE block IN ('Masques', 'Mirage')
-- GOLD: SELECT DISTINCT T1.baseSetSize, T2.setCode FROM sets AS T1
--       INNER JOIN set_translations AS T2 ON T2.setCode = T1.code WHERE T1.block IN (...)
```

**Errors:** [D] Wrong JOIN · [B] Wrong Output Columns

| Issue | Detail |
|---|---|
| Missing join to `set_translations` | Gold requires joining `set_translations` and returning `set_translations.setCode` specifically |
| Missing `DISTINCT` | Join fan-out may produce duplicate rows |

---

### Q444 / Q448 · card_games
**Questions:** "Name the foreign name of the card that has boros/abzan watermark? List out the type of this card."

```sql
-- PRED: SELECT foreign_data.name, cards.type ...
-- GOLD: SELECT DISTINCT cards.name, cards.type ...   (both name and type from cards table)
```

**Errors:** [B] Wrong Output Columns · [A] Operation/Logic (repeated pattern)

| Issue | Detail |
|---|---|
| `foreign_data.name` vs `cards.name` | Question says "foreign name" → intuitively `foreign_data.name`. Gold uses `cards.name`. This is a recurrent source of confusion across Q444, Q448, Q482 |
| Missing `DISTINCT` | JOIN with foreign_data fans out rows per language |

---

### Q445 · card_games
**Question:** "What is the language and flavor text of the card that has colorpie watermark? List out the type of this card."

```sql
-- PRED: SELECT fd.language, fd.flavortext, c.type ...
-- GOLD: SELECT DISTINCT T2.language, T2.flavorText ...  (no type column)
```

**Errors:** [B] Wrong Output Columns · [A] Operation/Logic

| Issue | Detail |
|---|---|
| Extra `type` column | Gold omits `type` despite the question mentioning it — predicted includes it |
| Missing `DISTINCT` | Fan-out from JOIN |

---

### Q446 · card_games
**Question:** "What is percentage of the cards with a converted Mana Cost of 10 in set of Abyssal Horror?"

```sql
-- PRED: counts all cards in the SET that contains 'Abyssal Horror', computes % with CMC=10
-- GOLD: filters WHERE name = 'Abyssal Horror' (single card scope), outputs name as 2nd column
```

**Errors:** [C] Wrong Field/Semantic · [B] Wrong Output Columns · [D] Wrong JOIN

| Issue | Detail |
|---|---|
| Scope misinterpretation | PRED: "in set of" → all cards in that set. GOLD: computed over the single 'Abyssal Horror' card record |
| Missing `name` output column | Gold outputs `name` as a second column |
| Missing `sets` JOIN | Gold joins through `sets` table |

---

### Q454 · card_games
**Question:** "Among the cards with a white border color, how many of them have unknown power?"

```sql
-- PRED: WHERE power = '*' OR power IS NULL
-- GOLD: WHERE power LIKE '%*%' OR power IS NULL
```

**Errors:** [A] Operation/Logic

| Issue | Detail |
|---|---|
| Exact match vs substring match | MTG power values like `'1+*'`, `'2*'`, `'*+1'` are unknown/variable. `= '*'` misses these; `LIKE '%*%'` correctly catches all variants |

---

### Q458 · card_games
**Question:** "How many artists have designed a card with a black border color and is available in both 'arena' and 'mtgo' printing type?"

```sql
-- PRED: COUNT(DISTINCT artist) WHERE ... LIKE '%arena%' AND LIKE '%mtgo%'
-- GOLD: COUNT(CASE WHEN availability LIKE '%arena,mtgo%' ... ) -- counts rows, not artists
```

**Errors:** [A] Operation/Logic

| Issue | Detail |
|---|---|
| Counts distinct artists vs card rows | Question asks "how many artists" — PRED is semantically correct, but gold counts card records |
| Separate LIKE conditions vs contiguous pattern | PRED matches any order of arena/mtgo in the string; gold requires `arena,mtgo` as adjacent substring in canonical order |

---

### Q465 · card_games
**Question:** "For the set of cards with 'Ancestor's Chosen' in it, is there a Korean version of it?"

```sql
-- PRED: JOIN foreign_data ON uuid; checks language = 'Korean' (card-level)
-- GOLD: JOIN set_translations ON setCode; checks language = 'Korean' AND translation IS NOT NULL (set-level)
```

**Errors:** [D] Wrong JOIN · [A] Operation/Logic

| Issue | Detail |
|---|---|
| `foreign_data` vs `set_translations` | Question asks about a "set" having a Korean version → set-level translation table is correct. PRED joins card-level foreign data |
| Wrong output format | Returns `1/0` instead of `'YES'/'NO'` |
| Missing `translation IS NOT NULL` | |

---

### Q469 · card_games
**Question:** "Did the set of cards with 'Angel of Mercy' appear on Magic: The Gathering Online?"

```sql
-- PRED: SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END ...
--       (just checks if the card exists, always returns 1)
-- GOLD: SELECT IIF(T2.mtgoCode IS NOT NULL, 'YES', 'NO') ...
```

**Errors:** [A] Operation/Logic · [B] Wrong Output Columns

| Issue | Detail |
|---|---|
| Meaningless existence check | PRED checks `COUNT(*) > 0` — always true if the card exists. Gold checks `mtgoCode IS NOT NULL` which is the actual MTGO availability flag in `sets` |
| Returns `0/1` instead of `'YES'/'NO'` | |

---

### Q473 · card_games
**Question:** "Is the set of cards with Adarkar Valkyrie only available outside the United States?"

```sql
-- PRED: SELECT "sets"."isForeignOnly" ... (returns raw 0/1)
-- GOLD: SELECT IIF(isForeignOnly = 1, 'YES', 'NO') ...
```

**Errors:** [A] Operation/Logic

| Issue | Detail |
|---|---|
| Missing boolean-to-string transformation | PRED returns raw integer; gold converts to `'YES'/'NO'` for a yes/no question |

---

### Q482 · card_games
**Question:** "What's the German type of the card 'Ancestor's Chosen'?"

```sql
-- PRED: SELECT foreign_data.type ... WHERE language = 'German'
-- GOLD: SELECT DISTINCT cards.type ... WHERE language = 'German'
```

**Errors:** [B] Wrong Output Columns · [A] Operation/Logic

| Issue | Detail |
|---|---|
| `foreign_data.type` vs `cards.type` | Question asks for "German type" (suggesting `foreign_data.type`) but gold selects from `cards.type` |
| Missing `DISTINCT` | |

---

### Q484 · card_games
**Question:** "Please list the Italian names of the cards in the set Coldsnap with the highest converted mana cost."

```sql
-- PRED: SELECT fd.name ... LIMIT 1   (returns only 1 result)
-- GOLD: SELECT cards.name ... (no LIMIT — returns ALL tied top-cost cards)
```

**Errors:** [A] Operation/Logic · [B] Wrong Output Columns

| Issue | Detail |
|---|---|
| `LIMIT 1` truncates ties | "Cards" is plural — multiple may share the top CMC. Gold has no LIMIT |
| `foreign_data.name` vs `cards.name` | Question asks for Italian names (semantic: `foreign_data.name`) but gold returns `cards.name` |

---

### Q494 · card_games
**Question:** "For all cards illustrated by Jim Pavelec… Do these cards have missing or degraded properties?"

```sql
-- PRED: SELECT id, artist, text, hasContentWarning (raw int)
-- GOLD: SELECT text, CASE WHEN hasContentWarning=1 THEN 'YES' ELSE 'NO' END
```

**Errors:** [B] Wrong Output Columns · [A] Operation/Logic

| Issue | Detail |
|---|---|
| Extra `id` and `artist` columns | Not requested in the output |
| Missing boolean transformation | PRED returns raw `0/1`; gold converts to `'YES'/'NO'` |

---

### Q499 · card_games
**Question:** "How many translations of the name of the set 'Tenth Edition'?"

```sql
-- PRED: COUNT(*)
-- GOLD: COUNT(DISTINCT T2.translation) ... AND T2.translation IS NOT NULL
```

**Errors:** [A] Operation/Logic

| Issue | Detail |
|---|---|
| `COUNT(*)` vs `COUNT(DISTINCT translation)` | PRED counts all rows (including NULLs and duplicates); gold counts distinct non-null translations only |

---

### Q500 · card_games
**Question:** "Tell the Japanese name of the set which card 'Fellwar Stone' is in it."

```sql
-- PRED: cards → sets → set_translations (3-table chain)
-- GOLD: cards → set_translations directly (2-table, using setCode FK)
```

**Errors:** [D] Wrong JOIN · [A] Operation/Logic

| Issue | Detail |
|---|---|
| Unnecessary intermediate join | PRED routes through `sets`; if a card's `setCode` has no `sets` entry, the INNER JOIN drops the row. Gold joins directly and is more robust |
| Missing `translation IS NOT NULL` filter | |

---

### Q514 · card_games
**Question:** "In duels, what are the top 10 cards with the highest unconverted mana cost?"

```sql
-- PRED: JOIN without DISTINCT → fan-out from legalities
-- GOLD: DISTINCT + subquery approach
```

**Errors:** [A] Operation/Logic

| Issue | Detail |
|---|---|
| Missing `DISTINCT` on JOIN | Cards with multiple legality entries appear multiple times, corrupting the top-10 ranking |

---

### Q515 · card_games
**Question:** "When was the oldest mythic card released and what are its legal play formats?"

```sql
-- PRED: ORDER BY originalReleaseDate ASC LIMIT 1 (no NULL check)
-- GOLD: WHERE originalReleaseDate IS NOT NULL ORDER BY originalReleaseDate LIMIT 1
```

**Errors:** [A] Operation/Logic

| Issue | Detail |
|---|---|
| Missing `IS NOT NULL` on date | NULL sorts first in SQLite ASC — PRED returns a card with no release date as "oldest" |

---

### Q519 · card_games
**Question:** "What is the language of the 'Battlebond' set?"

```sql
-- PRED: sets JOIN set_translations ON code = setCode  (natural FK)
-- GOLD: set_translations WHERE id IN (SELECT id FROM sets ...)  (id-based match)
```

**Errors:** [D] Wrong JOIN

| Issue | Detail |
|---|---|
| Different join key | Gold matches `set_translations.id` to `sets.id` (same PK column); PRED uses the canonical FK `sets.code = set_translations.setCode`. These produce different results if the schema's `id` columns are auto-increment PKs that don't cross-match |

---

### Q520 · card_games
**Question:** "Who is the illustrator that illustrated the least amount of cards? List the format of play…"

```sql
-- PRED: returns all artists tied at the minimum count
-- GOLD: GROUP BY artist ORDER BY COUNT ASC LIMIT 1 (single result)
```

**Errors:** [A] Operation/Logic

| Issue | Detail |
|---|---|
| All tied artists returned | PRED correctly handles ties semantically but gold expects exactly one row via `LIMIT 1` |

---

### Q523 · card_games
**Question:** "What is the annual average number of sets released between 2012–2015? Indicate the common language."

```sql
-- PRED: Logically computes avg sets/year (counts per year, averages them)
-- GOLD: SUM(T1.id) / COUNT(T1.id) / 4  (averages the id values, not set counts)
```

**Errors:** [A] Operation/Logic · [C] Wrong Field/Semantic

| Issue | Detail |
|---|---|
| Different average formula | Gold uses `SUM(id)/COUNT(id)/4` (averaging id values — an unconventional formula); PRED computes a proper annual average. Results differ significantly |
| Join key difference | Gold: `sets.id = set_translations.id`; PRED: uses `setCode` FK |

---

### Q530 · card_games
**Question:** "List all the frame styles and cards Allen Williams worked on and find any banned cards if there are any."

```sql
-- PRED: SELECT frameVersion, name, status (raw)
-- GOLD: SELECT frameVersion, name, IIF(status='Banned', name, 'NO')
```

**Errors:** [A] Operation/Logic · [B] Wrong Output Columns

| Issue | Detail |
|---|---|
| Raw `status` vs conditional expression | PRED exposes raw status strings; gold applies `IIF(Banned, name, 'NO')` — returns card name if banned, else 'NO' |

---

## 3. Summary Table

| Q# | DB | [A] Op/Logic | [B] Output Cols | [C] Semantic | [D] JOIN | Primary Root Cause |
|---|---|:---:|:---:|:---:|:---:|---|
| Q49 | schools | ✓ | ✓ | | | Missing `County` in SELECT; missing NULL & DISTINCT |
| Q51 | schools | ✓ | | | | Missing `IS NOT NULL` on sort column |
| Q53 | schools | ✓ | | | | Wrong aggregation: `SUM` vs per-row |
| Q56 | schools | | | ✓ | | `MailCity` vs `City` — mailing vs physical city confusion |
| Q65 | schools | ✓ | | | | Wrong denominator: `COUNT(*)` vs non-locally-funded count |
| Q72 | schools | ✓ | | | | Wrong aggregation + academic year type mismatch |
| Q75 | schools | ✓ | ✓ | | | Missing `EILName`; county code type mismatch |
| Q83 | schools | ✓ | ✓ | ✓ | ✓ | `GSserved` vs `GSoffered`; filter misplacement; wrong JOIN type |
| Q84 | schools | ✓ | | ✓ | | All 3 AdmFName fields pooled vs only AdmFName1 |
| Q85 | schools | | | ✓ | | Pre-computed column vs `Count*100/Enrollment` formula |
| Q437 | cards | | ✓ | | | `name` vs `id` in SELECT |
| Q442 | cards | | ✓ | | ✓ | Missing join to `set_translations` |
| Q444 | cards | ✓ | ✓ | | | `foreign_data.name` vs `cards.name`; missing DISTINCT |
| Q445 | cards | ✓ | ✓ | | | Extra `type` column; missing DISTINCT |
| Q446 | cards | | ✓ | ✓ | ✓ | Scope: whole set vs single card; missing output column |
| Q448 | cards | ✓ | ✓ | | | `foreign_data.name` vs `cards.name`; missing DISTINCT |
| Q454 | cards | ✓ | | | | `= '*'` vs `LIKE '%*%'` for unknown power |
| Q458 | cards | ✓ | | | | Count artists vs count rows; separate vs contiguous LIKE |
| Q465 | cards | ✓ | | | ✓ | `foreign_data` (card-level) vs `set_translations` (set-level) |
| Q469 | cards | ✓ | ✓ | | | Meaningless existence check vs `mtgoCode IS NOT NULL` |
| Q473 | cards | ✓ | | | | Missing `IIF` boolean-to-string transformation |
| Q482 | cards | ✓ | ✓ | | | `foreign_data.type` vs `cards.type`; missing DISTINCT |
| Q484 | cards | ✓ | ✓ | | | `LIMIT 1` truncates ties; wrong table's `name` column |
| Q494 | cards | ✓ | ✓ | | | Extra columns; missing boolean transformation |
| Q499 | cards | ✓ | | | | `COUNT(*)` vs `COUNT(DISTINCT translation)` |
| Q500 | cards | ✓ | | | ✓ | Unnecessary intermediate `sets` join; missing NULL filter |
| Q514 | cards | ✓ | | | | Missing DISTINCT on JOIN causes ranking corruption |
| Q515 | cards | ✓ | | | | Missing `IS NOT NULL` on date sort column |
| Q519 | cards | | | | ✓ | Natural FK join vs gold's `id`-based subquery match |
| Q520 | cards | ✓ | | | | Returns all tied artists vs gold's `LIMIT 1` |
| Q523 | cards | ✓ | | ✓ | | Different average formula; different join key |
| Q530 | cards | ✓ | ✓ | | | Raw `status` vs `IIF(Banned, name, 'NO')` |

---

## 4. Systemic Error Patterns

### 4.1 NULL Handling is a Pervasive Blind Spot
**Affects:** Q51, Q75, Q499, Q500, Q515 (+ indirectly Q49)

The model consistently omits `IS NOT NULL` guards on columns used in:
- `ORDER BY ... ASC LIMIT 1` → NULL sorts before real values in SQLite, returning the wrong "minimum"
- `COUNT(*)` aggregations → NULL rows are counted as valid entries
- Translation queries → NULL translation rows should be excluded

**Fix direction:** When generating ORDER BY + LIMIT queries or COUNT queries, always check whether the ranked/counted column can contain NULLs and add appropriate filters.

---

### 4.2 YES/NO Boolean Output Format Systematically Missed
**Affects:** Q465, Q469, Q473, Q494, Q530

When a question asks a yes/no or boolean question ("is there a Korean version?", "did the set appear on MTGO?", "do these cards have content warnings?"), the gold standard consistently uses `IIF(..., 'YES', 'NO')` or `CASE WHEN ... THEN 'YES' ELSE 'NO' END`. The model returns raw `0/1` integers or raw status strings.

**Fix direction:** Detect yes/no question intent and apply the appropriate boolean-to-string transformation pattern.

---

### 4.3 Aggregation Errors — SUM/COUNT Where Individual Rows Expected
**Affects:** Q53, Q72, Q83

The model applies `SUM()` or `COUNT(*)` aggregation in scenarios where the question expects individual per-entity values. "How many test takers at each Fresno school" ≠ "total across all Fresno schools."

**Fix direction:** Distinguish "total across all" (aggregate) from "how many at each / per school" (individual row). The presence of "school/s" (plural) in Q53 is a cue that multiple values may be expected, not a single sum.

---

### 4.4 DISTINCT Underused After JOIN Fan-out
**Affects:** Q49, Q444, Q445, Q448, Q482, Q499, Q514 (7 cases)

JOIN operations that produce multiple rows per base entity (e.g., `cards JOIN foreign_data` produces one row per language per card) are not deduplicated with `DISTINCT`, returning redundant rows that corrupt results — especially damaging when combined with `LIMIT`.

**Fix direction:** After any one-to-many JOIN, evaluate whether the SELECT columns require deduplication. Default to `SELECT DISTINCT` when the output should be unique entities.

---

### 4.5 `foreign_data` vs `cards` Column Selection Confusion
**Affects:** Q444, Q448, Q482, Q484 (4 cases — recurring pattern)

Both `cards` and `foreign_data` have `name` and `type` columns. When questions ask about "foreign name" or "German type," the model selects from `foreign_data` (semantically reasonable) but the gold standard uses `cards` (English canonical values). This is a schema-level ambiguity in the gold standard, but it causes consistent failures.

**Fix direction:** For questions in card_games involving "foreign" or localized attributes while also asking for the card's "type," prefer selecting from the `cards` table for `type`. Only use `foreign_data` for purely foreign-language-specific fields like `flavorText` or `translation`.

---

### 4.6 `set_translations` Join Path Confusion
**Affects:** Q442, Q465, Q500, Q519 (4 cases)

The model struggles with whether to join via `setCode` (FK) or `id` (PK matching), and whether to use `foreign_data` (card-level) vs `set_translations` (set-level) for language queries. The gold standard uses inconsistent join keys across different questions (`setCode` in some, `id` in others), making this a particularly difficult pattern to learn.

**Fix direction:** For set-level language/translation queries, prefer `set_translations` with `setCode` FK. Reserve `foreign_data` for card-level foreign text (flavor text, card names in other languages). When the gold uses `id`-based joins (Q519, Q523), this may reflect gold standard idiosyncrasies.

---

### 4.7 Wrong Denominator / Formula in Ratio Calculations
**Affects:** Q65, Q85, Q523

In ratio/percentage calculations, the model selects the wrong denominator or uses a pre-stored column instead of computing the intended formula:
- Q65: divides by all schools instead of non-locally-funded schools
- Q85: uses stored `Percent` column instead of `Count*100/Enrollment`
- Q523: gold uses non-intuitive `SUM(id)/COUNT(id)/4` formula

**Fix direction:** Parse ratio questions carefully — "X compared to Y" means `X/Y`, not `X/(X+Y)`. When a computed column exists in the schema, prefer the explicit formula if the question implies a calculation.

---

### 4.8 Physical City vs Mailing City Confusion (california_schools)
**Affects:** Q56

The `california_schools` schema has both `City` (physical location) and `MailCity` (mailing address). When a question mentions a mailing address in one clause and a city name in another clause, the model incorrectly applies "mailing" to both.

**Fix direction:** Apply contextual disambiguation: "active in [City]" refers to physical presence, not mailing address, even when the broader question mentions mailing state.

---

## 5. Error Count by Database

| Database | Total Incorrect | [A] Op/Logic | [B] Output Cols | [C] Semantic | [D] JOIN |
|---|---|---|---|---|---|
| california_schools | 10 | 7 | 4 | 4 | 1 |
| card_games | 22 | 23 | 14 | 4 | 5 |
| **Total** | **32** | **30** | **18** | **8** | **6** |

> card_games has more total errors despite similar question count, suggesting the multi-table schema (cards, foreign_data, set_translations, legalities, rulings) is significantly harder to navigate correctly.

---

## 6. Priority Fixes

Based on frequency and impact:

1. **NULL guards on ORDER BY / COUNT** — Simple rule: add `IS NOT NULL` when ordering to find min/max, and use `COUNT(DISTINCT col)` with NULL exclusion when counting distinct entities. *(Affects 5–6 cases)*

2. **Boolean-to-string output format** — Detect yes/no question intent; apply `IIF(cond, 'YES', 'NO')` pattern. *(Affects 5 cases)*

3. **DISTINCT after JOIN** — Default to DISTINCT when joining one-to-many relationships and the question expects unique entity-level results. *(Affects 7 cases)*

4. **Aggregation vs row-level output** — Clarify whether "how many X at Y" means aggregate or per-Y listing. Context cues: plural school references, "each", "per" → row-level. *(Affects 3 cases)*

5. **Wrong column from right table (cards vs foreign_data)** — Establish a schema-level rule: `type` always comes from `cards`, not `foreign_data`. *(Affects 4 cases)*

---

*Report generated based on 32 incorrect SQL pairs from `school_and_card_latter50_context_graph_incorrect.json`*
