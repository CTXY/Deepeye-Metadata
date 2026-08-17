# Error Analysis Report: financial_codebase_latter50_context_graph

**Pipeline Run:** `python runner/run_stream_pipeline.py --start-from dataset --question-id-ranges 142-194 624-716`  
**Memory Augmentation:** ContextGraph (run via `python runner/run_memory_augmentation.py`)  
**Dataset:** `sub_dev_financial_and_codebase_community_latter50`  
**Output File:** `workspace/sql_selection/bird/financial_codebase_latter50_context_graph.stream_incorrect.json`  
**Log File:** `financial_codebase.log`

**Final Result:** 106/145 correct → **EA = 0.7310**  
**Total Incorrect:** 39 cases (25 codebase_community + 14 financial)

---

## Error Category Summary

| Category | Count (total w/ overlap) | % of Cases |
|---|---|---|
| Wrong JOIN Keys / Path | 17 | 44% |
| Operation Errors | 20 | 51% |
| Wrong SELECT Content | 11 | 28% |
| Memory / Schema Mapping Errors | 10 | 26% |

> Many cases involve 2+ overlapping error types; percentages sum to >100%.

**Primary error distribution (single most impactful error per case):**

| Category | Primary Cases | % |
|---|---|---|
| Wrong JOIN Keys / Path | 14 | 36% |
| Operation Errors | 13 | 33% |
| Memory / Schema Mapping Errors | 7 | 18% |
| Wrong SELECT Content | 5 | 13% |

---

## Category 1: Wrong JOIN Keys / Path (17 cases)

Wrong table join path or wrong intermediate table used to connect entities.

### The `postHistory` Bypass Problem (codebase_community) — 8 cases

**QIDs: 631, 632, 635, 639, 640, 630, 637, 652**

The single most pervasive pattern. In the codebase_community schema, `postHistory` is the canonical user-activity log. For any query involving "user's posts", "user's votes", or "user's tags", the correct join path is:

```
users → postHistory → [target_table]
```

The model consistently shortcuts to `users → posts` via `posts.OwnerUserId` or directly to `users → votes` via `votes.UserId` (which has only ~3,425 non-NULL entries out of 38,930 total rows — nearly always wrong).

| QID | Question | PRED Path | GOLD Path |
|---|---|---|---|
| 631 | How many posts created by Daniel Vassallo? | users → posts (OwnerUserId) | users → **postHistory** |
| 632 | How many votes made by Harlan? | users → votes (UserId) | users → **postHistory** → votes (PostId) |
| 635 | Posts by Matt Parker with >4 votes? | users → posts → votes | users → **postHistory** → posts → votes |
| 639 | % of Community's posts using R? | users → posts | users → **postHistory** → tags |
| 640 | View count diff: Mornington vs Amos | users → posts | users → **postHistory** → posts |
| 630 | Tags used by John Salvatier? | users → posts → tags | users → **postHistory** → posts |
| 637 | Tags by Mark Meckes with 0 comments? | users → posts (OwnerUserId) | users → **postHistory** → posts |
| 652 | Post IDs and badges of Samuel in 2013? | posts → users → badges | **postHistory** → badges |

**Root cause:** The model has learned to join users to their content via direct ownership links (`OwnerUserId`), but the gold standard uses `postHistory` as the mediator for user-activity tracing. The `postHistory` table captures the full history including edits and contributions, while `posts.OwnerUserId` only captures current ownership.

---

### Other Wrong JOIN Paths (9 cases)

**QID=679** — *Which post has the highest score?*
- PRED: `SELECT Id, Title FROM posts ORDER BY Score DESC LIMIT 1` (posts alone)
- GOLD: `users → posts` via `OwnerUserId` — gold requires at least one matching user, filtering out orphan posts

**QID=142** — *Accounts that placed household payment orders in Pisek?*
- PRED: `order → account → district` (SIPO from the `order` table)
- GOLD: `trans → account → district` (SIPO from the `trans` table)

**QID=169** — *Growth rate of loans for male clients 1996–1997?*
- PRED: `loan → disp → client` (bypasses `account`)
- GOLD: `loan → account → disp → client` + missing `disp.type = 'OWNER'` filter

**QID=173** — *How often does account 3 request statement release? Aim of 3539 debit?*
- PRED: uses `trans.k_symbol` for the debit aim
- GOLD: uses `order.k_symbol` (permanent orders table, not transactions); also misses the aggregate join subquery

**QID=180** — *Clients born 1983–1987 in East Bohemia with their IDs?*
- PRED: `client → district` only
- GOLD: `client → district → account` AND `client → disp → account` (double-verifies the account belongs to the client in that district)

**QID=186** — *% of male clients requesting weekly statements?*
- PRED: `client → disp → account`
- GOLD: `client → district → account → disp` (through district, ensuring geographic linkage)

**QID=193** — *ID and district for OWNER-type clients?*
- PRED: `disp → client → district` (client's home district)
- GOLD: `account → district → disp` (account's district, which is different)

**QID=694** — *Latest 10 comments to 'Analysing wind data with R' with user names?*
- PRED: `posts → comments → users` (gets comment author via `comments.UserId`) — semantically correct
- GOLD: `users → posts → comments` (gets post owner, ordered by `users.CreationDate`) — gold's logic is questionable here

**QID=693** — *Posts and comments by user with latest created account?*
- PRED: `latest_user LEFT JOIN posts LEFT JOIN comments` — returns separate counts
- GOLD: `users → posts → comments (ON comments.PostId = posts.Id)` — returns a single combined count

---

## Category 2: Operation Errors (20 cases)

Correct tables/columns but wrong SQL operator, aggregation, ordering method, or formula.

### 2.1 ORDER BY + LIMIT vs MAX Subquery (2 cases)

**QID=628** — *Which users have the highest number of views?*
- PRED: `ORDER BY Views DESC LIMIT 1` — returns only 1 user, breaks ties
- GOLD: `WHERE Views = (SELECT MAX(Views) FROM users)` — correctly returns ALL tied users

**QID=687** — *How many comments on the post with the highest score?*
- PRED: subquery `(SELECT Id FROM posts ORDER BY Score DESC LIMIT 1)` → count comments
- GOLD: `GROUP BY T1.Id ORDER BY SUM(T1.Score) DESC LIMIT 1` — aggregation-first approach

### 2.2 CASE WHEN vs IIF / Dialect Differences (5 cases)

**QID=639, 640, 171, 683, 169** — PRED uses `CASE WHEN ... THEN 1 ELSE 0 END`; GOLD uses SQLite's `IIF(condition, 1, 0)`. While logically equivalent in most databases, the evaluation semantics and type handling can differ in SQLite.

**QID=692** — Date calculation method:
- PRED: `julianday(badges.Date) - julianday(users.CreationDate)` (returns fractional days)
- GOLD: `T1.Date - T2.CreationDate` (direct date string subtraction, SQLite-specific)

### 2.3 Wrong Aggregation Scope (3 cases)

**QID=683** — *% of posts whose owners had reputation >1000 in 2011:*
- PRED: applies `WHERE year = '2011'` filter → denominator = only 2011 posts
- GOLD: computes `SUM(IIF(year='2011' AND rep>1000,...))` over ALL posts → denominator = all user-post pairs

**QID=145** — *Account IDs with below-average credit card transactions in 1998:*
- PRED subquery: `AVG(amount) WHERE operation = 'VYBER KARTOU' AND year = '1998'` (credit-card-only average)
- GOLD subquery: `AVG(amount) WHERE year = '1998'` (all 1998 transactions average)

**QID=189** — *Female clients who are oldest with lowest average salary:*
- PRED: complex nested subquery, wrong multi-condition optimization
- GOLD: `ORDER BY birth_date ASC, A11 ASC LIMIT 1` (elegant two-condition sort)

### 2.4 DISTINCT Usage Errors (4 cases)

| QID | PRED | GOLD | Impact |
|---|---|---|---|
| 672 | `COUNT(DISTINCT u.Id)` | `COUNT(T1.Id)` | PRED de-duplicates users; GOLD counts per-post records |
| 150 | `COUNT(DISTINCT a.account_id)` | `COUNT(T2.account_id)` | PRED de-duplicates; GOLD counts all transaction-linked accounts |
| 182 | `COUNT(DISTINCT c.client_id)` | `COUNT(T1.account_id)` | PRED counts distinct clients; GOLD counts account records |
| 711 | `COUNT(DISTINCT u.Id)` | `COUNT(DISTINCT T1.id)` (T1=comments) | PRED deduplicates on users; GOLD deduplicates on comments |

### 2.5 Wrong Subtraction Order (1 case)

**QID=171** — *Crime difference between East and North Bohemia in 1996:*
- PRED: `north - east`
- GOLD: `east - north`
- The subtraction direction is reversed, yielding the negation of the correct answer.

### 2.6 Date Column Mismatch in ORDER BY (1 case)

**QID=667** — *Title of the post with the oldest post link:*
- PRED: `ORDER BY postLinks.CreationDate ASC` (link creation date)
- GOLD: `ORDER BY posts.CreaionDate` (the post's own creation date; note the schema typo `CreaionDate`)

### 2.7 Wrong Aggregation Target (2 cases)

**QID=649** — *Post history count and last edit date:*
- PRED: returns `COUNT(postHistory.Id)` and `MAX(LastEditDate)` (aggregated scalars)
- GOLD: returns individual `postHistory.Id` and `LastEditDate` per record

**QID=693** — *Posts and comments by latest user:*
- PRED: returns two separate counts (`COUNT(posts)`, `COUNT(comments)`)
- GOLD: returns a single combined count via nested joins

**QID=182** — Wrong COUNT target (see 2.4)

---

## Category 3: Wrong SELECT Content (11 cases)

Correct query logic but wrong columns selected.

| QID | Question | PRED Selects | GOLD Selects | Error |
|---|---|---|---|---|
| 628 | Users with highest views? | `DisplayName` | `Id, DisplayName` | Missing `Id` |
| 686 | Posts with views above average? | `COUNT(*)` | `Id` | Count vs list of IDs |
| 649 | Post history for specific title? | `COUNT(Id), MAX(LastEditDate)` | `postHistory.Id, LastEditDate` | Aggregated vs per-record |
| 156 | Owner of largest loan account? | `client.gender` | `disp.client_id` | Gender vs client ID |
| 177 | Sum in client 4's account after txn 851? | `trans.amount` | `trans.balance` | Amount vs balance |
| 180 | Clients born 1983–1987 in East Bohemia? | `client_id` | `client_id, account_id` | Missing account_id |
| 193 | ID and district for OWNER clients? | `client_id, district_id` | `client_id, district_id, A2` | Missing district name A2 |
| 682 | Most valuable post in 2010? | `posts.Id, users.DisplayName` | `posts.OwnerUserId, users.DisplayName` | Post ID vs user ID |
| 630 | Tags used by John Salvatier? | `tags.TagName` | `posts.Tags` | Parsed names vs raw tag string |
| 652 | Post IDs and badges of Samuel 2013? | `posts.Id, badges.Name` | `postHistory.PostId, badges.Name` | posts.Id vs postHistory.PostId |
| 179 (indirect) | — | — | — | — |

**Notable patterns:**
- **Field precision**: `trans.amount` vs `trans.balance` (QID=177), `posts.Id` vs `posts.OwnerUserId` (QID=682) — correct table but wrong field within the table
- **Missing columns**: QIDs 628, 180, 193 select a subset of what gold requires
- **Aggregate vs list**: QIDs 649, 686 — model aggregates when gold wants individual records

---

## Category 4: Memory / Schema Mapping Errors (10 cases)

The Augmented Memory Graph provided user selections or schema context that led to wrong table/column interpretation.

### 4.1 `posts.Score` vs `comments.Score` Confusion (3 cases)

**QIDs: 709, 710, 646** — Both `posts` and `comments` have a `Score` column. When queries reference "score" in a joined posts-comments context, the model applies the condition to `comments.Score` while gold applies it to `posts.Score`.

| QID | Question | PRED Filter | GOLD Filter |
|---|---|---|---|
| 709 | Comments with 0 score, posts with viewCount <5? | `comments.Score = 0` AND `posts.ViewCount < 5` | `posts.Score = 0` AND `posts.ViewCount < 5` |
| 710 | Posts with 1 comment, comments with 0 score? | `posts.CommentCount = 1` AND `comments.Score = 0` | `posts.CommentCount = 1` AND `posts.Score = 0` |
| 646 | Post titles with positive comments? | `comments.Score > 60` | `posts.Score > 60` |

**Root cause:** When the memory context or schema linking retrieves both `posts.Score` and `comments.Score`, the model defaults to the table explicitly mentioned in the question (comments) rather than the post-level score that gold requires. This suggests the memory augmentation needs to explicitly disambiguate which Score field applies.

### 4.2 `tags.TagName` vs `posts.Tags` String (3 cases)

**QIDs: 696, 630, 639** — For tag-related queries, PRED uses `posts.Tags` (a delimited string like `<r><python>`) while GOLD uses the normalized `tags` table (`tags.TagName`, `tags.ExcerptPostId`).

| QID | PRED Approach | GOLD Approach |
|---|---|---|
| 696 | `posts WHERE Tags LIKE '%<careers>%'` | `tags WHERE TagName = 'careers'` |
| 630 | `tags t ON p.Tags LIKE '%<t.TagName>%'` | `posts.Tags` (raw string, no parse) |
| 639 | `posts.Tags LIKE '%<r>%'` | `tags WHERE TagName = 'r'` via `tags.ExcerptPostId` |

**Root cause:** The schema has two sources of truth for tags: the normalized `tags` table and the denormalized `posts.Tags` string. Gold uses different sources for different questions (normalized table for counting/filtering, raw string for "what tags" retrieval), which is inconsistent and hard to disambiguate without explicit memory guidance.

### 4.3 Wrong Table for Financial Operations (2 cases)

**QID=142** — SIPO (household payment):
- PRED uses `order.k_symbol = 'SIPO'` (permanent standing orders)
- GOLD uses `trans.k_symbol = 'SIPO'` (actual transactions executed)

**QID=173** — Debit aim:
- PRED uses `trans.k_symbol` to identify purpose of debit
- GOLD uses `order.k_symbol` (permanent order purpose), with `SUM(amount) = 3539` aggregation

The financial database has overlapping semantic coverage between `order` (permanent orders/mandates) and `trans` (actual transaction records). Without clear memory guidance on which table to use for each payment type, the model selects arbitrarily.

### 4.4 Year Value Error in Memory (1 case)

**QID=144** — *Average credit card transaction amount in 2021:*
- PRED filters `STRFTIME('%Y', date) = '2021'` (matches question literally)
- GOLD filters `STRFTIME('%Y', date) = '1998'` (the actual data range)

The question says "2021" but the financial database only contains data up to ~1998. The augmented memory should provide guidance that the database's actual date range is 1995–1999, allowing the model to infer that "2021" in the question is either a mistranslation or that the intended year is 1998.

### 4.5 Missing OWNER Filter in Financial `disp` (2 cases — overlaps with Category 1)

**QIDs: 156, 169** — The `disp` table has `type IN ('OWNER', 'DISPONENT')`. Without filtering `disp.type = 'OWNER'`, queries that should return the account owner may return co-holders. Memory augmentation should provide guidance that client ownership queries require this filter.

---

## Cross-Cutting Insights

### Insight 1: `postHistory` is the core architectural gap

The `postHistory` table serves as the canonical user-activity log in codebase_community. **8 out of 25 codebase_community errors (32%) stem from bypassing it.** The model consistently routes user→content queries through direct ownership links (`posts.OwnerUserId`) rather than the canonical activity path.

**Recommendation:** Add explicit memory guidance in the context graph: "For queries about a user's posts, votes, or tags, always route through `postHistory` as the intermediary table. `votes.UserId` is sparsely populated (< 9% of rows); do not use it for counting user votes."

### Insight 2: Several gold SQLs may contain annotation errors

During analysis, 4 cases showed gold SQLs with questionable logic:

| QID | Questionable Gold Pattern |
|---|---|
| 686 | Question asks "total number" (→ COUNT) but gold returns `Id` (list) |
| 694 | Gold orders by `users.CreationDate` for "latest 10 comments" (should be comment date) |
| 709/710 | Gold applies `Score = 0` to `posts` in a question clearly about comments |
| 682 | Gold filters on `users.CreationDate` for "most valuable post in 2010" (counter-intuitive) |

These cases should be reviewed for re-annotation before using them to measure accuracy.

### Insight 3: SQLite dialect gap

The model uses standard SQL idioms; gold uses SQLite-specific ones:
- `CASE WHEN ... THEN 1 ELSE 0 END` → `IIF(...)`
- `julianday(date1) - julianday(date2)` → `date1 - date2`
- `CAST(x AS FLOAT)` → `CAST(x AS REAL)`

This produces ~5 incorrect classifications that are logically equivalent. Memory augmentation could inject a dialect note: "Prefer SQLite-native functions: IIF over CASE WHEN, direct date subtraction over julianday, REAL over FLOAT."

### Insight 4: DISTINCT overuse is systematic

The model adds `DISTINCT` defensively when multiple joins could produce duplicates, causing 4 incorrect cases (150, 672, 182, 711). In 2 of these (150, 182), DISTINCT is incorrect per gold. In the other 2 (672, 711), the DISTINCT target is wrong (different column).

**Recommendation:** Be explicit about when DISTINCT is required. Memory/schema guidance should indicate when join conditions already guarantee uniqueness (e.g., primary key joins).

### Insight 5: Financial `disp.type = 'OWNER'` filter is consistently missing

2 financial cases miss this filter. The `disp` table links clients to accounts with a role type. Any query that asks about the account "owner" must include `disp.type = 'OWNER'`. This should be a permanent schema-level note in the memory graph for the financial database.

---

## Detailed Case Index

### codebase_community (25 incorrect)

| QID | Question Summary | Error Categories |
|---|---|---|
| 628 | Users with highest views | SELECT (missing Id) + Operation (LIMIT vs MAX subquery) |
| 630 | Tags used by John Salvatier | JOIN (bypass postHistory) + SELECT (TagName vs Tags) + Memory |
| 631 | Posts created by Daniel Vassallo | JOIN (bypass postHistory) |
| 632 | Votes made by Harlan | JOIN (bypass postHistory, sparse votes.UserId) |
| 635 | Posts by Matt Parker with >4 votes | JOIN (bypass postHistory) |
| 637 | Tags by Mark Meckes in uncommented posts | JOIN (bypass postHistory) + Operation (recursive CTE overengineering) |
| 639 | % of Community's R posts | JOIN (bypass postHistory) + Memory (posts.Tags vs tags.TagName) + Operation |
| 640 | View count diff: Mornington vs Amos | JOIN (bypass postHistory) + Operation (CASE vs IIF) |
| 646 | Post titles with positive comments | Memory (wrong Score: comments vs posts) |
| 649 | Post history count & last edit date | SELECT (COUNT vs individual IDs) |
| 652 | Post IDs and badges of Samuel 2013 | JOIN (bypass postHistory) + SELECT (posts.Id vs postHistory.PostId) |
| 667 | Oldest post link title | Operation (wrong ORDER BY column: link date vs post date) |
| 672 | UK users with ≥4 favorite posts | Operation (wrong DISTINCT) |
| 679 | Post with highest score | JOIN (posts alone vs users→posts) |
| 682 | Most valuable post in 2010 | SELECT (posts.Id vs OwnerUserId) + Memory (wrong date column) |
| 683 | % posts with owner rep >1000 in 2011 | Operation (wrong aggregation scope: WHERE vs IIF) |
| 686 | Posts with above-average views | SELECT (COUNT vs Id list) |
| 687 | Comments on highest-scored post | Operation (subquery vs GROUP BY approach) |
| 692 | Days for Zolomon to get badge | Operation (julianday vs direct subtraction) |
| 693 | Posts & comments by latest user | JOIN + SELECT (two counts vs one joined count) |
| 694 | Latest 10 comments on post title | JOIN (comment author path) + Operation (wrong ORDER BY) |
| 696 | Count posts with 'careers' tag | Memory (posts.Tags string vs tags table) |
| 709 | Comments with 0 score, posts viewCount <5 | Memory (wrong Score: comments vs posts) |
| 710 | Posts with 1 comment, comments with 0 score | Memory (wrong Score: comments vs posts) |
| 711 | Users aged 40 with 0-score comments | Operation (DISTINCT on wrong column: users vs comments) |

### financial (14 incorrect)

| QID | Question Summary | Error Categories |
|---|---|---|
| 142 | Accounts with SIPO orders in Pisek | JOIN (order vs trans for SIPO) |
| 144 | Avg credit card transaction in 2021 | Memory (wrong year: 2021 vs 1998) + JOIN (missing card chain) |
| 145 | Account IDs below-avg CC transactions in 1998 | Operation (wrong AVG scope in subquery) + JOIN (missing account join) |
| 150 | Accounts in North Bohemia, bank=AB | Operation (wrong DISTINCT) |
| 156 | Owner of largest loan account | SELECT (gender vs client_id) + Memory (missing OWNER filter) |
| 169 | Loan growth rate male clients 1996–1997 | Operation (wrong growth formula) + JOIN (missing OWNER filter) |
| 171 | Crime difference East vs North Bohemia 1996 | Operation (reversed subtraction order) |
| 173 | Account 3 statement frequency; debit aim of 3539 | Memory (trans vs order table) + JOIN |
| 177 | Sum in client 4's account after txn 851 | SELECT (amount vs balance) |
| 180 | Clients born 1983–1987 in East Bohemia | SELECT (missing account_id) + JOIN (missing disp→account path) |
| 182 | Male customers born 1974–1976 with SIPO >4000 | Operation (COUNT client vs COUNT account) |
| 186 | % male clients requesting weekly statements | JOIN (wrong path: via district vs direct) |
| 189 | Account IDs: oldest female, lowest avg salary | Operation (complex subquery vs ORDER BY multi-column) |
| 193 | ID and district for OWNER clients | SELECT (missing A2) + JOIN (client district vs account district) |

---

## Recommendations

### R1: Memory Graph — `postHistory` as mandatory intermediary
Add a schema-level note to the codebase_community context graph: "For all user-activity queries (posts by user, votes by user, tags by user), the join path MUST go through `postHistory`. Do NOT use `posts.OwnerUserId` as the primary user-post link. `votes.UserId` is sparsely populated (<9%) and should not be used for counting votes."

### R2: Memory Graph — Score field disambiguation
Add guidance: "In joined queries involving both `posts` and `comments`, `Score = 0` conditions typically refer to `posts.Score` unless the question explicitly says 'comment score'. Verify which table the score condition applies to."

### R3: Memory Graph — Tag normalization guidance
Add guidance: "Use the `tags` table (TagName, Count) for filtering/counting by tag. Use `posts.Tags` only when the question asks to retrieve the raw tag string. Do not LIKE-match against `posts.Tags` for tag counting — use `tags.TagName` instead."

### R4: Memory Graph — Financial `disp.type = 'OWNER'` rule
Add a persistent schema note: "When querying the financial database for an account's owner (client with ownership rights), always filter `disp.type = 'OWNER'`. Without this filter, co-holders (DISPONENT) are also returned."

### R5: Memory Graph — `order` vs `trans` for financial payments
Add guidance: "In the financial database: `trans` records actual executed transactions (use for counting/summing actual payments). `order` records standing/permanent orders (use for querying what recurring payments are set up). SIPO queries for 'placed orders' use `trans`; SIPO queries for 'payment mandate' use `order`."

### R6: Dialect normalization
Prompt or post-process to use SQLite-native functions: `IIF` over `CASE WHEN` for conditionals, direct date string subtraction over `julianday()`, `REAL` over `FLOAT` in casts.

### R7: Re-annotation review
Flag QIDs 686, 694, 709, 710, 682 for gold annotation review — the gold SQLs in these cases show questionable logic that may not accurately reflect the intended question semantics.

### R8: DISTINCT discipline
Add a heuristic: use DISTINCT only when the join is known to produce duplicates due to one-to-many relationships that are not filtered by the WHERE clause. Avoid adding DISTINCT defensively on COUNT columns.
