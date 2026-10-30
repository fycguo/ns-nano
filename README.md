# Neurosynth Backend (ns-nano)

A Flask + PostgreSQL/PostGIS backend that answers **functional dissociation** queries over the
Neurosynth database: 14,371 studies, 507,891 activation peaks (MNI) and 3,228 abstract terms.
Built on the course template [spiketren/ns-nano](https://github.com/spiketren/ns-nano).

- **Live service:** `https://<your-app>.onrender.com`
- **Code:** [`app.py`](app.py) (the whole backend); data are loaded by [`create_db.py`](create_db.py)

---

## Endpoints

| Endpoint | Returns |
|---|---|
| `GET /dissociate/terms/<term_a>/<term_b>` | studies that mention **term A but not term B** |
| `GET /dissociate/locations/<x1_y1_z1>/<x2_y2_z2>` | studies reporting a peak **near A but none near B** |
| `GET /terms/<term>/studies` | studies that mention a term |
| `GET /locations/<x_y_z>/studies` | studies reporting a peak near `(x, y, z)` |
| `GET /terms?q=<text>` | the term vocabulary (optionally filtered) |
| `GET /`, `/img`, `/test_db` | index page and smoke tests |

**Definitions**

- A study *mentions a term* when the term's TF-IDF weight in its abstract exceeds `threshold`
  (default `0.001`, the Neurosynth/NiMARE default).
- A study *mentions a location* when it reports at least one peak within `r` mm of it
  (default `6`, Neurosynth's default; `r=0` means that exact coordinate).
- Terms are case-insensitive and `_` may replace spaces (`posterior_cingulate`).
  Coordinates are `x_y_z` in MNI millimetres (`0_-52_26`).

**Query parameters**

| Parameter | Default | Meaning |
|---|---|---|
| `threshold` | `0.001` | term weight cut-off (0–1) |
| `r` | `6` | search radius in mm (0–50) |
| `both` | off | `both=1` also returns the reverse direction (B but not A) |
| `limit`, `offset` | all, `0` | page through the study list (counts always cover the full set) |
| `callback` | – | wrap the JSON for JSONP (`callback=fn` → `fn({...});`) |

**Response** (`/dissociate/locations/0_-52_26/-2_50_-6?limit=1`)

```json
{
  "dissociation": "locations",
  "a": [0, -52, 26],
  "b": [-2, 50, -6],
  "params": {"radius_mm": 6.0, "limit": 1, "offset": 0},
  "counts": {"a": 479, "b": 497, "a_and_b": 73, "a_not_b": 406, "b_not_a": 424},
  "count": 406,
  "studies": [
    {"study_id": "24027520",
     "title": "Cortical Midline Structures and Autobiographical-Self Processes: ...",
     "authors": "Araujo HF, Kaplan J, Damasio A", "journal": "Frontiers in human neuroscience",
     "year": 2013, "distance_mm": 0.0}
  ]
}
```

Term queries return `weight` (TF-IDF, highest first) instead of `distance_mm` (nearest first).
Errors are JSON: `400` for a malformed coordinate or parameter, `404` for an unknown term
(with spelling `suggestions`) or endpoint, `503` if the database is unreachable.
Every response carries `Access-Control-Allow-Origin: *`, so a frontend on any origin can `fetch()` it.

**How the queries work**: each side (A or B) is a set of `(study_id, score)` rows. `A \ B` is
SQL `EXCEPT` and `A ∩ B` is `INTERSECT`. "Near (x, y, z)" is PostGIS
`ST_3DDWithin(geom, point, r)`, which PostgreSQL answers with the GIST index on `geom`.

---

## Quick start

1. **Provision PostgreSQL** (e.g., a free Render Postgres, same region as the web service).
2. **Verify the connection:** `python check_db.py --url "postgresql://<USER>:<PASSWORD>@<HOST>:5432/<DBNAME>"`
3. **Populate the database:** `python create_db.py --url "postgresql://<USER>:<PASSWORD>@<HOST>:5432/<DBNAME>"`
4. **Deploy `app.py`** as a Render Web Service:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - Environment variable: `DB_URL=<the database's Internal Database URL>`
5. **Smoke tests:** `https://<your-app>.onrender.com/img` and `/test_db`

Locally: `DB_URL=postgresql://... gunicorn app:app` (or `flask --app app run`).

## Notes

- `app.py` requests the psycopg2 driver explicitly (`postgresql+psycopg2://`). SQLAlchemy 2.1+
  maps a plain `postgresql://` URL to psycopg 3, which `requirements.txt` does not install.
- `create_db.py` strips column prefixes with `^terms_[^_]*__`, which cannot match the two-part
  prefix `terms_abstract_tfidf__`, so stored terms keep that prefix. `app.py` accepts both forms
  and always returns the clean term.
- The data are static, so the term vocabulary is read once per worker process.

## Requirements

Python 3.10+, PostgreSQL 13+ with PostGIS; Python packages in [`requirements.txt`](requirements.txt).

## License

MIT (see [LICENSE](LICENSE)); data and template from the Psychoinformatics & Neuroinformatics course.
