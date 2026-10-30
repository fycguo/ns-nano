# app.py
"""Neurosynth backend (HW08): functional-dissociation API on Flask + PostgreSQL/PostGIS.

The database is built by the course's create_db.py (schema `ns`):
  ns.annotations_terms(study_id, contrast_id, term, weight)  TF-IDF of each abstract term
  ns.coordinates(study_id, geom)                            activation peaks, POINTZ in MNI mm
  ns.metadata(study_id, title, authors, journal, year, ...)

A study mentions a term when its weight exceeds `threshold` (default 0.001, the
Neurosynth/NiMARE default) and mentions a location when it reports a peak within `r` mm
of it (default 6 mm, Neurosynth's default; r=0 means that exact coordinate).
All set logic runs in PostgreSQL: A \\ B with EXCEPT, A ∩ B with INTERSECT, and
"near (x, y, z)" with PostGIS ST_3DDWithin.

Query parameters: threshold, r, both=1 (add B \\ A to the response), limit, offset,
callback (JSONP).
"""
from difflib import get_close_matches
from functools import lru_cache
import math
import os
import re

from flask import Flask, jsonify, request, send_file
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

SCHEMA = "ns"
SRID = 4326                              # create_db.py's default --srid
TERM_PREFIX = "terms_abstract_tfidf__"   # see term_vocabulary()
DEFAULT_THRESHOLD = 0.001
DEFAULT_RADIUS_MM = 6.0
MAX_RADIUS_MM = 50.0
EMPTY_SET = "SELECT NULL::text AS study_id, NULL::float8 AS score WHERE FALSE"
JSONP_CALLBACK = re.compile(r"^[A-Za-z_$][\w$]*(\.[A-Za-z_$][\w$]*)*$")

INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>ns-nano: Neurosynth backend</title>
<meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="font-family: system-ui, sans-serif; max-width: 54rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.6">
<h1>Neurosynth backend</h1>
<p>Server working! JSON API for functional dissociations over the Neurosynth database
(Flask + PostgreSQL/PostGIS).</p>
<h2>Functional dissociation (A but not B)</h2>
<ul>
<li><a href="/dissociate/terms/posterior%20cingulate/ventromedial%20prefrontal">/dissociate/terms/posterior cingulate/ventromedial prefrontal</a></li>
<li><a href="/dissociate/terms/ventromedial%20prefrontal/posterior%20cingulate">/dissociate/terms/ventromedial prefrontal/posterior cingulate</a></li>
<li><a href="/dissociate/locations/0_-52_26/-2_50_-6">/dissociate/locations/0_-52_26/-2_50_-6</a></li>
<li><a href="/dissociate/locations/-2_50_-6/0_-52_26">/dissociate/locations/-2_50_-6/0_-52_26</a></li>
</ul>
<h2>Studies</h2>
<ul>
<li><a href="/terms/amygdala/studies">/terms/amygdala/studies</a></li>
<li><a href="/locations/-22_0_-20/studies">/locations/-22_0_-20/studies</a></li>
<li><a href="/terms?q=cingulate">/terms?q=cingulate</a> (vocabulary)</li>
</ul>
<h2>Options</h2>
<p><code>?threshold=0.001</code> term weight cut-off &middot; <code>?r=6</code> radius in mm (0 = exact)
&middot; <code>?both=1</code> also return B but not A &middot; <code>?limit=&amp;offset=</code> paging
&middot; <code>?callback=fn</code> JSONP</p>
<p>Smoke tests: <a href="/img">/img</a> &middot; <a href="/test_db">/test_db</a></p>
</body></html>"""

_engine = None

def get_engine():
    global _engine
    if _engine is not None:
        return _engine
    db_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Missing DB_URL (or DATABASE_URL) environment variable.")
    # Normalize old 'postgres://' scheme to 'postgresql://'
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://"):]
    # Name the driver: SQLAlchemy >= 2.1 maps plain 'postgresql://' to psycopg 3,
    # but requirements.txt installs psycopg2.
    if db_url.startswith("postgresql://"):
        db_url = "postgresql+psycopg2://" + db_url[len("postgresql://"):]
    _engine = create_engine(
        db_url,
        pool_pre_ping=True,
    )
    return _engine


class ApiError(Exception):
    """A client error, returned as JSON {"error": message, **extra} with an HTTP status."""

    def __init__(self, status, message, **extra):
        super().__init__(message)
        self.status = status
        self.payload = {"error": message, **extra}


# -----------------------------
# Request parsing
# -----------------------------
def float_arg(name, default, lo, hi):
    raw = request.args.get(name, "")
    if raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ApiError(400, f"query parameter '{name}' must be a number, got {raw!r}")
    if not lo <= value <= hi:   # also rejects nan
        raise ApiError(400, f"query parameter '{name}' must be between {lo} and {hi}, got {raw!r}")
    return value


def page_args():
    """(limit, offset) for the study list; limit=None returns every study."""
    values = []
    for name in ("limit", "offset"):
        raw = request.args.get(name, "")
        if raw == "":
            values.append(None if name == "limit" else 0)
        elif raw.isdigit():
            values.append(int(raw))
        else:
            raise ApiError(400, f"query parameter '{name}' must be a non-negative integer, got {raw!r}")
    return tuple(values)


def flag_arg(name):
    return request.args.get(name, "").lower() in {"1", "true", "yes", "on"}


def normalize_term(raw):
    """URL form -> vocabulary form: case-insensitive; '_' and runs of spaces mean one space."""
    return " ".join(raw.replace("_", " ").lower().split())


@lru_cache(maxsize=1)
def term_vocabulary():
    """{term: term as stored}. The data are static, so this is read once per process.

    create_db.py strips column prefixes with ^terms_[^_]*__, which cannot match the two-part
    prefix "terms_abstract_tfidf__", so stored terms keep it; both forms are accepted.
    """
    with get_engine().connect() as conn:
        stored = conn.execute(text(f"SELECT DISTINCT term FROM {SCHEMA}.annotations_terms")).scalars().all()
    return {(s[len(TERM_PREFIX):] if s.startswith(TERM_PREFIX) else s): s for s in stored}


def resolve_term(raw):
    """(clean term, stored term), or 404 with spelling suggestions for an unknown term."""
    term = normalize_term(raw)
    vocab = term_vocabulary()
    if term not in vocab:
        raise ApiError(404, f"unknown term {term!r}",
                       suggestions=get_close_matches(term, list(vocab), n=5, cutoff=0.6))
    return term, vocab[term]


def parse_xyz(raw):
    """'x_y_z' (MNI mm; commas also accepted) -> [x, y, z]."""
    try:
        xyz = [float(v) for v in raw.replace(",", "_").split("_")]
    except ValueError:
        xyz = []
    if len(xyz) != 3 or not all(math.isfinite(v) and abs(v) <= 200 for v in xyz):
        raise ApiError(400, f"a location must look like x_y_z in MNI mm, e.g. 0_-52_26; got {raw!r}")
    return [int(v) if v.is_integer() else v for v in xyz]


# -----------------------------
# SQL: each set is (study_id, score) for one side, "a" or "b"
# -----------------------------
def term_set(side):
    """Studies whose abstract weight for :<side> exceeds :threshold; score = that weight."""
    return f"""
        SELECT study_id, MAX(weight) AS score
        FROM {SCHEMA}.annotations_terms
        WHERE term = :{side} AND weight > :threshold
        GROUP BY study_id"""


def location_set(side):
    """Studies reporting a peak within :r mm of (:<side>_x, :<side>_y, :<side>_z);
    score = distance of the nearest such peak. PostGIS answers ST_3DDWithin with the GIST
    index (geom && ST_Expand(point, r)) before checking the exact 3-D distance."""
    point = f"ST_SetSRID(ST_MakePoint(:{side}_x, :{side}_y, :{side}_z), {SRID})"
    return f"""
        SELECT study_id, MIN(ST_3DDistance(geom, {point})) AS score
        FROM {SCHEMA}.coordinates
        WHERE ST_3DDWithin(geom, {point}, :r)
        GROUP BY study_id"""


def count_sets(conn, a_sql, b_sql, params):
    """Sizes of A, B, A ∩ B, A \\ B and B \\ A."""
    return dict(conn.execute(text(f"""
        WITH a AS ({a_sql}), b AS ({b_sql})
        SELECT (SELECT COUNT(*) FROM a) AS a,
               (SELECT COUNT(*) FROM b) AS b,
               (SELECT COUNT(*) FROM (SELECT study_id FROM a INTERSECT SELECT study_id FROM b) s) AS a_and_b,
               (SELECT COUNT(*) FROM (SELECT study_id FROM a EXCEPT SELECT study_id FROM b) s) AS a_not_b,
               (SELECT COUNT(*) FROM (SELECT study_id FROM b EXCEPT SELECT study_id FROM a) s) AS b_not_a
        """), params).mappings().one())


def fetch_studies(conn, a_sql, b_sql, params, score_key, descending, limit, offset):
    """Studies in A \\ B with their metadata, best score first."""
    rows = conn.execute(text(f"""
        WITH a AS ({a_sql}), b AS ({b_sql}),
             hits AS (SELECT study_id FROM a EXCEPT SELECT study_id FROM b)
        SELECT a.study_id, m.title, m.authors, m.journal, m.year, a.score
        FROM hits
        JOIN a ON a.study_id = hits.study_id
        LEFT JOIN {SCHEMA}.metadata m ON m.study_id = hits.study_id
        ORDER BY a.score {"DESC" if descending else "ASC"}, a.study_id
        LIMIT :limit OFFSET :offset
        """), {**params, "limit": limit, "offset": offset}).mappings()
    return [{"study_id": r["study_id"], "title": r["title"], "authors": r["authors"],
             "journal": r["journal"], "year": r["year"], score_key: round(r["score"], 4)}
            for r in rows]


def single_set(kind, label, a_sql, params, echo, score_key, descending):
    limit, offset = page_args()
    with get_engine().connect() as conn:
        count = count_sets(conn, a_sql, EMPTY_SET, params)["a"]
        studies = fetch_studies(conn, a_sql, EMPTY_SET, params, score_key, descending, limit, offset)
    return jsonify({kind: label, "params": {**echo, "limit": limit, "offset": offset},
                    "count": count, "studies": studies})


def dissociate(kind, a_label, b_label, a_sql, b_sql, params, echo, score_key, descending):
    limit, offset = page_args()
    with get_engine().connect() as conn:
        counts = count_sets(conn, a_sql, b_sql, params)
        payload = {
            "dissociation": kind, "a": a_label, "b": b_label,
            "params": {**echo, "limit": limit, "offset": offset},
            "counts": counts,
            "count": counts["a_not_b"],
            "studies": fetch_studies(conn, a_sql, b_sql, params, score_key, descending, limit, offset),
        }
        if flag_arg("both"):
            payload["b_not_a"] = {
                "count": counts["b_not_a"],
                "studies": fetch_studies(conn, b_sql, a_sql, params, score_key, descending, limit, offset),
            }
    return jsonify(payload)


def create_app():
    app = Flask(__name__)
    app.json.sort_keys = False
    app.json.ensure_ascii = False

    @app.get("/", endpoint="health")
    def health():
        return INDEX_HTML

    @app.get("/img", endpoint="show_img")
    def show_img():
        return send_file(os.path.join(app.root_path, "amygdala.gif"), mimetype="image/gif")

    @app.get("/terms", endpoint="terms")
    def list_terms():
        q = normalize_term(request.args.get("q", ""))
        terms = sorted(t for t in term_vocabulary() if q in t)
        return jsonify({"q": q, "count": len(terms), "terms": terms})

    @app.get("/terms/<term>/studies", endpoint="terms_studies")
    def get_studies_by_term(term):
        term, stored = resolve_term(term)
        threshold = float_arg("threshold", DEFAULT_THRESHOLD, 0.0, 1.0)
        return single_set("term", term, term_set("a"), {"a": stored, "threshold": threshold},
                          {"threshold": threshold}, "weight", descending=True)

    @app.get("/locations/<coords>/studies", endpoint="locations_studies")
    def get_studies_by_coordinates(coords):
        x, y, z = parse_xyz(coords)
        r = float_arg("r", DEFAULT_RADIUS_MM, 0.0, MAX_RADIUS_MM)
        return single_set("location", [x, y, z], location_set("a"),
                          {"a_x": x, "a_y": y, "a_z": z, "r": r},
                          {"radius_mm": r}, "distance_mm", descending=False)

    @app.get("/dissociate/terms/<term_a>/<term_b>", endpoint="dissociate_terms")
    def dissociate_terms(term_a, term_b):
        a, a_stored = resolve_term(term_a)
        b, b_stored = resolve_term(term_b)
        threshold = float_arg("threshold", DEFAULT_THRESHOLD, 0.0, 1.0)
        return dissociate("terms", a, b, term_set("a"), term_set("b"),
                          {"a": a_stored, "b": b_stored, "threshold": threshold},
                          {"threshold": threshold}, "weight", descending=True)

    @app.get("/dissociate/locations/<coords_a>/<coords_b>", endpoint="dissociate_locations")
    def dissociate_locations(coords_a, coords_b):
        a, b = parse_xyz(coords_a), parse_xyz(coords_b)
        r = float_arg("r", DEFAULT_RADIUS_MM, 0.0, MAX_RADIUS_MM)
        params = {"a_x": a[0], "a_y": a[1], "a_z": a[2],
                  "b_x": b[0], "b_y": b[1], "b_z": b[2], "r": r}
        return dissociate("locations", a, b, location_set("a"), location_set("b"),
                          params, {"radius_mm": r}, "distance_mm", descending=False)

    @app.get("/test_db", endpoint="test_db")

    def test_db():
        eng = get_engine()
        payload = {"ok": False, "dialect": eng.dialect.name}

        try:
            with eng.begin() as conn:
                # Ensure we are in the correct schema
                conn.execute(text("SET search_path TO ns, public;"))
                payload["version"] = conn.exec_driver_sql("SELECT version()").scalar()

                # Counts
                payload["coordinates_count"] = conn.execute(text("SELECT COUNT(*) FROM ns.coordinates")).scalar()
                payload["metadata_count"] = conn.execute(text("SELECT COUNT(*) FROM ns.metadata")).scalar()
                payload["annotations_terms_count"] = conn.execute(text("SELECT COUNT(*) FROM ns.annotations_terms")).scalar()

                # Samples
                try:
                    rows = conn.execute(text(
                        "SELECT study_id, ST_X(geom) AS x, ST_Y(geom) AS y, ST_Z(geom) AS z FROM ns.coordinates LIMIT 3"
                    )).mappings().all()
                    payload["coordinates_sample"] = [dict(r) for r in rows]
                except Exception:
                    payload["coordinates_sample"] = []

                try:
                    # Select a few columns if they exist; otherwise select a generic subset
                    rows = conn.execute(text("SELECT * FROM ns.metadata LIMIT 3")).mappings().all()
                    payload["metadata_sample"] = [dict(r) for r in rows]
                except Exception:
                    payload["metadata_sample"] = []

                try:
                    rows = conn.execute(text(
                        "SELECT study_id, contrast_id, term, weight FROM ns.annotations_terms LIMIT 3"
                    )).mappings().all()
                    payload["annotations_terms_sample"] = [dict(r) for r in rows]
                except Exception:
                    payload["annotations_terms_sample"] = []

            payload["ok"] = True
            return jsonify(payload), 200

        except Exception as e:
            payload["error"] = str(e)
            return jsonify(payload), 500

    @app.errorhandler(ApiError)
    def api_error(err):
        return jsonify(err.payload), err.status

    @app.errorhandler(404)
    def not_found(err):
        return jsonify({"error": "no such endpoint", "see": "/"}), 404

    @app.errorhandler(SQLAlchemyError)
    def db_error(err):
        app.logger.exception("database error")
        return jsonify({"error": "database error", "type": type(err).__name__}), 503

    @app.after_request
    def add_cors_and_jsonp(resp):
        # Any frontend origin may fetch() the API (as in the 2_py_backend_flask.py tutorial) ...
        resp.headers["Access-Control-Allow-Origin"] = "*"
        # ... or load it with <script src="...?callback=fn"> (JSONP, as in 7_get_jsonp.html).
        callback = request.args.get("callback", "")
        if callback and resp.mimetype == "application/json" and JSONP_CALLBACK.match(callback):
            resp.set_data(f"{callback}({resp.get_data(as_text=True).strip()});")
            resp.mimetype = "application/javascript"
        return resp

    return app

# WSGI entry point (no __main__)
app = create_app()
