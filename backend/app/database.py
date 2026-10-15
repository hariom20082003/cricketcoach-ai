from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

def normalize_database_url(database_url: str, require_ssl: bool = False) -> str:
    """Use psycopg 3 for PostgreSQL and require TLS for hosted connections."""
    if database_url.startswith("postgres://"):
        database_url = "postgresql://" + database_url[len("postgres://"):]
    url = make_url(database_url)
    if url.drivername in {"postgresql", "postgresql+psycopg"}:
        url = url.set(drivername="postgresql+psycopg")
        sslmode = url.query.get("sslmode")
        if require_ssl and sslmode is None:
            url = url.update_query_dict({"sslmode": "require"})
        elif require_ssl and sslmode not in {"require", "verify-ca", "verify-full"}:
            raise ValueError("Production PostgreSQL DATABASE_URL must require SSL")
        return url.render_as_string(hide_password=False)
    return database_url


DATABASE_URL = normalize_database_url(settings.DATABASE_URL, require_ssl=settings.is_production)
engine_options = {"pool_pre_ping": True}
if DATABASE_URL.startswith("postgresql+psycopg://"):
    # Supabase's transaction pooler does not preserve prepared statements. Disabling
    # automatic preparation also works with the direct and session-pooler endpoints.
    engine_options.update({
        "pool_size": settings.DB_POOL_SIZE,
        "max_overflow": settings.DB_MAX_OVERFLOW,
        "pool_recycle": settings.DB_POOL_RECYCLE_SECONDS,
        "pool_timeout": 30,
        "connect_args": {"prepare_threshold": None},
    })
elif DATABASE_URL.startswith("sqlite"):
    engine_options["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_options)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
