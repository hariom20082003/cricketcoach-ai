"""Harden CricketCoach tables against direct Supabase Data API access.

Revision ID: f6a1c3d9e742
Revises: b7fd8d64cd78
Create Date: 2026-10-16

The FastAPI service continues to connect as the existing database owner. This
migration does not create Supabase Auth policies or change Supabase-managed
schemas, tables, sequences, roles, or storage privileges.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f6a1c3d9e742"
down_revision: Union[str, None] = "b7fd8d64cd78"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


APPLICATION_TABLES = (
    "users",
    "auth_sessions",
    "videos",
    "analysis_jobs",
    "shot_predictions",
    "pose_results",
    "technique_observations",
    "model_versions",
    "job_model_version",
    "practice_drills",
    "completed_drills",
    "session_notes",
    # Alembic bookkeeping is not a business table, but must not be exposed.
    "alembic_version",
)

APPLICATION_SEQUENCES = (
    "users_id_seq",
    "videos_id_seq",
    "analysis_jobs_id_seq",
    "completed_drills_id_seq",
    "model_versions_id_seq",
    "pose_results_id_seq",
    "practice_drills_id_seq",
    "session_notes_id_seq",
    "shot_predictions_id_seq",
    "technique_observations_id_seq",
)

CLIENT_ROLES = ("anon", "authenticated")


def _is_postgresql() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _role_exists(role: str) -> bool:
    return bool(
        op.get_bind().execute(
            sa.text("SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :role)"),
            {"role": role},
        ).scalar()
    )


def _revoke_existing_object_privileges() -> None:
    for role in CLIENT_ROLES:
        if not _role_exists(role):
            continue
        for table in APPLICATION_TABLES:
            op.execute(f'REVOKE ALL PRIVILEGES ON TABLE public."{table}" FROM "{role}"')
        for sequence in APPLICATION_SEQUENCES:
            op.execute(f'REVOKE ALL PRIVILEGES ON SEQUENCE public."{sequence}" FROM "{role}"')


def _restore_existing_object_privileges() -> None:
    # This restores the grants observed on the Supabase project before this
    # migration. The migration is scoped to the known application objects.
    for role in CLIENT_ROLES:
        if not _role_exists(role):
            continue
        tables = ", ".join(f'public."{name}"' for name in APPLICATION_TABLES)
        sequences = ", ".join(f'public."{name}"' for name in APPLICATION_SEQUENCES)
        op.execute(f'GRANT ALL PRIVILEGES ON TABLE {tables} TO "{role}"')
        op.execute(
            f'GRANT USAGE, SELECT, UPDATE ON SEQUENCE {sequences} TO "{role}"'
        )


def _harden_default_privileges() -> None:
    for role in CLIENT_ROLES:
        if not _role_exists(role):
            continue
        op.execute(
            f'ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public '
            f'REVOKE ALL ON TABLES FROM "{role}"'
        )
        op.execute(
            f'ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public '
            f'REVOKE ALL ON SEQUENCES FROM "{role}"'
        )
    # There are no application functions. PUBLIC's default EXECUTE privilege
    # is global to functions created by postgres, so changing it here could
    # affect future managed functions in other schemas. Review any future
    # public-schema function separately and place private functions outside
    # the exposed schema.


def _restore_default_privileges() -> None:
    for role in CLIENT_ROLES:
        if not _role_exists(role):
            continue
        op.execute(
            f'ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public '
            f'GRANT ALL ON TABLES TO "{role}"'
        )
        op.execute(
            f'ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public '
            f'GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO "{role}"'
        )


def upgrade() -> None:
    if not _is_postgresql():
        return

    bind = op.get_bind()
    public_functions = bind.execute(
        sa.text(
            "SELECT count(*) FROM pg_proc p "
            "JOIN pg_namespace n ON n.oid = p.pronamespace "
            "WHERE n.nspname = 'public'"
        )
    ).scalar_one()
    if public_functions:
        raise RuntimeError(
            "Refusing to harden public function grants without a function-specific review"
        )

    for table in APPLICATION_TABLES:
        op.execute(f'ALTER TABLE public."{table}" ENABLE ROW LEVEL SECURITY')

    _revoke_existing_object_privileges()
    _harden_default_privileges()


def downgrade() -> None:
    if not _is_postgresql():
        return

    _restore_default_privileges()
    _restore_existing_object_privileges()
    op.execute('ALTER TABLE public."alembic_version" DISABLE ROW LEVEL SECURITY')
