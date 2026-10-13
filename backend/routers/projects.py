"""Project routes (Phase F) — organize tracks into projects with notes.

Deleting a project never deletes its tracks: they fall back to unsorted.
"""
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from backend import security, store
from backend.models.schemas import Project, ProjectCreate, ProjectUpdate

router = APIRouter(
    prefix="/projects",
    tags=["projects"],
    dependencies=[Depends(security.rate_limit_api)],
)


def _project_view(project: Project) -> dict:
    view = project.model_dump()
    view["track_count"] = len(store.project_track_ids(project.id))
    view["track_ids"] = store.project_track_ids(project.id)
    return view


@router.get("")
def list_projects() -> list[dict]:
    return [_project_view(p) for p in store.list_projects()]


@router.post("", status_code=status.HTTP_201_CREATED)
def create_project(body: ProjectCreate) -> dict:
    project = Project(
        id=f"project-{uuid4().hex[:12]}",
        name=body.name.strip(),
        notes=body.notes,
        created_at=datetime.now(timezone.utc),
    )
    return _project_view(store.save_project(project))


@router.get("/{project_id}")
def get_project(project_id: str) -> dict:
    security.validate_id(project_id, "project id")
    project = store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return _project_view(project)


@router.patch("/{project_id}")
def update_project(project_id: str, body: ProjectUpdate) -> dict:
    security.validate_id(project_id, "project id")
    project = store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if body.name is not None:
        project.name = body.name.strip()
    if body.notes is not None:
        project.notes = body.notes
    return _project_view(store.save_project(project))


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: str) -> None:
    security.validate_id(project_id, "project id")
    if not store.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return None
