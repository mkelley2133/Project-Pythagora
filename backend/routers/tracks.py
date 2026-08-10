from fastapi import APIRouter, HTTPException

from backend.models.schemas import TelemetryBundle, TrackAnalysisRecord, TrackRecord

router = APIRouter(prefix="/tracks", tags=["tracks"])


@router.get("", response_model=list[TrackRecord])
def list_tracks() -> list[TrackRecord]:
    return [
        TrackRecord(
            id="demo-track-001",
            title="Sample Signal",
            artist="Pythagoras",
            original_file_path="/storage/demo.mp3",
            vocal_stem_path="/storage/demo_vocals.wav",
            instrumental_stem_path="/storage/demo_inst.wav",
        )
    ]


@router.post("", response_model=TrackRecord)
def create_track(track: TrackRecord) -> TrackRecord:
    return track


@router.get("/{track_id}/analysis", response_model=TrackAnalysisRecord)
def get_analysis(track_id: str) -> TrackAnalysisRecord:
    if track_id != "demo-track-001":
        raise HTTPException(status_code=404, detail="Track not found")

    telemetry = TelemetryBundle(
        bpm=92.4,
        musical_key="D",
        mode="minor",
        sub_genres=[{"name": "dream pop", "confidence": 0.82}],
        emotional_profile={"hope": 0.7, "melancholy": 0.8},
        chords=["Dm", "Bb", "F", "C"],
        rhyme_scheme_summary="ABCB",
        musicological_essay="A concise placeholder analysis for the starter pipeline.",
    )
    return TrackAnalysisRecord(id=track_id, telemetry_json=telemetry)
