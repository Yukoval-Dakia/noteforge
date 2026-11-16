"""Request and response shapes. Everything a caller sends is checked here before any work starts."""
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Mode(str, Enum):
    phira = 'phira'
    mania_4k = 'mania_4k'


class Aggregation(str, Enum):
    vote = 'vote'
    single = 'single'
    most_agreeing = 'most_agreeing'


class Segment(Strict):
    start_ms: float = Field(description='Red line time on the audio clock, ms')
    bpm: float = Field(gt=0, le=2000)
    meter: int = Field(default=4, ge=1, le=32)


class Timing(Strict):
    """Exactly one of: full segments, a BPM (the beat phase is found automatically), or an uploaded chart."""
    segments: Optional[list[Segment]] = Field(default=None, min_length=1, max_length=2000)
    bpm: Optional[float] = Field(default=None, gt=0, le=2000)
    chart_id: Optional[str] = None

    @model_validator(mode='after')
    def one_source(self):
        if sum(x is not None for x in (self.segments, self.bpm, self.chart_id)) != 1:
            raise ValueError('give exactly one of segments, bpm, chart_id')
        return self


class Style(Strict):
    """Requested shares, 0 to 1. Anything left out comes from opening_chart_id, then from the planner."""
    hold: Optional[float] = Field(default=None, ge=0, le=1)
    chord: Optional[float] = Field(default=None, ge=0, le=1)
    flick: Optional[float] = Field(default=None, ge=0, le=1, description='Phira only')
    drag: Optional[float] = Field(default=None, ge=0, le=1, description='Phira only')


class CurvePoint(Strict):
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    density: float = Field(ge=0, le=60, description='Notes per second')


class DraftRequest(Strict):
    audio_id: str
    mode: Mode
    level: Optional[float] = Field(default=None, gt=0, le=60, description='Overall notes per second. Required on Phira.')
    timing: Optional[Timing] = None
    opening_chart_id: Optional[str] = Field(default=None, description="The charter's chart of at least the opening: style comes from its first 30 s")
    style: Optional[Style] = None
    density_curve: Optional[list[CurvePoint]] = Field(default=None, min_length=1, max_length=20000, description='Replaces the planned curve')
    aggregation: Aggregation = Aggregation.vote
    samples: int = Field(default=8, ge=1, le=16, description='Also the number of candidates available per section')
    profile: Literal['relaxed', 'standard', 'expert', 'none'] = Field(default='standard', description='Phira playability limits')
    seed: int = Field(default=0, ge=0, le=2**31-1)
    plan: Literal['balanced', 'fast', 'sequential'] = Field(default='balanced', description='MLX backend only. balanced: 3 parallel rounds, quality matched the frozen setup on 100 songs. fast: 2 rounds, quicker, mania likeness fell clearly. sequential: core by core, the slowest')
    title: Optional[str] = Field(default=None, max_length=200)
    artist: Optional[str] = Field(default=None, max_length=200)

    @model_validator(mode='after')
    def consistent(self):
        if self.mode == Mode.phira and self.level is None:
            raise ValueError('Phira needs level: the automatic guess did not pass its check on Phira')
        if self.mode == Mode.mania_4k and self.style and (self.style.flick is not None or self.style.drag is not None):
            raise ValueError('flick and drag exist only on Phira')
        if self.aggregation != Aggregation.single and self.samples < 2:
            raise ValueError(f'{self.aggregation.value} needs at least 2 samples')
        if self.density_curve and any(p.end_s <= p.start_s for p in self.density_curve):
            raise ValueError('each density_curve point needs end_s > start_s')
        return self


class Note(Strict):
    time_ms: float = Field(ge=0)
    end_ms: float = Field(ge=0)
    type: Literal['tap', 'hold', 'flick', 'drag']
    lane: Optional[int] = Field(default=None, ge=0, le=3, description='mania_4k')
    x: Optional[float] = Field(default=None, description='Phira position; left out to let the model place it')

    @model_validator(mode='after')
    def valid(self):
        if self.end_ms < self.time_ms:
            raise ValueError('end_ms is before time_ms')
        if self.type != 'hold' and self.end_ms != self.time_ms:
            raise ValueError('only a hold can have end_ms after time_ms')
        return self


class Base(Strict):
    kind: Literal['vote', 'sample'] = 'vote'
    sample: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode='after')
    def filled(self):
        if (self.kind == 'sample') != (self.sample is not None):
            raise ValueError('base.sample goes with base.kind sample, and only with it')
        return self


class Choice(Strict):
    """One section: its notes are replaced by a sample's notes, or by the caller's own."""
    start_ms: float = Field(ge=0)
    end_ms: float = Field(gt=0)
    sample: Optional[int] = Field(default=None, ge=0)
    notes: Optional[list[Note]] = Field(default=None, max_length=20000, description="The charter's finished notes for this section. Generated notes that collide with them give way")

    @model_validator(mode='after')
    def valid(self):
        if self.end_ms <= self.start_ms:
            raise ValueError('end_ms must be after start_ms')
        if (self.sample is None) == (self.notes is None):
            raise ValueError('give exactly one of sample, notes')
        if self.notes and any(not self.start_ms <= n.time_ms < self.end_ms for n in self.notes):
            raise ValueError('every note must start inside the section')
        return self


class RevisionRequest(Strict):
    base: Base = Base()
    choices: list[Choice] = Field(default_factory=list, max_length=2000)


class ContinueRequest(Strict):
    """Keep everything before from_ms and write the rest. The audio is not read again."""
    from_ms: float = Field(ge=0)
    revision: Optional[int] = Field(default=None, ge=0, description='Revision to continue from; latest by default')
    seed: int = Field(default=1, ge=0, le=2**31-1, description='Change it to get a different continuation')
