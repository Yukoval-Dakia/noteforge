"""Phira .rpe chart-document writer used by the export path.

``rpe`` serialises placed rhythm events (t_us/end_us/type/x) into the Phira
JSON chart schema on a fixed 60 BPM clock that encodes seconds; ``plain_background``
returns the minimal PNG the package requires. Both are pure functions shared by
the HTTP export path (``noteforge_api.ops``), the Phira preview exporter, and the
learned-layout path.
"""
from __future__ import annotations

import struct
import zlib
from fractions import Fraction


def beat(us):
    q = Fraction(us, 1000000)
    whole = q.numerator // q.denominator
    rest = q - whole
    return [whole, rest.numerator, rest.denominator]


def rpe(events, label, speed=10.):
    def constant(value):
        return [dict(startTime=[0, 0, 1], endTime=[1000, 0, 1], start=value, end=value,
                     easingType=1, easingLeft=0., easingRight=1., bezier=0,
                     bezierPoints=[0., 0., 0., 0.], linkgroup=0)]
    kinds = dict(tap=1, hold=2, flick=3, drag=4)
    notes = [dict(above=1, alpha=255, startTime=beat(e['t_us']), endTime=beat(e['end_us']),
                  isFake=0, positionX=e['x'], size=1., speed=1., type=kinds[e['type']],
                  visibleTime=999999., yOffset=0.) for e in events]
    return dict(BPMList=[dict(bpm=60., startTime=[0, 0, 1])],
                META=dict(RPEVersion=150, background='background.png', charter='NoteForge / display layout',
                          composer='Sakuzyo', id='', level=label, name='Altale', offset=0, song='audio.ogg'),
                judgeLineGroup=['Default'], multiLineString='', multiScale=1.,
                judgeLineList=[dict(Group=0, Name='Static display line', Texture='line.png',
                                    bpmfactor=1., father=-1, isCover=1, zOrder=0,
                                    eventLayers=[dict(alphaEvents=constant(255), moveXEvents=constant(0.),
                                                      moveYEvents=constant(-230.), rotateEvents=constant(0.),
                                                      speedEvents=constant(float(speed)))],
                                    numOfNotes=len(notes), notes=notes)])


def plain_background():
    # A plain color file required by the package; not an illustrative image.
    w, h = 16, 9
    raw = b''.join(b'\0' + bytes((13, 20, 31))*w for _ in range(h))
    def chunk(kind, data):
        return struct.pack('>I', len(data))+kind+data+struct.pack('>I', zlib.crc32(kind+data))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
