"""Frozen low-level inference presets shipped with the model."""
TABLE = {
 'mania_star': {'heads_per_s': [2.119, 2.934, 4.021, 4.885, 5.66, 6.087, 6.99, 7.77, 8.884, 9.498, 9.498,
                                10.475, 10.702],
                'x': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]},
 'mania_star_algorithm': 'approximate legacy osu!mania strain (noteforge.events.difficulty_presets.mania_star)',
 'phira_level': {'fit': {'kind': 'log-linear', 'per_level_factor': 1.1281},
                 'heads_per_s': [1.486, 1.676, 1.891, 2.133, 2.406, 2.714, 3.062, 3.454, 3.896, 4.395, 4.958,
                                 5.592, 6.308],
                 'x': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]},
 'schema': 'noteforge.difficulty-presets.v1',
 'unit': 'heads (distinct onsets) per second, song level'}
