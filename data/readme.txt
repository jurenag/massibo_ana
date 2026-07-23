MASSIBO data-quality flags
==========================

Purpose
-------
These files record data-quality information for MASSIBO production measurements.
They flag SiPMs whose data-taking was spoiled by INSTRUMENTAL ARTIFACTS --
problems of the measurement/setup, not of the SiPM itself. Examples: sinusoidal
pickup that exhausted the trigger budget with spurious triggers or confused
the peak-finder into spotting false dark counts, or a DAQ fault that corrupted
the trigger settings.

This is deliberately distinct from a SiPM being intrinsically noisy (a real,
physical property of the device, e.g. a genuinely high DCR). Those are NOT
instrumental artifacts and are not flagged for exclusion here.

The intended use is to derive a good-run list by excluding the flagged
entries. Not every flag implies exclusion: whether a flag excludes a SiPM
or is merely informational is decided per artifact type (see below), not
per SiPM.


Files
-----
darknoise_artifacts.csv	Catalogue of artifact types for darknoise measurements (the "why"). One row per type.
darknoise_flags.csv    	Register of SiPMs whose darknoise measurement is affected (the "what"). One row per SiPM.
gain_artifacts.csv	Catalogue of artifact types for gain measurements. One row per type.
gain_flags.csv		Register of SiPMs whose gain measurement is affected. One row per SiPM.

The two files are normalised: <type>_flags.csv references <type>_artifacts.csv
through the `artifact` key, so long descriptions are not repeated on every row.
Every `artifact` value appearing in <type>_flags.csv must exist in
<type>_artifacts.csv.


<type>_artifacts.csv schema
--------------------
artifact     Short key identifying the artifact type (e.g. sine_pickup).
exclude      True  -> results from SiPMs with this artifact are not reliable
             False -> informational only
description  Human-readable description of the artifact. Quote it (") whenever
             it contains a comma, so that the file stays a well-formed CSV.


<type>_flags.csv schema
----------------
set        Set number
meas       Measurement number
strip      Strip-ID
sipm       SiPM position on the card (1 to 6).
artifact   Artifact key; joins to <type>_artifacts.csv.

Granularity: one row per affected SiPM. A whole card affected by an artifact
is written as its 6 individual rows (there is no wildcard/"all" convention;
every row is a concrete SiPM).


Extending
---------
- New flags: append rows to <type>_flags.csv (one per affected SiPM). It is 
  convenient to order the rows by set and meas.
- New artifact types: add a row to <type>_artifacts.csv first, then reference
  its key from <type>_flags.csv.
- If a SiPM ever needs an exclusion decision that differs from its artifact's
  default, prefer adding a dedicated artifact type (or an explicit per-row
  override column) rather than making `exclude` ambiguous.


Note
----
<type>_flags.csv may contain SiPMs which had already been discarded by the
`analysis_reliability` variable of the analysis pipelines (see
`apps/darknoise_batch_analyzer.ipynb` and `apps/gain_batch_analyzer.ipynb`).