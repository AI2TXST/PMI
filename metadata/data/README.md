# TXST Donor and Weather Raw Data

This directory contains the original donor-, death-, placement-, and weather-level data used to build the TXST decomposition dataset.

---

## Files and Purpose

### `Intake_OG.xlsx`
Donor intake and initial handling information.

Key content:
- Donor IDs: `IntakeID`, `TXSTDonorNumber`
- Dates: `DateReceived`, `DateOfIntake`
- Body handling: `InitialPlacementLocation` (Surface, Burial, Freezer, Cooler, Cremains), approximate time of placement (`ApproxTime`)
- Body measurements: `CadaverStaturecm`, `FootLengthcm`, `CadaverWeightlbs`, `WaistCircumferencecm`
- Personal effects: presence and description (`PersonalEffects`, `PersonalEffectsDescription`)
- Documentation of samples and photography: `SamplesCollected`, `PhotographID`, per-region photo flags (overall body, face, limbs, scars, tattoos, injuries, jewelry)
- Autopsy and organ donation flags: `Autopsy`, `FullAutopsy`, regional autopsy flags, `OrganDonor`, `DonatedEyes`, `DonatedSkin`, `DonatedBones`, `DonatedInternalOrgans`
- Data provenance: `RecordEntryBy`, `RecordEntryDate`, `RecordAuditBy`, `RecordAuditDate`, `RecordUpdateBy`, `RecordUpdateDate`, `Paperless`

Used for:
- Baseline body condition and measurements at intake
- Initial placement context and personal effects
- Linking intake events to downstream bioprofile and imaging via `TXSTDonorNumber`

---

### `Research_History_OG.xlsx`
Longitudinal record of each donor’s use in research projects.

Typical content (by row, per donor-project event):
- `TXSTDonorNumber`
- Study/project identifiers and dates
- Types of research activities or sampling events
- Administrative tracking fields (entry/audit metadata)

Used for:
- Tracking which donors were involved in which studies
- Deriving research exposure metadata and destructive sampling context
- Merging into a donor-level research history table

---

### `Facts_of_Death_OG.xlsx`
Cause-of-death and circumstances-of-death information for each donor.

Key columns:
- `FactsOfDeathID`
- `TXSTDonorNumber`
- Demographics at death: `AgeAtDeath`
- Death timing: `DateOfDeath`, `DODSpecify` (e.g., “Found”, “Actual”)
- Medical cause: `ImmediateCOD` (e.g., hypertensive cardiovascular disease, metastatic cancer, drug overdose)
- `MannerOfDeath`: Natural, Accident, Suicide, Homicide, Undetermined, etc.
- Circumstances and location:
  - `LocationOfDeath` (Private residence, Hospital inpatient, Hospice facility, Nursing home, Other)
  - `FacilityOfDeath` (hospital or facility name where applicable)
  - `CityOfDeath`, `StateOfDeath`
- Data lineage: `RecordEntryBy`, `RecordEntryDate`, `RecordAuditBy`, `RecordAuditDate`, `RecordUpdateBy`, `RecordUpdateDate`

Used for:
- Age-at-death filtering (e.g., adult-only subsets)
- Cause/manner-of-death features and grouping
- Urban/rural and facility type context at time of death

---

### `Life_History_Query_OG.xlsx`
Life history and background information for donors.

Typical content:
- `TXSTDonorNumber`
- Life history attributes (e.g., occupational, medical, or social history fields depending on the original query)
- Additional demographics or risk factors as defined in the life-history query
- Record entry and audit metadata

Used for:
- Enriching donors with life-history covariates
- Building a comprehensive donor bioprofile when merged with intake and facts-of-death

---

### `Donor-Placement-Info-OG.csv` and `Donor_Placement_Info_OG.xlsx`
Placement history and body disposition details.

Content (per placement event):
- `TXSTDonorNumber`
- Placement dates and times
- Placement location and type (e.g., FARF surface, burial plot, cooler/freezer, other facilities)
- Any notes on movement between facilities/locations
- Administrative entry/audit fields

Used for:
- Deriving time-since-placement and environmental exposure intervals
- Linking donors to specific facility/placement contexts
- Building time-series alignment with weather and imaging

---

### `Destructive_Analysis_Query_OG.xlsx`
Records of destructive sampling and elements sampled from each donor.

Key content:
- `TXSTDonorNumber`
- Sampling metadata:
  - Sampling event IDs
  - `SamplingType`
  - `ElementsSampled` (e.g., bone, tooth, tissue codes)
  - Free-text description (`SpecifySamples`) detailing which anatomical elements were removed
- Researcher names involved in sampling
- Entry/audit metadata

Used for:
- Flagging donors involved in destructive sampling
- Linking specific anatomical sampling to research history
- Constructing a “destructive analysis” component of the donor bioprofile

---

### `Bioprofile_Query_OG.xlsx` and `Bioprofile_Query.xlsx`
Pre-assembled bioprofile views of donors.

Content:
- `TXSTDonorNumber`
- Demographic profile: age, sex, race, Hispanic origin, height, weight (and sometimes estimated cadaver stature/weight)
- Core life history and medical variables as defined in the bioprofile query
- Some death and placement context fields (depending on the exported view)
- Record entry/audit metadata

Used for:
- As a base bioprofile table to merge with Facts-of-Death and other donor tables
- Cross-checking demographics and ensuring consistency across separate source files

---

### `open-meteo-29.91N97.96W239m.csv`
Hourly (or sub-hourly) weather time series downloaded from Open-Meteo for the Freeman Ranch vicinity (approx. 29.91°N, 97.96°W, elevation 239 m).

Typical columns:
- `time` stamps in local or UTC
- Weather variables such as:
  - Air temperature
  - Relative humidity
  - Precipitation and precipitation type
  - Wind speed/direction
  - Possibly radiation, cloud cover, and pressure (depending on the selected API options)

Used for:
- Environmental covariates at or around donor placement and imaging times
- Filling gaps or extending local station data with a model-based reanalysis source
- Sensitivity checks or alternate weather feature sets relative to station data

---

## How These Files Fit Together

At a high level, these raw files are linked by `TXSTDonorNumber` and dates:

- Donor identity and demographics: from `Bioprofile_Query_OG.xlsx` / `Bioprofile_Query.xlsx`
- Intake and initial placement: from `Intake_OG.xlsx`
- Life history: from `Life_History_Query_OG.xlsx`
- Cause and manner of death: from `Facts_of_Death_OG.xlsx`
- Placement trajectory: from `Donor-Placement-Info-OG.*`
- Research and destructive sampling: from `Research_History_OG.xlsx` and `Destructive_Analysis_Query_OG.xlsx`
- Weather context: from `open-meteo-29.91N97.96W239m.csv`

Together, they form the raw inputs used in downstream notebooks to construct a unified donor-level and image-level analytic dataset.
