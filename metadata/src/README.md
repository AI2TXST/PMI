# TXST Decomposition Dataset – Data Processing Pipeline

This repository documents the data processing and feature preparation pipeline used to build a per‑image dataset combining donor metadata, weather information, and image‑derived features.

The pipeline is implemented in a series of Jupyter notebooks:

- `1_data_processing.ipynb`
- `2_data_cleaning-2.ipynb`
- `3_filtering_selectedImages-3.ipynb`
- `4_final_metadata-4.ipynb`
- `filling_missing_weather-6.ipynb`
- `feature_selection-5.ipynb`

---

## 1. Overview

The goal of the pipeline is to:

1. Merge multiple donor‑level spreadsheets into a single bioprofile table.  
2. Clean and downsample high‑frequency weather data to an hourly time series.  
3. Align photos with donor metadata and weather data by date/time.  
4. Handle missing weather values so no image rows are lost during joins.  
5. Build a unified per‑image table with labels, donor context, and environmental variables.  
6. Select a compact subset of informative features for modeling.

The final output is a training‑ready metadata CSV in which each row represents one image with:

- File paths and split (train/val/test).  
- Donor identifiers and bioprofile information.  
- Weather conditions around the imaging time.  
- Decomposition‑related annotations and engineered features.  
- A `classlabel` column for modeling.

---

## 2. Raw Data Inputs

The pipeline uses several raw tables and data sources:

- **Donor metadata (multiple spreadsheets)**  
  - Intake data  
  - Life history  
  - Research history  
  - Placement information  
  - Destructive sampling / facts‑of‑death  

- **Facts‑of‑death / bioprofile table**  
  - Age at death, sex, race  
  - Cause and manner of death  
  - Location and facility of death  

- **Weather data (Freeman Ranch 2015)**  
  - High‑frequency logger output with two‑row headers  
  - Columns such as date, time, TempOut, high/low temp, humidity, dewpoint, wind speed/direction, rain, solar radiation/energy, etc.

- **Photo mapping metadata**  
  - Donor ID, donor number/year  
  - Photo type (e.g., Intake, Decomposition, Disarticulation)  
  - Photo dates

- **Image‑derived features**  
  - Hundreds to thousands of per‑image feature columns (`imgfXXX`) created by earlier image processing steps.

---

## 3. Donor‑Level Processing (`1_data_processing.ipynb`)

This notebook builds a unified donor‑level bioprofile:

1. **Load donor tables**  
   - Intake, life history, research history, placement, and destructive sampling datasets.

2. **Type cleaning and filtering**  
   - Convert `AgeAtDeath` to numeric and drop non‑adult donors (age \< 18).  
   - Filter to curated donors where applicable.

3. **Merging donor sources**  
   - Merge intake, life, research, and destructive sampling tables on `TXSTDonorNumber` (and relevant keys such as age and dates).  
   - Consolidate into a single donor‑level table with biographical, death, and sampling information.

4. **Export intermediate tables**  
   - Save merged donor‑level CSVs such as `BioProfileDestructiveFacts.csv` and year‑specific subsets like `BioProfileDestructiveFacts2015.csv` (adult donors with facts‑of‑death and sampling details).

---

## 4. Weather Cleaning and Hourly Aggregation (`2_data_cleaning-2.ipynb`)

This notebook cleans the raw 2015 weather logger file:

1. **Header normalization**  
   - Combine a two‑row header into a single row of column names.  
   - Rename to readable names such as:  
     - `date`, `time`  
     - `TempOut`, `temphigh`, `templow`  
     - `humidityout`, `dewpoint`  
     - `windspeed`, `winddir`, `WindRun`  
     - `Rain`, `RainRate`, `SolarRad`, `SolarEnergy`  
     - Indoor humidity/temperature and other station diagnostics.

2. **Column selection and standardization**  
   - Retain the key weather variables required for modeling.  
   - Ensure consistent types for date/time and numeric weather fields.

3. **Hourly downsampling**  
   - Filter rows to keep only times where `minute == 0` (e.g., 00:00, 01:00, …) to obtain an hourly series.

4. **Export cleaned weather**  
   - Save the hourly dataset as `FreemanRanchWeather2015hourlycleaned.csv` for downstream joins.

---

## 5. Joining Weather, Photos, and Donors (`2_data_cleaning-2.ipynb`)

The same notebook also performs key joins:

1. **Load donor‑level facts and photo mapping**  
   - Import `BioProfileDestructiveFacts2015.csv`.  
   - Import the TXST DSC photo mapping file (donor number, year, photo type, photo date).

2. **Clean photo dates**  
   - Combine year/month/day columns to form a proper `PhotoDate` datetime.  
   - Drop extraneous columns after consolidation.

3. **Join weather with photos**  
   - Match hourly weather rows to photos by date (`date` == `PhotoDate`).  
   - This yields per‑photo rows with hourly environmental measurements around each capture time.

4. **Join with donor bioprofile**  
   - Merge the weather+photo table with donor‑level bioprofile on `TXSTDonorNumber`.  
   - The result is a wide, per‑photo table that includes:  
     - Donor identity and demographics  
     - Death and sampling information  
     - Hourly weather variables  
     - Photo type and date

5. **Reorder and export**  
   - Move `TXSTDonorNumber`, `date`, and `time` to the front for readability.  
   - Save the combined table as `BioProfileDestructiveFactsWeather2015.csv`.

---

## 6. Image Filtering and Final Metadata Layout

Two notebooks further refine these image‑level tables:

### `3_filtering_selectedImages-3.ipynb`

- Filter to a curated subset of “selected” images (e.g., certain time points or photo types).  
- Ensure each selected image has an associated donor record and weather entry.  
- Add image‑specific information such as:  
  - Image file paths (and segmentation mask paths if available).  
  - Image type (Intake, Decomposition, Disarticulation, etc.).  
  - Unique per‑image IDs.

### `4_final_metadata-4.ipynb`

- Define the final schema for the modeling dataset:  
  - Paths and splits (`train` / `val` / `test`).  
  - Donor IDs and study identifiers.  
  - Date/time and derived temporal features (year, month, day, season flags).  
  - Decomposition labels and other annotations.  
  - Weather features aligned to the imaging time.  
  - Target label (`classlabel`).

- Export a final per‑image metadata CSV to be used directly in modeling.

---

## 7. Handling Missing Weather (`filling_missing_weather-6.ipynb`)

This notebook focuses on missingness in the weather fields:

1. **Identify missing values**  
   - Compute counts/percentages of missing data for each weather variable (e.g., `TempOut`, `humidityout`, `windspeed`, `rainrate`).

2. **Imputation strategy**  
   - Apply filling or interpolation (e.g., forward/backward fill or numeric interpolation) on continuous weather variables.  
   - The goal is to avoid dropping image rows due to missing weather at a given timestamp.

3. **Save revised metadata**  
   - Write a revised metadata table with complete weather columns so later modeling and feature selection can assume no missing weather values.

---

## 8. Feature Engineering and Selection (`feature_selection-5.ipynb`)
**This was run on Leap2 GPU powered*
This notebook builds the final modeling table and prunes features:

### 8.1 Base modeling table

- Start from a “full” per‑image dataset that includes:  
  - **Decomposition annotations** – binary/ordinal labels capturing visual state, such as:  
    - `nearnormalcontour`, `paleskin`, `smoothintactskin`,  
    - `wet tissue breakdown`, `bodyfluidsvisible`,  
    - `collapsevisible`, `liquefactionvisible`,  
    - `milddiscoloration`, `marblingvisible`,  
    - and related indicators.  
  - **Weather variables** – `TempOut`, `templow`, `temphigh`, `humidityout`, `dewpoint`, `windspeed`, `WindRun`, `rainrate`, solar metrics, etc.  
  - **Temporal/context variables** – `year`, `month`, `day`, `winddirdeg`, seasonal dummy variables.  
  - **Image features** – many numerical columns named like `imgfXXXX` derived from the image processing pipeline.  
  - **Target** – a `classlabel` column (later converted to category codes).

- Drop non‑feature columns (IDs, text fields, filenames, etc.) from `X` and use `classlabel` as `y`.  
- Split into train/test sets and standardize numeric features with a `StandardScaler`.

### 8.2 Feature selection methods

Define a generic `performfeatureselection(...)` function that runs multiple methods:

- **Filter/linear methods**  
  - VarianceThreshold (remove near‑constant features).  
  - L1‑penalized logistic regression (sparse coefficients).

- **Tree‑based feature importance**  
  - Random Forest importance.  
  - Gradient Boosting importance.  
  - LightGBM importance.  
  - XGBoost importance.  
  - CatBoost importance.

- **Permutation importance**  
  - Permutation importance using Random Forest.  
  - Permutation importance using logistic regression.

- **Wrapper methods (RFE / SFS)**  
  - RFE with Random Forest.  
  - RFE with logistic regression.  
  - Sequential Feature Selector (SFS) with KNN.  
  - SFS with logistic regression.

For each method, the function returns:

- Method name  
- Selected feature indices (or a ranking)  
- Scores (importance, coefficient magnitude, etc.)  
- Score type

### 8.3 Aggregation and export

- Collect results from all methods into a single long table with columns such as:  
  - `Method`  
  - `Feature Index`  
  - `Feature Name`  
  - `Score`  
  - `Score Type`

- Save this table as `featureselectionresultsPMI.csv`.  
- Filter to meaningful `Importance` scores and non‑zero values to produce a compact list of high‑value features, including:  
  - Key decomposition labels (`dryleatherytissue`, `mildfullnessorswelling`, `nearnormalcontour`, etc.).  
  - A subset of high‑importance `imgfXXXX` image features.  
  - Selected environmental variables.

- Use this feature list to slice the full modeling table into a reduced feature matrix for training.

---

## 9. Final Outputs

The end products of the pipeline are:

1. **Donor‑level tables**  
   - `BioProfileDestructiveFacts.csv`  
   - `BioProfileDestructiveFacts2015.csv`  

2. **Weather tables**  
   - `FreemanRanchWeather2015hourlycleaned.csv`  
   - An imputed version with missing weather filled (name as used in your notebooks).

3. **Combined donor + weather + photo tables**  
   - `BioProfileDestructiveFactsWeather2015.csv` and related intermediate CSVs.

4. **Final modeling metadata**  
   - A per‑image metadata CSV including:  
     - Image path, mask path, train/val/test split  
     - Donor IDs and study identifiers  
     - Date/time, temporal/seasonal features  
     - Decomposition labels and weather features  
     - Selected image features (`imgfXXXX`)  
     - Target label (`classlabel`)

5. **Feature selection summary**  
   - `featureselectionresultsPMI.csv` summarizing which features are favored by different selection methods and their scores.

---

## 10. Reproducibility Notes

- Each notebook is designed to be run in order; later notebooks expect CSVs produced earlier.  
- Ensure that file paths and environment variables (e.g., data directory, image root) are configured correctly before running.  
- The feature selection notebook depends on the cleaned and joined metadata created by the earlier processing notebooks.
