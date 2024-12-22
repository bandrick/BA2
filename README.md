# Bachelor Thesis - Investigating the Relationship Between Event Log Characteristics and the Performance of Process Discovery Algorithms

This repository includes the scripts and key data used for the thesis.

## Code Overview

- **`Manipulate_Logs.ipynb`**: Notebook providing preprocessing tools to prepare / generate event logs.  
- **`calcEventLogPs.py`**: Core Python script for event log analysis.  
- **`createDatabase.ipynb`**: Notebook for processing event logs, calculating event log characteristics, and performing quality evaluation on discovered process models.  
- **`Analysen.ipynb`**: Notebook showcasing applied methodologies, including deviation calculations, linear correlation analysis, and regression model training.  
- **`auswertung_ohne.ipynb`**: Notebook dedicated to gathering actionable insights from the analysis results.  

This version uses consistent structure, eliminates redundancy, and corrects typos. Let me know if you'd like further refinements!


## Input and Output Data
These two folders contain, the respective input and outpout data. in the input folder, csv files of all xes files can be found, aswell as a list of all downloaded files. additionally the metadata for the analysis can be found in the zipfile database.zip it contains all calculated properties, quality metrics and filter parameters.
The output folder contains all data created using the file **`Analysen.ipynb`**, its splitted into 6 subfolders, coorrmatrix_org contains all corrrlations matritzes splitted by preprocessing type and Process discovery algorithm, corrmatrix_miner_org all Correlations matritzes for the second step, researching the linear corrrelations across all preprocessing types at once. Last but not least for the linear correlations, the Corrmatrix combined, contains one corrmatrix, independatnd of the process discovery algorithm. All correlation matrixes exist as png file aswell as a csv file.
The folders regressionResults_Miner and regressionResults_Miner_X_Preprocessing contain outputs of the regression models, when trying to predict the quality metrics based on the Event Log properties. the two bestminer folders contain the results of the models trying to predict the best miner for a set of event log properties


## Input and Output Data

### Input Data
The **`input`** folder contains the following:
- A  list of all downloaded files.
- Metadata required for the analysis, provided in the `database.zip` file, which includes:
  - All calculated properties.
  - Quality metrics.
  - Filter parameters.

### Output Data
The **`output`** folder contains data generated using the file **`Analysen.ipynb`**, organized into subfolders:

1. **`coorrmatrix_org`**: Contains correlation matrices split by preprocessing type and process discovery algorithm.  
2. **`corrmatrix_miner_org`**: Contains correlation matrices from the second step, examining linear correlations across all preprocessing types simultaneously.  
3. **`Corrmatrix_combined`**: Contains a single correlation matrix that is independent of the process discovery algorithm.  
   - All correlation matrices are available in both PNG and CSV formats.

4. **`regressionResults_Miner`**: Outputs from regression models aimed at predicting quality metrics based on event log properties.  
5. **`regressionResults_Miner_X_Preprocessing`**: Outputs from regression models that combine process discovery algorithms and preprocessing types to predict quality metrics.  
6. **`bestminer` folders**: Results from models attempting to predict the best process discovery miner based on event log properties.
