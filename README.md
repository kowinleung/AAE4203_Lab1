# AAE4203_Lab1_Report 
This repository contains the code, data, and supporting files for the Lab 1: GNSS Positioning experiment in the AAE4203 Guidance and Navigation course. The project focuses on collecting raw GNSS data using a u-blox ZED-F9P receiver, processing it with RTKLIB software, implementing a Single Point Positioning (SPP) algorithm using Weighted Least Squares Estimation (WLSE) in Python, and analyzing the results in an urban canyon environment.

Key topics include:
- GNSS data collection in challenging urban settings.
- Raw data analysis.
- SPP implementation with optional weighting based on satellite elevation.
- Performance evaluation with error metrics.
The experiment demonstrates the limitations of standalone GNSS in urban areas due to multipath and signal obstructions.

**Data download**
Raw data can be donwloaded in github or from the DropBox link below:
https://www.dropbox.com/scl/fo/b1fdy5pnpxvjzsqkues6x/AHNhQb65d_mF6wkFSRcKonc?rlkey=ucgfu80l1apoel1lkqfs0zrtc&st=fdbzwjso&dl=0

## Repository Structure

- **`report/`**: Contains the full lab report PDF.
  - `PolyＵ_AAE4203_Lab_report_Group_7_LEUNG_Ho_Fung.pdf`: The complete report with abstract, sections, figures, tables, and references.
- **`Used_code/`**: Python scripts for data processing and SPP implementation.
  - `WLSE_SPP.py`: Main script for Weighted Least Squares SPP algorithm. Implements pseudorange corrections, elevation-based weighting, and iterative position estimation.
  - `csv_converter_gui.py`: GUI tool to convert UBX/NMEA logs to CSV; depends on pyubx2.
  - `rinex2csv.m`: MATLAB script to convert RINEX observation/navigation files into CSV; depends on matRTKLIB.
- **`Raw_data/`**: Raw data files used in the analysis.
  - Com3 and Com4 raw GNSS data file was recorded on 27th October, 2025.
  - The UBX files were converted into OBS, KML, NAV file using RTKLIB.
- **`Pictures_Graphs/`**: Generated plots and visualizations.
  - Includes graphs generated from RTKPLOT, MATLAB, and the python codes.
- **`Output_CSV/`**: Generated CSV files from python codes
- **`README.md`**

### Prerequisites Software
- Python 3.x
- Libraries: `numpy`, `pandas`, `matplotlib`, `cartopy`, `pyubx2`.
- RTKLIB v2.4.2 or later.
- MATLAB.
- u-blox u-center software.

## References
- P. Misra and P. Enge, *GPS: Signals, Measurements, and Performance*, 2nd ed., Ganga-Jamuna Press, 2011.
- P. J. G. Teunissen and O. Montenbruck, *Springer Handbook of GNSS*, Springer, 2017.
- T. Takasu, *RTKLIB ver.2.4.2 Manual*, 2013. [Link](https://www.rtklib.com/rtklib_document.htm).
- T. Takasu, RTKLIB GitHub Repository. [Link](https://github.com/tomojitakasu/RTKLIB).
- E. D. Kaplan and C. J. Hegarty (eds.), *Understanding GPS/GNSS: Principles and Applications*, 3rd ed., Artech House, 2017.
- European Space Agency (ESA), *GNSS Data Processing, Volume I: Fundamentals and Algorithms*, ESA TM-23/1, 2013.
