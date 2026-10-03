"""End-to-end radiomics pipeline.

DICOM images + masks -> preprocessing -> mask alignment/registration -> PyRadiomics
features -> label matching -> leakage-free nested-CV feature selection, model training
and hyper-parameter tuning -> metrics, plots and an HTML report.
"""

__version__ = "1.0.0"
