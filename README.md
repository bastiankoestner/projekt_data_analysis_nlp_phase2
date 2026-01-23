## Beschreibung
Dieses Repository enthält Phase 2 (Erarbeitungs-/Reflexionsphase) des Projekts *Data Analysis – NLP-Techniken*.
Die Pipeline bereinigt Beschwerdetexte (Clean + spaCy-Lemmatisierung) und erstellt zwei Repräsentationen: TF-IDF und SBERT.
Topic Modeling erfolgt mit NMF, LDA und BERTopic; die Topic-Anzahl K für NMF/LDA wird per Coherence-Scan (c_v) gewählt.

### Outputs
Erzeugt u. a. cleaned_sample.csv, topics_*.csv, prevalence_*.csv/.png, coherence_scan.csv/.png und run_summary.txt.