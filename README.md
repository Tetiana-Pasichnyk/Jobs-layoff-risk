# Jobs Layoff Risk

Analyse des Einflusses von KI auf das Entlassungsrisiko am Arbeitsmarkt.

## Projektstruktur

```
Jobs-layoff-risk/
├── data/                          # CSV-Datensatz und E-Mail-Beispiele
├── output_analysis/               # Generierte Grafiken
├── src/
│   ├── paths.py                   # Zentrale Pfade (Projektroot, data/, output/)
│   ├── database/                  # MySQL-Schema, Datenbereinigung, DB-Import
│   ├── statistics/                # EDA, statistische Tests, Visualisierung
│   └── ml_model/                  # Decision Tree, Naive Bayes, LLM-E-Mail-Analyse
├── docs/
│   └── ML_UND_LLM.md              # Beschreibung der ML- und LLM-Module
├── requirements.txt
└── .env.example
```

Ausführliche Dokumentation zu Decision Tree, Naive Bayes und LLM-E-Mail-Analyse: [docs/ML_UND_LLM.md](docs/ML_UND_LLM.md)

## Erstes Setup (einmalig)

### 1. Ins Projektverzeichnis wechseln

**Alle Befehle müssen vom Projektroot ausgeführt werden** — nicht aus `src/`, `src/statistics/` oder anderen Unterordnern.

```bash
cd /Users/tatanapasecnik/Desktop/Abschlussproject/Jobs-layoff-risk
```

Prüfen, ob Sie im richtigen Ordner sind:

```bash
pwd
```

Erwartete Ausgabe:

```
/Users/tatanapasecnik/Desktop/Abschlussproject/Jobs-layoff-risk
```

Die Eingabezeile im Terminal sollte mit `Jobs-layoff-risk %` enden — **nicht** mit `statistics %` oder `database %`.

### 2. Virtuelle Umgebung und Abhängigkeiten

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 3. Umgebungsvariablen setzen

Variablen aus `.env` werden nicht automatisch geladen. Entweder manuell exportieren:

```bash
export DB_PASSWORD=root
export DB_PORT=8889
export HUGGINGFACE_TOKEN=ihr_token   # nur für E-Mail-Vorhersage nötig
```

Oder die Werte direkt in `.env` eintragen und vor jedem Start exportieren:

```bash
export $(grep -v '^#' .env | xargs)
```

### 4. MySQL-Datenbank anlegen

MySQL muss laufen. Schema erstellen (löscht vorhandene DB `AI_Impact_DB` und legt sie neu an):

```bash
mysql -u root -p < src/database/sql-ai-impact-jobs-layoff-risk.sql
```

---

## Ausführen — Schritt für Schritt

**Wichtig:** Vor jedem Befehl ins Projektroot wechseln:

```bash
cd /Users/tatanapasecnik/Desktop/Abschlussproject/Jobs-layoff-risk
```

### Empfohlene Reihenfolge

| Schritt | Befehl | MySQL nötig? | Beschreibung |
|---------|--------|--------------|--------------|
| 1 | `python -m src.database.data_clean` | Ja | CSV bereinigen und in MySQL laden |
| 2 | `python -m src.statistics.run_statistics` | Nein | Statistische Analyse + Grafiken |
| 3 | `python -m src.ml_model.DecisionTree_VS_NaiveBayes` | Ja | Decision Tree vs. Naive Bayes |
| 4 | `python -m src.ml_model.DecisionTree_Predictor` | Ja + HF-Token | Vorhersage aus E-Mail |
| 5 | `python -m src.ml_model.decision_tree_visualization` | Ja | Decision Tree grafisch darstellen |

---

### 1. Datenbank — CSV in MySQL laden

**Voraussetzungen:** MySQL läuft, Schema wurde angelegt (siehe Setup), `DB_PASSWORD` und `DB_PORT` sind gesetzt.

```bash
cd /Users/tatanapasecnik/Desktop/Abschlussproject/Jobs-layoff-risk
python -m src.database.data_clean
```

**Was passiert:** Liest `data/rohdaten-ai-impact-jobs-layoff-risk-dataset.csv`, bereinigt die Daten und schreibt sie in die Tabellen `employees` und `ai_impact_metrics`.

**Erfolg:** Meldung `SUCCESS: Inserted … clean records into MySQL.`

**Hinweis:** Jeder erneute Lauf fügt Datensätze erneut ein. Vor einem zweiten Import die Tabellen leeren oder die DB neu anlegen.

---

### 2. Statistik — EDA, Tests und Grafiken

**Voraussetzungen:** Nur die CSV-Datei in `data/` — **keine MySQL-Verbindung nötig**.

```bash
cd /Users/tatanapasecnik/Desktop/Abschlussproject/Jobs-layoff-risk
python -m src.statistics.run_statistics
```

**Was passiert:** Deskriptive Statistik, Ausreißererkennung, Korrelation, Chi²-Tests, Shapiro-/Z-Tests; anschließend drei Grafiken.

**Ergebnis:** Dateien in `output_analysis/`:

- `scatter_plot.png`
- `correlation_heatmap.png`
- `chi2_barplot.png`

---

### 3. ML-Modelle — Vergleich Decision Tree vs. Naive Bayes

**Voraussetzungen:** MySQL mit geladenen Daten (Schritt 1).

```bash
cd /Users/tatanapasecnik/Desktop/Abschlussproject/Jobs-layoff-risk
python -m src.ml_model.DecisionTree_VS_NaiveBayes
```

**Was passiert:** Lädt Daten aus MySQL, trainiert beide Modelle mit verschiedenen Feature-Kombinationen und gibt Accuracy-Werte sowie Beispiel-Vorhersagen aus.

---

### 4. ML-Modell — Vorhersage aus E-Mail (LLM + Decision Tree)

**Voraussetzungen:** MySQL mit geladenen Daten, gültiger `HUGGINGFACE_TOKEN`.

```bash
cd /Users/tatanapasecnik/Desktop/Abschlussproject/Jobs-layoff-risk
export HUGGINGFACE_TOKEN=ihr_token
python -m src.ml_model.DecisionTree_Predictor
```

**Was passiert:** Llama extrahiert Felder aus `data/email_sample02.txt`, das Decision-Tree-Modell sagt das Entlassungsrisiko voraus.

Standard-E-Mail ändern: in `DecisionTree_Predictor.py` den Parameter `email_filename` anpassen oder `main("email_sample01.txt")` aufrufen.

---

### 5. ML-Modell — Decision Tree visualisieren

**Voraussetzungen:** MySQL mit geladenen Daten.

```bash
cd /Users/tatanapasecnik/Desktop/Abschlussproject/Jobs-layoff-risk
python -m src.ml_model.decision_tree_visualization
```

**Was passiert:** Trainiert das Modell und öffnet ein Fenster mit der Baumstruktur.

---

## Alle Befehle auf einen Blick

Vom Projektroot aus:

```bash
# Datenbank
python -m src.database.data_clean

# Statistik
python -m src.statistics.run_statistics

# ML
python -m src.ml_model.DecisionTree_VS_NaiveBayes
python -m src.ml_model.DecisionTree_Predictor
python -m src.ml_model.decision_tree_visualization
```

---

## Fehlerbehebung

### `ModuleNotFoundError: No module named 'src'`

Sie befinden sich nicht im Projektroot. Lösung:

```bash
cd /Users/tatanapasecnik/Desktop/Abschlussproject/Jobs-layoff-risk
python -m src.statistics.run_statistics
```

Nicht aus `src/statistics/` oder anderen Unterordnern starten.

### `[CONNECTION ERROR]: Failed to connect to MySQL`

- MySQL-Server starten
- `DB_PORT` und `DB_PASSWORD` prüfen (Standard: Port `8889`, Passwort `root`)
- Schema mit `mysql -u root -p < src/database/sql-ai-impact-jobs-layoff-risk.sql` anlegen

### ML-Befehle liefern leere oder fehlerhafte Ergebnisse

Zuerst `python -m src.database.data_clean` ausführen, damit die Tabellen gefüllt sind.

---

## Umgebungsvariablen

| Variable | Beschreibung | Standard |
|----------|--------------|----------|
| `DB_PASSWORD` | MySQL-Passwort | `root` |
| `DB_PORT` | MySQL-Port | `8889` |
| `HUGGINGFACE_TOKEN` | Token für LLM-E-Mail-Analyse | — |
