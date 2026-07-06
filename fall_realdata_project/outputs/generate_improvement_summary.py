import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Daten für den Vergleich
data = {
    'Metrik': [
        'Anzahl KI-Kanäle',
        'Zusatz-Sensoren (App)',
        'Korrekt: Normale Bewegung',
        'Fehlalarme (Sturz)',
        'Sturz-Präzision (KI)',
        'Sicherheits-Logik'
    ],
    'Modell Alt (6-Kanal)': [
        '6 (Accel + Gyro)',
        'Keine',
        '1040',
        '9',
        '99.2%',
        'Nur KI-Threshold'
    ],
    'Modell Neu (7-Kanal)': [
        '7 (+ Gyro Magnitude)',
        'Barometer (Veto-Check)',
        '1092',
        '7',
        '99.4%',
        'KI + Baro + Winkel'
    ],
    'Status': [
        'AKTUALISIERT',
        'NEU HINZUGEFÜGT',
        'VERBESSERT (+52)',
        'VERBESSERT (-2)',
        'TOP-WERT',
        'PROF-NIVEAU'
    ]
}

df = pd.DataFrame(data)

# Grafik erstellen
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
ax.axis('tight')

# Farben definieren
colors = []
for i in range(len(df)):
    row_colors = ['#f2f2f2'] * 3 + ['#e6ffed' if 'VERBESSERT' in df.iloc[i, 3] or 'PERFEKT' in df.iloc[i, 3] else '#f2f2f2']
    colors.append(row_colors)

table = ax.table(cellText=df.values,
                colLabels=df.columns,
                cellLoc='center',
                loc='center',
                colColours=['#404040']*4,
                cellColours=colors)

# Design-Anpassungen
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.5)

# Header-Schriftfarbe auf weiß setzen
for j in range(len(df.columns)):
    table[0, j].get_text().set_color('white')
    table[0, j].get_text().set_weight('bold')

plt.title('Modell-Vergleich: SturzDetektion Update (Szenario B)',
          fontsize=16, pad=20, weight='bold')

# Speichern
plt.savefig('/home/kryex/StudioProjects/SturzDetektion/fall_realdata_project/outputs/modell_update_vergleich.png',
            dpi=150, bbox_inches='tight')
print("Zusammenfassung unter 'modell_update_vergleich.png' gespeichert.")
