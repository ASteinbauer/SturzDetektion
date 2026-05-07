#  Prioritätenliste – Fall Detection App (Android / Java)

## PRIORITÄT 1 – MUSS funktionieren
Diese Punkte zuerst machen.

### 1. Projekt erstellen
- [ ] Android Studio Projekt erstellen
- [ ] Java auswählen
- [ ] App auf Handy/Emulator starten

---

### 2. UI erstellen
- [ ] Startscreen bauen
- [ ] Start-Button hinzufügen
- [ ] Status-Text anzeigen

Beispiel:
- „Überwachung aus“
- „Überwachung aktiv“

---

### 3. Accelerometer verbinden
- [ ] SensorManager einbauen
- [ ] Accelerometer initialisieren
- [ ] Sensorwerte auslesen

Ziel:
- Werte im Log anzeigen

---

### 4. Sturz erkennen
- [ ] Beschleunigung berechnen
- [ ] Grenzwert definieren
- [ ] Bei hohem Wert Alarm auslösen

Formel:

a = √(x² + y² + z²)

---

### 5. Alarm anzeigen
- [ ] Toast anzeigen
- [ ] Text ändern:
  - „STURZ ERKANNT“
- [ ] Optional: Vibration

---

#  PRIORITÄT 2 – WICHTIG
Nach den Grundfunktionen.

### 6. Countdown
- [ ] 10 Sekunden Timer
- [ ] Abbrechen-Button
- [ ] Ablauf simulieren

---

### 7. Notrufscreen
- [ ] Meldung anzeigen:
  - „Notruf würde gestartet werden“

Optional:
- Telefonnummer öffnen

---

#  PRIORITÄT 3 – DOKUMENTATION
Für Präsentation und Bewertung.

### 8. Architekturdiagramm erstellen
- [ ] MainActivity
- [ ] SensorManager
- [ ] FallDetector
- [ ] AlertSystem

---

### 9. Präsentation vorbereiten
- [ ] Problem erklären
- [ ] Lösung erklären
- [ ] Technologien nennen
- [ ] Demo vorbereiten

---

#  OPTIONAL – NUR WENN ZEIT ÜBRIG
Nicht wichtig für MVP.

- [ ] Schönere UI
- [ ] Soundeffekte
- [ ] Hintergrundservice
- [ ] Smartwatch
- [ ] SMS
- [ ] GPS

---

# ⏱ Zeitplan (6 Stunden)

## Stunde 1
- Projekt
- UI

## Stunde 2
- Sensoren

## Stunde 3
- Fall Detection

## Stunde 4
- Alarm

## Stunde 5
- Countdown + Notrufscreen

## Stunde 6
- Architekturdiagramm
- Präsentation
- Tests

---

#  MVP Ziel

Die App soll:
1. Bewegung messen
2. Sturz erkennen
3. Alarm anzeigen
4. Countdown starten

Das reicht komplett für ein gutes Kurzprojekt.
