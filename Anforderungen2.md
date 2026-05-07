# Projekt: Android Sturzdetektions-App

##  Anforderungen

### 1. Funktionale Anforderungen (Features)
* **F1 - Echtzeit-Sensorik:** Kontinuierliches Auslesen der Beschleunigungswerte (X, Y, Z) über den Android `SensorManager`.
* **F2 - Sturzerkennungs-Algorithmus:** Mathematische Erkennung eines Sturzes (Freier Fall gefolgt von einem Aufprall-Peak).
* **F3 - Lokale Alarmierung:** Akustisches Signal (Alarmton) und Vibration bei erkanntem Sturz.
* **F4 - Fehlalarm-Management:** Ein 10-sekündiger Countdown ermöglicht es dem Nutzer, den Alarm manuell abzubrechen, bevor weitere Schritte eingeleitet werden.
* **F5 - Steuerung:** Start- und Stop-Funktionalität der Überwachung über eine einfache Benutzeroberfläche.
