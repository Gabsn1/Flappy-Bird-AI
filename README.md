# Flappy-Bird-AI

Dieses Projekt trainiert eine KI, die Flappy Bird selbst spielt. Dafuer wird
ein DQN-Agent (`Deep Q-Network`) verwendet. Die KI entscheidet in jedem Schritt,
ob der Vogel flappen soll oder ob nichts getan wird.

Das Programm nutzt:

- `flappy_bird_gymnasium` als Spielumgebung
- `gymnasium` fuer die Reinforcement-Learning-Umgebung
- `numpy` fuer Zahlen und Arrays
- `tensorflow`/`keras` fuer das neuronale Netz

## Dateien

| Datei | Zweck |
| --- | --- |
| `.venv/flappy_bird.py` | Trainiert die KI. |
| `watch_ai.py` | Laesst ein gespeichertes Modell spielen. |
| `flappy_bird_dqn_v1.keras` | Bereits trainiertes Modell der alten Version. |

## Wie die KI funktioniert

Die Umgebung liefert 12 normalisierte Eingabewerte an das neuronale Netz.
Diese Werte beschreiben den aktuellen Spielzustand:

- Positionen der naechsten Roehren
- y-Position des Vogels
- vertikale Geschwindigkeit des Vogels
- Rotation des Vogels

Das neuronale Netz gibt fuer zwei Aktionen einen Wert aus:

- `0`: nichts tun
- `1`: flappen

Die Aktion mit dem hoeheren Wert wird ausgefuehrt. Beim Training probiert die
KI am Anfang noch viele zufaellige Aktionen aus. Mit der Zeit verlaesst sie
sich immer mehr auf das gelernte Modell.

## Training starten

Im Projektordner ausfuehren:

```powershell
.\.venv\Scripts\python.exe .\.venv\flappy_bird.py
```

Standardmaessig werden 2000 Episoden trainiert. Nach dem Training wird das
Modell gespeichert.

Falls du das Modell unter einem bestimmten Namen speichern moechtest:

```powershell
$env:FLAPPY_MODEL="flappy_bird_dqn_v1.keras"
.\.venv\Scripts\python.exe .\.venv\flappy_bird.py
```

## Trainiertes Modell anschauen

Um die KI spielen zu sehen:

```powershell
.\.venv\Scripts\python.exe .\watch_ai.py
```

Falls du ein bestimmtes Modell laden willst:

```powershell
$env:FLAPPY_MODEL="flappy_bird_dqn_v1.keras"
.\.venv\Scripts\python.exe .\watch_ai.py
```

## Einstellungen

Einige Werte koennen ueber Umgebungsvariablen geaendert werden.

| Variable | Bedeutung |
| --- | --- |
| `FLAPPY_EPISODES` | Anzahl der Trainings-Episoden. |
| `FLAPPY_MODEL` | Modell-Datei, die gespeichert oder geladen wird. |
| `FLAPPY_RENDER_TRAINING` | Zeigt das Spiel waehrend des Trainings, wenn `1`. |
| `FLAPPY_RENDER_DEMO` | Zeigt nach dem Training eine Demo, wenn `1`. |
| `FLAPPY_WATCH_EPISODES` | Anzahl der Runden beim Anschauen. |
| `FLAPPY_RENDER` | Zeigt beim Anschauen das Spielfenster, wenn `1`. |
| `FLAPPY_BATCH_SIZE` | Groesse eines Trainings-Batches. |
| `FLAPPY_MIN_BUFFER` | Anzahl gespeicherter Schritte, bevor trainiert wird. |
| `FLAPPY_MAX_STEPS` | Maximale Schritte pro Episode. |
| `FLAPPY_EPS_START` | Zufallsrate am Anfang des Trainings. |
| `FLAPPY_EPS_MIN` | Minimale Zufallsrate. |
| `FLAPPY_EPS_DECAY_EPISODES` | Wie schnell die Zufallsrate sinkt. |

Beispiel: Nur 500 Episoden trainieren:

```powershell
$env:FLAPPY_EPISODES="500"
.\.venv\Scripts\python.exe .\.venv\flappy_bird.py
```

Beispiel: Beim Anschauen nur 3 Runden spielen:

```powershell
$env:FLAPPY_WATCH_EPISODES="3"
.\.venv\Scripts\python.exe .\watch_ai.py
```

## Hinweis

Die alte Version arbeitet direkt mit den 12 Features aus der Umgebung. Sie ist
einfacher aufgebaut und kann gut funktionieren, wenn das Modell lange genug
trainiert wurde. Bei sehr starken Hoehenwechseln der Roehren kann es trotzdem
passieren, dass der Vogel nicht schnell genug reagiert.
