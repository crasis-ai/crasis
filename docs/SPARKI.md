# Sparki + Crasis: Origin Story & Build Plan

## The Problem

Running a 4GB LLM at 5 tokens/sec to drive a $150 toy robot is insane. The model can write poetry, explain the French Revolution, and debug Python — none of which a robot needs. All Sparki needs is:

- ~15 action primitives
- ~9 sensor values
- Natural language → intent → action

A frontier model is a nuclear weapon for that problem. The entire AI ecosystem has ignored the obvious answer: **tiny, specialized models trained to do one job and do it fast.**

That's what Crasis fixes.

---

## The Hardware

| Machine | Specs | Role |
|---|---|---|
| HP Envy X360 (laptop) | i7, 16GB RAM, 256GB SSD, no GPU | Edge node — inference + Sparki comms |
| Workstation | i9, 32GB RAM, RTX 4060 | Crasis machine — data gen + fine-tuning |

This split IS the architecture. Train on the workstation, deploy to constrained hardware. The same pattern applies to spacecraft, air-gapped defense systems, embedded industrial controllers.

---

## Sparki Hardware Interface

**Robot:** ArcBotics Sparki (Arduino-based)  
**Comms:** USB Serial (115200 baud) or Bluetooth (9600 baud)  
**Library:** SparkiDuino / Sparki.h

### Action Space (~15 primitives)

```cpp
sparki.moveForward()
sparki.moveBackward()
sparki.moveLeft()
sparki.moveRight()
sparki.moveStop()
sparki.gripperOpen()
sparki.gripperClose()
sparki.gripperStop()
sparki.servo(angle)       // SERVO_LEFT, SERVO_CENTER, SERVO_RIGHT
sparki.RGB(color)         // RGB_RED, RGB_GREEN, RGB_BLUE, RGB_OFF
sparki.beep()
sparki.beep(freq, duration)
sparki.print(text)        // LCD
```

### Sensor Space (~9 values)

```cpp
sparki.ping()             // ultrasonic distance in cm (integer)
sparki.edgeLeft()         // line sensor left (boolean)
sparki.lineLeft()         // line sensor (boolean)
sparki.lineCenter()       // line sensor (boolean)
sparki.lineRight()        // line sensor (boolean)
sparki.edgeRight()        // line sensor right (boolean)
sparki.lightLeft()        // ambient light left (integer)
sparki.lightRight()       // ambient light right (integer)
sparki.readIR()           // IR remote code (integer)
```

---

## Step 0: Serial Bridge (Do This First)

Nothing else matters until Python can talk to Sparki.

### Arduino Sketch (flash to Sparki)

```cpp
#include <Sparki.h>

void setup() {
  sparki.begin();
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    if (cmd == "forward")        sparki.moveForward();
    else if (cmd == "backward")  sparki.moveBackward();
    else if (cmd == "left")      sparki.moveLeft();
    else if (cmd == "right")     sparki.moveRight();
    else if (cmd == "stop")      sparki.moveStop();
    else if (cmd == "beep")      sparki.beep();
    else if (cmd == "grip_open") sparki.gripperOpen();
    else if (cmd == "grip_close") sparki.gripperClose();
    else if (cmd == "sense") {
      // Return sensor state as JSON
      Serial.print("{");
      Serial.print("\"ping\":");    Serial.print(sparki.ping());
      Serial.print(",\"light_l\":"); Serial.print(sparki.lightLeft());
      Serial.print(",\"light_r\":"); Serial.print(sparki.lightRight());
      Serial.print(",\"line_l\":"); Serial.print(sparki.lineLeft());
      Serial.print(",\"line_c\":"); Serial.print(sparki.lineCenter());
      Serial.print(",\"line_r\":"); Serial.print(sparki.lineRight());
      Serial.println("}");
    }
  }
}
```

### Python Bridge

```python
# sparki/bridge.py
import serial
import json
import time

class SparkiBridge:
    def __init__(self, port='/dev/ttyUSB0', baud=9600):
        self.ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)  # Arduino reset delay
    
    def send(self, command: str):
        self.ser.write(f"{command}\n".encode())
    
    def sense(self) -> dict:
        self.send("sense")
        response = self.ser.readline().decode().strip()
        return json.loads(response)
    
    def close(self):
        self.ser.close()

# Test it
if __name__ == "__main__":
    sparki = SparkiBridge(port='COM3')  # Windows: COM3, Linux: /dev/ttyUSB0
    sparki.send("beep")
    time.sleep(1)
    sparki.send("forward")
    time.sleep(1)
    sparki.send("stop")
    print(sparki.sense())
    sparki.close()
```

**Confirm this works before moving to anything else.**

---

## Build Plan

### Phase 1: v0.1 — Crasis Intent Specialist (Weekend 1)

Generate synthetic training data for natural language → robot command mapping,
train a BERT-Tiny ONNX specialist on the workstation, deploy to the laptop.

**Saturday Morning**
- [ ] Flash serial sketch to Sparki
- [ ] Confirm Python bridge works (Sparki beeps on command)
- [ ] Write `specialists/sparki-intent/spec.yaml`

**Saturday Afternoon**
- [ ] `crasis build --spec specialists/sparki-intent/spec.yaml`
- [ ] Wire specialist to serial bridge
- [ ] Confirm end-to-end: "go forward" → Sparki moves
- [ ] Record demo GIF

**Saturday Evening**
- [ ] Deploy ONNX to laptop (`crasis pull` or `scp`)
- [ ] Confirm inference works on laptop CPU with no GPU

**Sunday**
- [ ] Polish README
- [ ] Write the tweet
- [ ] Ship v0.1

---

### Phase 2: v0.2 — Multi-label Action Sequencing (Week 2)

Extend the specialist to handle multi-step commands ("go forward then beep").
The Phase 1 specialist maps one input to one action class. Phase 2 extends the
spec to multiclass or sequence output — still BERT-Tiny, still ONNX, still local.

**Synthetic Data Generation**

```python
# crasis generate handles this via spec.yaml
# Run once on workstation — costs ~$2 via OpenRouter
# Output: data/sparki-intent/train.jsonl
```

**Deployment**

```
workstation: crasis build → models/sparki-intent-onnx/ (~4.3MB ONNX)
                                        ↓
laptop:      Specialist.load() → classify() → serial bridge → Sparki
             NO internet required, NO API calls, NO cloud runtime
```

---

### Phase 3: Reactive Agent (Week 3+)

Add sensor feedback loop — the model reads sensor state and adjusts behavior.

```
[user intent] → intent model → initial action
     ↑                              ↓
[sensor state] ←← Sparki ←← execute action
     ↓
[reactive policy model] → corrective action
```

This is where it gets interesting — a second tiny model trained on 
(sensor_state, intent) → corrective_action mappings.

**Example behaviors that emerge:**
- "find the light" → move forward, continuously read lightLeft/lightRight, 
   steer toward brighter side
- "don't hit anything" → move forward, ping() every loop, stop if < 10cm
- "follow the line" → read lineLeft/lineCenter/lineRight, 
   correct steering continuously

---

## Repo Structure

```
crasis/
├── README.md
├── SPARKI.md              ← this file
├── crasis/
│   ├── factory.py         ← synthetic training data generation
│   ├── train.py           ← fine-tune pipeline (runs on GPU workstation)
│   ├── deploy.py          ← inference wrapper (runs on edge hardware)
│   └── intent.py          ← v0.1 ollama intent parser
├── scripts/sparki/
│   ├── bridge.py          ← Python serial bridge
│   ├── commands.py        ← action primitives + sensor reader
│   └── sketches/
│       └── serial_bridge.ino  ← Arduino sketch for Sparki
├── data/
│   └── synthetic/         ← generated training data
├── models/
│   └── .gitkeep           ← weights go here, not in git
└── demo/
    └── sparki_demo.gif    ← the thing that gets retweeted
```

---

## The Bigger Vision (Crasis Framework)

Sparki is demo target #1. The framework generalizes to any constrained system:

| Target | Action Space | Constraint |
|---|---|---|
| Sparki robot | 15 primitives | Old laptop, no GPU |
| Spacecraft fault detection | Fault taxonomy | Air-gapped, on-orbit |
| Industrial controller | PLC command set | Edge hardware, no internet |
| Medical device | Clinical decision tree | HIPAA, no cloud |
| Defense system | Mission primitives | ITAR, air-gapped |

**The pattern is always the same:**
1. Describe problem domain to frontier model
2. Generate synthetic training data
3. Fine-tune tiny specialist model on GPU workstation
4. Deploy 50-200MB weights to constrained target
5. Runtime has zero frontier model dependency

This is the microservices pattern applied to AI.  
Frontier model = architect. Tiny specialist = worker.  
You only need the architect once.

---

## The Launch Tweet

> "Ran a 4GB LLM at 5 tok/sec trying to drive a $150 robot. That's insane.
>
> So I built Crasis — describe your problem to a frontier model, get a tiny
> specialist that does one job fast.
> 
> 50MB model. Natural language → Sparki robot. Runs on a 10yr old laptop.
> 
> [GIF]
> 
> github: [link]"

---

## Status

- [ ] Step 0: Serial bridge working
- [ ] v0.1: Ollama intent parser end-to-end
- [ ] Demo GIF captured
- [ ] GitHub repo published
- [ ] v0.2: DistilBERT classifier trained + deployed
- [ ] Reactive sensor loop
- [ ] Framework generalized beyond Sparki
