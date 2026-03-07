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

### Phase 1: v0.1 — Ollama Intent Parser (Weekend 1)

Ship something working fast. Use a small local LLM as the intent parser — no training required.

**Saturday Morning**
- [ ] Flash serial sketch to Sparki
- [ ] Confirm Python bridge works (Sparki beeps on command)
- [ ] `ollama pull phi3:mini` on workstation
- [ ] Write intent parser wrapper

```python
# crasis/intent.py
import ollama
import json

SYSTEM_PROMPT = """
You are a robot command parser for a Sparki robot.
Given a natural language command, respond ONLY with a JSON array 
of commands from this list:
["forward", "backward", "left", "right", "stop", 
 "beep", "grip_open", "grip_close", "sense"]

Example:
Input: "go forward and beep"
Output: ["forward", "beep", "stop"]

Input: "find the light"  
Output: ["sense", "forward"]

Respond with ONLY the JSON array. No explanation.
"""

def parse_intent(natural_language: str) -> list[str]:
    response = ollama.chat(
        model='phi3:mini',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': natural_language}
        ]
    )
    return json.loads(response['message']['content'])
```

**Saturday Afternoon**
- [ ] Wire intent parser to serial bridge
- [ ] Confirm end-to-end: "go forward" → Sparki moves
- [ ] Record demo GIF

**Saturday Evening**
- [ ] Generate synthetic training data (see Phase 2 prep)
- [ ] Set up GitHub repo

**Sunday**
- [ ] Polish README
- [ ] Write the tweet
- [ ] Ship v0.1

---

### Phase 2: v0.2 — Distilled Classifier (Week 2)

Replace the Ollama runtime dependency with a fine-tuned 66MB DistilBERT classifier.  
Train on workstation RTX 4060, deploy weights to laptop.

**Synthetic Data Generation**

```python
# crasis/generate.py
# Run once on workstation — costs ~$2 via API, or use local Ollama

GENERATION_PROMPT = """
Generate 1000 examples of natural language commands a person might 
give to a small mobile robot with a gripper. 

For each command provide:
- The natural language input (varied phrasing, casual speech, typos)
- The sequence of robot primitives to execute

Robot primitives: forward, backward, left, right, stop, 
                  beep, grip_open, grip_close, sense

Output as JSON array:
[
  {"input": "move ahead a bit", "actions": ["forward", "stop"]},
  {"input": "grab that thing", "actions": ["grip_open", "grip_close"]},
  ...
]

Include: casual phrasing, typos, multi-step commands, 
         sensor-driven commands, ambiguous cases.
"""
```

**Fine-tuning Pipeline**

```python
# crasis/train.py
# Runs on workstation with RTX 4060
# Fine-tunes DistilBERT as multi-label classifier
# Input: natural language string
# Output: array of action labels
# Model size: ~66MB
# Training time: ~10 minutes on RTX 4060
# Inference time: <50ms on laptop CPU
```

**Deployment**

```
workstation: crasis/train.py → models/sparki-intent-v1.bin (66MB)
                                        ↓
laptop:      crasis/deploy.py loads weights, runs inference locally
             NO internet required, NO API calls, NO Ollama runtime
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
