#  User documentation

## Purpose

This documentation is intended for users who want to **run the Robot Soccer Kit simulator with neural dynamics enabled**, without modifying the codebase.

It targets students and researchers who use the simulator as a tool (e.g. for control or reinforcement learning), not developers extending the neural models.

For technical details, refer to the
[Developer documentation](../developer).

---

## What does this project provide?

A version of the Robot Soccer Kit simulator where **robot motion dynamics are driven by a neural model trained from real-world data**, instead of a purely analytical model.

From a user perspective:
- The standard RSK simulator is used
- Dynamics are closer to real robot behavior
- No changes to existing strategies or control code are required

---

## Getting started

Installation and launch instructions are provided in the **Quickstart section of the main README**:

[`README.md` at the repository root](../../README.md)

---

## Running a simulation

Once installed, run:

```bash
game_controller --simulated
```

If a trained neural model is available, it is automatically used by the simulator.

---

## Known limitations

* Ball dynamics are not learned
* Robot–robot collisions are not learned
* Performance depends on the training data coverage
