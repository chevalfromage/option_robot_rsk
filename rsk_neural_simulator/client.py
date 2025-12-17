import rsk
import time
import numpy as np
import json

DT = 1/30  # 30 FPS

targets = [(0.3, 0.3, 0.0), (0.3, -0.3, np.pi/2), (-0.3, -0.3, 0.0), (-0.3, 0.3, -np.pi/2)]
duration = 15  # [s]

with rsk.Client(host="127.0.0.1") as client:
    robot = client.green1

    # Placing the robot in initial position
    robot.goto(targets[0])
    time.sleep(4)

    # Callback to record states
    states = []
    t_start = time.monotonic()

    def get_state(client, dt):
        state = {"timestamp": time.monotonic() - t_start,
                "robot_pose": list(robot.pose),
                "ball_position": list(client.ball)}
        states.append(state)
    client.on_update = get_state

    # Main loop to move the robot between targets
    t = 0
    i = 0
    commands = []
    while t < duration:
        if np.linalg.norm(np.array(robot.pose) - np.array(targets[i])) < 0.05:
            i = (i + 1) % len(targets)
        
        arrived, orders = robot.goto_compute_order(targets[i])
        robot.control(*orders)
        commands.append({"timestamp": time.monotonic() - t_start,
                        "orders": orders})
        
        
        t += DT
        while time.monotonic() - t_start < t:
            time.sleep(1e-3)

    robot.control(0, 0, 0)

    # Save recorded data
    with open("data.json", "w") as f:
        json.dump({"states": states, "commands": commands}, f, indent=4)
    print("Data saved to data.json")