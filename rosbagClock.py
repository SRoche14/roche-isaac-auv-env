import rosbag

bag_path = "/home/warp/isaacsim4.5/IsaacLab/roche-isaac-auv-env/MITRE/Up/mitreUp6_2025-12-16-15-08-29.bag"
topic_name = "/warpauv_1/control/motor_controller_feather/pwm_command_list"

EPS = 0.08  # treat tiny floating noise as zero; adjust if needed
K = 5  # consecutive non-zero PWM messages to qualify

t0 = None
first_nonzero_rel_t = None
streak = 0

with rosbag.Bag(bag_path) as bag:
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        ts = t.to_sec()
        if t0 is None:
            t0 = ts  # set initial time to 0 at first PWM message

        # msg.motor_commands is a list; each item has position/speed/acceleration
        nonzero = False
        for mc in msg.motor_commands:
            if (abs(mc.position) > EPS):
                nonzero = True
                break

        if nonzero:
            streak += 1
        else:
            streak = 0

        if streak >= K:
            first_nonzero_rel_t = ts - t0
            break

if t0 is None:
    print(f"No messages found on topic {topic_name}")
elif first_nonzero_rel_t is None:
    print(f"Never found {K} consecutive non-zero PWM commands (>|{EPS}|) on {topic_name}")
else:
    print(f"First non-zero PWM time (relative to first PWM msg): {first_nonzero_rel_t:.6f} s")
