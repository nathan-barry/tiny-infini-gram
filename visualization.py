#!/usr/bin/env python3
"""
Side-by-side visualization of infini-gram (Go) vs GPT (Python) text generation.
Displays both outputs with animation speeds proportional to their actual generation times.
"""

import os
import subprocess
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

# Import from gpt.py
from gpt import Model, data, decode, device, block_size

# Constants
TARGET_CHARS = 1000
PROMPT_LEN = 14  # "First Citizen:" is 14 chars
TEMP = 0.8


def generate_infinigram(target_chars: int) -> tuple[str, float]:
    """Run infini-gram.go and return (output, generation_time)."""
    import re

    result = subprocess.run(
        ["go", "run", "infini-gram.go"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    output = result.stdout
    lines = output.split("\n")

    # Parse generation time from output (e.g., "Generated 1000 chars in 0.0123s")
    elapsed = 0.0
    text_lines = []
    for line in lines:
        if line.startswith("Generated"):
            match = re.search(r"in ([\d.]+)s", line)
            if match:
                elapsed = float(match.group(1))
            break
        if line.startswith("  Level"):
            break
        text_lines.append(line)

    text = "\n".join(text_lines).strip()
    return text, elapsed


@torch.no_grad()
def generate_gpt(target_chars: int) -> tuple[str, float]:
    """Run GPT generation and return (output, generation_time)."""
    weights_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "weights/gpt.pt"
    )

    model = Model()
    model = model.to(device)
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.eval()

    x = data[:PROMPT_LEN].unsqueeze(0).to(device)

    start = time.perf_counter()
    for _ in range(target_chars - PROMPT_LEN):
        cur_context = x[:, -block_size:]
        logits, _ = model(cur_context)
        logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits / TEMP, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)
    elapsed = time.perf_counter() - start

    return decode(x[0].tolist()), elapsed


def split_into_words(text: str) -> list[str]:
    """Split text into words, preserving whitespace."""
    words = []
    current_word = ""

    for char in text:
        if char in " \n\t":
            if current_word:
                words.append(current_word)
                current_word = ""
            words.append(char)
        else:
            current_word += char

    if current_word:
        words.append(current_word)

    return words


def wrap_text(text: str, max_chars_per_line: int = 50) -> str:
    """Wrap text to fit within a certain width."""
    lines = []
    current_line = ""

    for char in text:
        if char == "\n":
            lines.append(current_line)
            current_line = ""
        elif len(current_line) >= max_chars_per_line and char == " ":
            lines.append(current_line)
            current_line = ""
        else:
            current_line += char

    if current_line:
        lines.append(current_line)

    return "\n".join(lines)


def animate_generation(
    left_output: str,
    right_output: str,
    left_time: float,
    right_time: float,
    left_title: str = "Infini-gram",
    right_title: str = "NanoGPT",
):
    """Animate both outputs word by word using matplotlib."""
    left_words = split_into_words(left_output)
    right_words = split_into_words(right_output)

    # Timing: 2 seconds normal speed, then 10x speed for the rest
    normal_duration = 2.0
    speedup_factor = 10.0
    max_time = max(left_time, right_time)

    # At normal speed, we'd need max_time seconds of animation
    # After 2s normal, remaining (max_time - 2) is shown at 10x = (max_time - 2) / 10
    # Total = 2 + (max_time - 2) / 10

    # Calculate time per word at 1x speed (relative to max_time)
    left_time_per_word = (left_time / len(left_words)) if left_words else 0
    right_time_per_word = (right_time / len(right_words)) if right_words else 0

    # Set up the figure
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 10))

    for ax in (ax_left, ax_right):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax_left.set_title(
        f"{left_title}\n(Generated in {left_time:.2f}s)", fontsize=12, fontweight="bold"
    )
    ax_right.set_title(
        f"{right_title}\n(Generated in {right_time:.2f}s)",
        fontsize=12,
        fontweight="bold",
    )

    # Text objects
    left_text_obj = ax_left.text(
        0.02,
        0.98,
        "",
        transform=ax_left.transAxes,
        fontsize=9,
        fontfamily="monospace",
        verticalalignment="top",
        horizontalalignment="left",
        wrap=True,
    )
    right_text_obj = ax_right.text(
        0.02,
        0.98,
        "",
        transform=ax_right.transAxes,
        fontsize=9,
        fontfamily="monospace",
        verticalalignment="top",
        horizontalalignment="left",
        wrap=True,
    )

    # Speed indicator text (centered at bottom of figure)
    speed_text = fig.text(
        0.5,
        0.02,
        "",
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        color="red",
    )

    # Add boxes around text areas
    for ax in (ax_left, ax_right):
        ax.add_patch(
            plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="gray", linewidth=1)
        )

    # State for animation
    pause_duration = 2.0  # seconds to pause before restarting
    state = {
        "left_text": "",
        "right_text": "",
        "left_idx": 0,
        "right_idx": 0,
        "left_sim_time": 0.0,  # Simulated time in the generation
        "right_sim_time": 0.0,
        "start_time": None,
        "last_real_time": 0.0,
        "speedup_active": False,
        "finished": False,
        "finish_time": None,
    }

    def reset_state():
        state["left_text"] = ""
        state["right_text"] = ""
        state["left_idx"] = 0
        state["right_idx"] = 0
        state["left_sim_time"] = 0.0
        state["right_sim_time"] = 0.0
        state["start_time"] = None
        state["last_real_time"] = 0.0
        state["speedup_active"] = False
        state["finished"] = False
        state["finish_time"] = None

    def init():
        left_text_obj.set_text("")
        right_text_obj.set_text("")
        speed_text.set_text("")
        return left_text_obj, right_text_obj, speed_text

    def update(frame):
        # Handle pause and restart
        if state["finished"]:
            if time.perf_counter() - state["finish_time"] >= pause_duration:
                reset_state()
            else:
                return left_text_obj, right_text_obj, speed_text

        if state["start_time"] is None:
            state["start_time"] = time.perf_counter()
            state["last_real_time"] = 0.0

        real_time = time.perf_counter() - state["start_time"]
        delta_real = real_time - state["last_real_time"]
        state["last_real_time"] = real_time

        # Determine speed multiplier based on real time elapsed
        if real_time <= normal_duration:
            speed = 1.0
            if state["speedup_active"]:
                state["speedup_active"] = False
                speed_text.set_text("")
        else:
            speed = speedup_factor
            if not state["speedup_active"]:
                state["speedup_active"] = True
                speed_text.set_text(">>> 10x SPEED >>>")

        # Advance simulated time
        delta_sim = delta_real * speed
        state["left_sim_time"] += delta_sim
        state["right_sim_time"] += delta_sim

        # Add left words that are due based on simulated time
        while state["left_idx"] < len(left_words):
            word_time = state["left_idx"] * left_time_per_word
            if state["left_sim_time"] >= word_time:
                state["left_text"] += left_words[state["left_idx"]]
                state["left_idx"] += 1
            else:
                break

        # Add right words that are due based on simulated time
        while state["right_idx"] < len(right_words):
            word_time = state["right_idx"] * right_time_per_word
            if state["right_sim_time"] >= word_time:
                state["right_text"] += right_words[state["right_idx"]]
                state["right_idx"] += 1
            else:
                break

        # Update text displays
        left_text_obj.set_text(wrap_text(state["left_text"], 55))
        right_text_obj.set_text(wrap_text(state["right_text"], 55))

        # Mark finished when both are done
        if state["left_idx"] >= len(left_words) and state["right_idx"] >= len(
            right_words
        ):
            if not state["finished"]:
                state["finished"] = True
                state["finish_time"] = time.perf_counter()
                speed_text.set_text("")

        return left_text_obj, right_text_obj, speed_text

    # Create animation (interval in ms, blit=False needed for fig.text)
    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=None,
        interval=33,
        blit=False,
        cache_frame_data=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.08)
    plt.show()


def main():
    print("Generating text with both models...")
    print("This may take a moment...\n")

    print("Running infini-gram (Go)...")
    infinigram_output, infinigram_time = generate_infinigram(TARGET_CHARS)
    print(f"  Done in {infinigram_time:.2f}s ({len(infinigram_output)} chars)")

    print("Running nanoGPT (Python)...")
    gpt_output, gpt_time = generate_gpt(TARGET_CHARS)
    print(f"  Done in {gpt_time:.2f}s ({len(gpt_output)} chars)")

    print("\nLaunching matplotlib visualization...")
    print("(Animation speed is relative to actual generation times)")

    animate_generation(
        infinigram_output,
        gpt_output,
        infinigram_time,
        gpt_time,
    )


if __name__ == "__main__":
    main()
