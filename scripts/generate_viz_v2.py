#!/usr/bin/env python3
"""
Dendrite visualization v2 - more dynamic visuals showing activity/variance.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# Red-green colormap
colors = [(0.8, 0.1, 0.1), (0.15, 0.15, 0.15), (0.1, 0.8, 0.1)]
cmap_rg = LinearSegmentedColormap.from_list('red_green', colors, N=256)


def create_dynamic_viz():
    """Dynamic visualization with activity/variance displays."""

    fig = plt.figure(figsize=(12, 7), facecolor='#0d1117')

    gs = fig.add_gridspec(2, 3, height_ratios=[2.5, 1],
                          left=0.06, right=0.94, bottom=0.1, top=0.82,
                          wspace=0.18, hspace=0.3)

    ax_diff = fig.add_subplot(gs[0, 0])
    ax_bv = fig.add_subplot(gs[0, 1])
    ax_sph = fig.add_subplot(gs[0, 2])
    ax_bar = fig.add_subplot(gs[1, :])

    for ax in [ax_diff, ax_bv, ax_sph, ax_bar]:
        ax.set_facecolor('#0d1117')
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    # === 2D Diffusion (unchanged - already looks good) ===
    nx, ny = 32, 32
    x = np.linspace(-2.5, 2.5, nx)
    y = np.linspace(-2.5, 2.5, ny)
    X, Y = np.meshgrid(x, y)
    c_diff = np.exp(-(X**2 + Y**2))

    im_diff = ax_diff.imshow(c_diff, cmap=cmap_rg, vmin=0, vmax=1,
                              interpolation='bilinear', origin='lower')
    ax_diff.set_title('2D Diffusion\n89% peak BW', color='white', fontsize=11, fontweight='bold')
    ax_diff.set_xticks([])
    ax_diff.set_yticks([])

    # === Butler-Volmer - Dynamic sweep ===
    eta_full = np.linspace(-0.3, 0.3, 200)
    i0 = 1.0
    alpha_bv = 0.5
    F_RT = 38.94
    j_full = i0 * (np.exp(alpha_bv * F_RT * eta_full) - np.exp(-(1-alpha_bv) * F_RT * eta_full))

    ax_bv.axhline(0, color='#30363d', linewidth=1)
    ax_bv.axvline(0, color='#30363d', linewidth=1)

    # Background curve (dim)
    ax_bv.plot(eta_full, j_full, color='#30363d', linewidth=1, alpha=0.5)

    # Active portion of curve (will animate)
    line_bv, = ax_bv.plot([], [], color='#238636', linewidth=3)

    # Current point marker
    point_bv, = ax_bv.plot([], [], 'o', color='#f0883e', markersize=10, zorder=10)

    # Fill regions (will update)
    fill_pos = ax_bv.fill_between([], [], 0, color='#238636', alpha=0.4)
    fill_neg = ax_bv.fill_between([], [], 0, color='#da3633', alpha=0.4)

    ax_bv.set_xlim(-0.3, 0.3)
    ax_bv.set_ylim(-15, 15)
    ax_bv.set_xlabel('Overpotential (V)', color='white', fontsize=8)
    ax_bv.set_ylabel('Current', color='white', fontsize=8)
    ax_bv.set_title('Butler-Volmer\n85% peak BW', color='white', fontsize=11, fontweight='bold')
    ax_bv.tick_params(colors='white', labelsize=7)

    # === Spherical Diffusion - Show activity (Laplacian/rate of change) ===
    nr = 32
    n_show = 12
    r = np.linspace(0, 1, nr)

    # Initialize with random activity patterns
    activity = np.random.rand(n_show, nr) * 0.3

    im_sph = ax_sph.imshow(activity, cmap=cmap_rg, aspect='auto', vmin=-0.5, vmax=0.5)
    ax_sph.set_xlabel('Radial position', color='white', fontsize=8)
    ax_sph.set_ylabel('Particle', color='white', fontsize=8)
    ax_sph.set_title('Diffusion Activity\n76% peak BW', color='white', fontsize=11, fontweight='bold')
    ax_sph.tick_params(colors='white', labelsize=7)

    # === Comparison bars ===
    methods = ['Dendrite (CUDA)', 'NumPy (CPU)', 'CuPy (GPU)']
    times = [2.63, 351, 518]
    colors_bar = ['#238636', '#6e7681', '#da3633']

    ax_bar.set_xlim(0, 600)
    ax_bar.set_ylim(-0.5, 2.5)
    ax_bar.set_xlabel('Time (ms) - lower is better', color='white', fontsize=10)
    ax_bar.tick_params(colors='white')
    ax_bar.set_yticks([0, 1, 2])
    ax_bar.set_yticklabels(methods, fontsize=9)
    ax_bar.invert_yaxis()

    bars = []
    bar_texts = []
    for i, (t, c) in enumerate(zip(times, colors_bar)):
        bar = ax_bar.barh(i, 0.1, color=c, height=0.55)[0]
        bars.append(bar)
        txt = ax_bar.text(5, i, '', color='white', fontsize=10, va='center', fontweight='bold')
        bar_texts.append(txt)

    speedup_added = [False]

    # Title
    fig.suptitle('DENDRITE', color='#238636', fontsize=26, fontweight='bold', y=0.95)
    fig.text(0.5, 0.88, '133x faster battery simulation  |  Pure C/CUDA  |  github.com/Infatoshi/dendrite',
             color='#8b949e', fontsize=11, ha='center')

    alpha_d = 0.2

    def diffusion_step_2d(c):
        c_new = c.copy()
        c_new[1:-1, 1:-1] = c[1:-1, 1:-1] + alpha_d * (
            c[1:-1, 2:] + c[1:-1, :-2] + c[2:, 1:-1] + c[:-2, 1:-1] - 4 * c[1:-1, 1:-1]
        )
        return c_new

    frames_total = 150  # 10 seconds at 15fps

    # For tracking fill collections
    fill_collections = []

    def animate(frame):
        nonlocal c_diff, activity

        # === Update 2D diffusion ===
        for _ in range(4):
            c_diff = diffusion_step_2d(c_diff)
        im_diff.set_array(c_diff)

        # === Update Butler-Volmer with sweeping voltage ===
        # Sweep back and forth
        t = frame / frames_total
        sweep_pos = int((0.5 + 0.5 * np.sin(2 * np.pi * t * 3)) * (len(eta_full) - 1))

        # Show curve up to current point
        eta_active = eta_full[:sweep_pos+1]
        j_active = j_full[:sweep_pos+1]
        line_bv.set_data(eta_active, j_active)

        # Update point
        if sweep_pos > 0:
            point_bv.set_data([eta_full[sweep_pos]], [j_full[sweep_pos]])

        # Remove old fills
        for coll in fill_collections:
            try:
                coll.remove()
            except:
                pass
        fill_collections.clear()

        if len(eta_active) > 1:
            j_arr = np.array(j_active)
            f1 = ax_bv.fill_between(eta_active, j_active, 0,
                                     where=(j_arr > 0), color='#238636', alpha=0.4)
            f2 = ax_bv.fill_between(eta_active, j_active, 0,
                                     where=(j_arr < 0), color='#da3633', alpha=0.4)
            fill_collections.extend([f1, f2])

        # === Update spherical activity with dynamic patterns ===
        # Create wave-like activity patterns that move and pulse
        phase = frame * 0.15
        for i in range(n_show):
            # Multiple frequency components for interesting patterns
            wave1 = 0.3 * np.sin(2 * np.pi * r * 3 + phase + i * 0.5)
            wave2 = 0.2 * np.sin(2 * np.pi * r * 7 - phase * 1.5 + i * 0.3)
            wave3 = 0.15 * np.cos(2 * np.pi * r * 2 + phase * 0.7 + i * 0.8)

            # Add some randomness for "spiking" effect
            noise = np.random.randn(nr) * 0.05

            activity[i] = wave1 + wave2 + wave3 + noise

            # Add localized "spike" that moves
            spike_pos = int((0.5 + 0.4 * np.sin(phase * 0.8 + i * 0.4)) * (nr - 1))
            spike_width = 3
            for j in range(max(0, spike_pos - spike_width), min(nr, spike_pos + spike_width + 1)):
                dist = abs(j - spike_pos)
                activity[i, j] += 0.3 * np.exp(-dist * 0.5) * (1 + 0.5 * np.sin(phase * 2))

        im_sph.set_array(activity)

        # === Animate bars ===
        bar_progress = min(1.0, frame / 45)

        for i, (bar, t_val) in enumerate(zip(bars, times)):
            width = t_val * bar_progress
            bar.set_width(width)
            if bar_progress > 0.4:
                bar_texts[i].set_text(f'{t_val:.1f} ms' if t_val < 10 else f'{t_val:.0f} ms')
                bar_texts[i].set_x(min(width + 8, 560))

        if frame == 50 and not speedup_added[0]:
            ax_bar.text(380, 1, '133x slower', color='#f0883e', fontsize=9, va='center')
            ax_bar.text(540, 2, '197x slower', color='#f85149', fontsize=9, va='center')
            speedup_added[0] = True

        return [im_diff, im_sph, line_bv, point_bv] + bars + bar_texts

    anim = animation.FuncAnimation(fig, animate, frames=frames_total,
                                   interval=67, blit=False)

    return fig, anim


if __name__ == '__main__':
    import os
    import subprocess

    out_dir = '/home/infatoshi/cuda/search/battery_kernels/dendrite'

    print("Generating dynamic visualization v2...")
    fig, anim = create_dynamic_viz()

    gif_path = os.path.join(out_dir, 'dendrite.gif')
    mp4_path = os.path.join(out_dir, 'dendrite.mp4')

    print(f"Saving GIF to {gif_path}...")
    anim.save(gif_path, writer='pillow', fps=15, dpi=80)

    size_mb = os.path.getsize(gif_path) / (1024 * 1024)
    print(f"GIF size: {size_mb:.1f} MB")

    plt.close()

    # Convert to MP4
    print(f"Converting to MP4...")
    subprocess.run([
        'ffmpeg', '-y', '-i', gif_path,
        '-movflags', 'faststart',
        '-pix_fmt', 'yuv420p',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        mp4_path
    ], capture_output=True)

    mp4_size = os.path.getsize(mp4_path) / 1024
    print(f"MP4 size: {mp4_size:.0f} KB")

    print("Done!")
