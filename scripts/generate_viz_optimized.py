#!/usr/bin/env python3
"""
Optimized visualization GIF for Dendrite - smaller file size for GitHub/X.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# Create red-green colormap
colors = [(0.8, 0.1, 0.1), (0.15, 0.15, 0.15), (0.1, 0.8, 0.1)]
cmap_rg = LinearSegmentedColormap.from_list('red_green', colors, N=256)


def create_optimized_viz():
    """Optimized combined visualization - smaller file size."""

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

    # === 2D Diffusion (smaller grid) ===
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

    # === Butler-Volmer ===
    eta = np.linspace(-0.3, 0.3, 80)
    i0 = 1.0
    alpha_bv = 0.5
    F_RT = 38.94
    j_bv = i0 * (np.exp(alpha_bv * F_RT * eta) - np.exp(-(1-alpha_bv) * F_RT * eta))

    ax_bv.axhline(0, color='#30363d', linewidth=1)
    ax_bv.axvline(0, color='#30363d', linewidth=1)
    ax_bv.plot(eta, j_bv, color='#238636', linewidth=2)
    ax_bv.fill_between(eta, j_bv, 0, where=(j_bv > 0), color='#238636', alpha=0.3)
    ax_bv.fill_between(eta, j_bv, 0, where=(j_bv < 0), color='#da3633', alpha=0.3)
    ax_bv.set_xlim(-0.3, 0.3)
    ax_bv.set_ylim(-15, 15)
    ax_bv.set_xlabel('Overpotential (V)', color='white', fontsize=8)
    ax_bv.set_ylabel('Current', color='white', fontsize=8)
    ax_bv.set_title('Butler-Volmer\n85% peak BW', color='white', fontsize=11, fontweight='bold')
    ax_bv.tick_params(colors='white', labelsize=7)

    # === Spherical Diffusion ===
    nr = 32
    n_show = 8
    r = np.linspace(0, 1, nr)
    particles = np.array([0.5 + 0.4 * np.cos(np.pi * r + i * 0.5) for i in range(n_show)])

    im_sph = ax_sph.imshow(particles, cmap=cmap_rg, aspect='auto', vmin=0, vmax=1)
    ax_sph.set_xlabel('Radial pos (r/R)', color='white', fontsize=8)
    ax_sph.set_ylabel('Particle', color='white', fontsize=8)
    ax_sph.set_title('Spherical Diffusion\n76% peak BW', color='white', fontsize=11, fontweight='bold')
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

    def spherical_step(p):
        p_new = p.copy()
        for i in range(1, nr - 1):
            p_new[:, i] = p[:, i] + 0.15 * ((p[:, i+1] - p[:, i]) - (p[:, i] - p[:, i-1]))
        p_new[:, -1] = p_new[:, -2] - 0.003
        p_new[:, 0] = p_new[:, 1]
        return np.clip(p_new, 0, 1)

    # 150 frames at 15fps = 10 seconds
    frames_total = 150

    def animate(frame):
        nonlocal c_diff, particles

        # Update diffusion (more steps per frame for visible change)
        for _ in range(4):
            c_diff = diffusion_step_2d(c_diff)
        im_diff.set_array(c_diff)

        # Update spherical
        for _ in range(5):
            particles = spherical_step(particles)
        im_sph.set_array(particles)

        # Animate bars (first 45 frames = 3 seconds)
        bar_progress = min(1.0, frame / 45)

        for i, (bar, t) in enumerate(zip(bars, times)):
            width = t * bar_progress
            bar.set_width(width)
            if bar_progress > 0.4:
                bar_texts[i].set_text(f'{t:.1f} ms' if t < 10 else f'{t:.0f} ms')
                bar_texts[i].set_x(min(width + 8, 560))

        # Add speedup labels
        if frame == 50 and not speedup_added[0]:
            ax_bar.text(380, 1, '133x slower', color='#f0883e', fontsize=9, va='center')
            ax_bar.text(540, 2, '197x slower', color='#f85149', fontsize=9, va='center')
            speedup_added[0] = True

        return [im_diff, im_sph] + bars + bar_texts

    anim = animation.FuncAnimation(fig, animate, frames=frames_total,
                                   interval=67, blit=False)  # ~15fps

    return fig, anim


if __name__ == '__main__':
    import os

    out_dir = '/home/infatoshi/cuda/search/battery_kernels/dendrite'

    print("Generating optimized visualization...")
    fig, anim = create_optimized_viz()

    gif_path = os.path.join(out_dir, 'dendrite.gif')
    print(f"Saving to {gif_path}...")

    # Lower dpi and fps for smaller file
    anim.save(gif_path, writer='pillow', fps=15, dpi=80)
    print(f"Saved: {gif_path}")

    # Static fallback
    fig.savefig(os.path.join(out_dir, 'dendrite_static.png'),
                facecolor='#0d1117', dpi=120, bbox_inches='tight')

    plt.close()

    # Check file size
    size_mb = os.path.getsize(gif_path) / (1024 * 1024)
    print(f"GIF size: {size_mb:.1f} MB")
