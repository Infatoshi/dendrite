#!/usr/bin/env python3
"""
Generate visualization GIF for Dendrite - GPU battery simulation kernels.
Shows 2D diffusion with red (negative) / green (positive) color scheme.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patches as mpatches

# Create red-green colormap (red = low, green = high)
colors = [(0.8, 0.1, 0.1), (0.2, 0.2, 0.2), (0.1, 0.8, 0.1)]  # red -> dark gray -> green
cmap_rg = LinearSegmentedColormap.from_list('red_green', colors, N=256)

def create_diffusion_viz():
    """Create 2D diffusion visualization with stencil highlight."""

    # Grid setup
    nx, ny = 64, 64
    dx = dy = 1.0
    D = 1.0
    dt = 0.1
    alpha = D * dt / (dx * dx)

    # Initial condition: Gaussian pulse
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x, y)
    c = np.exp(-(X**2 + Y**2))

    # Normalize to [-1, 1] for red-green viz
    c = 2 * c - 0.5

    fig = plt.figure(figsize=(16, 9), facecolor='#0d1117')

    # Layout: main diffusion grid on left, info panel on right
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1],
                          left=0.05, right=0.95, bottom=0.1, top=0.9,
                          wspace=0.3, hspace=0.3)

    ax_main = fig.add_subplot(gs[:, 0])
    ax_stencil = fig.add_subplot(gs[0, 1])
    ax_compare = fig.add_subplot(gs[0, 2])
    ax_info = fig.add_subplot(gs[1, 1:])

    # Style all axes
    for ax in [ax_main, ax_stencil, ax_compare, ax_info]:
        ax.set_facecolor('#0d1117')
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    # Main diffusion plot
    im = ax_main.imshow(c, cmap=cmap_rg, vmin=-0.5, vmax=1.0,
                        interpolation='bilinear', origin='lower')
    ax_main.set_title('2D Diffusion Kernel', color='white', fontsize=16, fontweight='bold')
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    # Stencil visualization
    ax_stencil.set_xlim(-0.5, 4.5)
    ax_stencil.set_ylim(-0.5, 4.5)
    ax_stencil.set_aspect('equal')
    ax_stencil.set_title('5-Point Stencil', color='white', fontsize=12, fontweight='bold')
    ax_stencil.set_xticks([])
    ax_stencil.set_yticks([])

    # Draw stencil grid
    stencil_colors = np.zeros((5, 5, 3))
    stencil_colors[2, 2] = [0.1, 0.8, 0.1]  # center - green
    stencil_colors[2, 1] = [0.8, 0.4, 0.1]  # left - orange
    stencil_colors[2, 3] = [0.8, 0.4, 0.1]  # right
    stencil_colors[1, 2] = [0.8, 0.4, 0.1]  # top
    stencil_colors[3, 2] = [0.8, 0.4, 0.1]  # bottom

    for i in range(5):
        for j in range(5):
            if stencil_colors[i, j].sum() > 0:
                rect = Rectangle((j-0.4, 4-i-0.4), 0.8, 0.8,
                                  facecolor=stencil_colors[i, j],
                                  edgecolor='white', linewidth=2)
                ax_stencil.add_patch(rect)
            else:
                rect = Rectangle((j-0.4, 4-i-0.4), 0.8, 0.8,
                                  facecolor='#21262d', edgecolor='#30363d', linewidth=1)
                ax_stencil.add_patch(rect)

    # Stencil formula
    ax_stencil.text(2, -0.8, r'$c_{out} = c + \alpha(\nabla^2 c)$',
                    color='white', fontsize=10, ha='center', va='top')

    # Comparison bars
    ax_compare.set_xlim(0, 1)
    ax_compare.set_ylim(0, 4)
    ax_compare.set_title('Speed Comparison', color='white', fontsize=12, fontweight='bold')
    ax_compare.set_xticks([])
    ax_compare.set_yticks([])

    # Bar data (normalized to Dendrite = 1)
    methods = ['Dendrite', 'NumPy', 'CuPy']
    times = [2.63, 351, 518]  # ms
    normalized = [t / times[0] for t in times]
    bar_colors = ['#238636', '#6e7681', '#da3633']  # green, gray, red

    # Draw comparison bars (will animate)
    bars = []
    bar_labels = []
    for i, (method, norm, col) in enumerate(zip(methods, normalized, bar_colors)):
        bar = Rectangle((0.1, 3.2 - i * 1.1), 0.01, 0.8, facecolor=col, edgecolor='none')
        ax_compare.add_patch(bar)
        bars.append(bar)

        label = ax_compare.text(0.08, 3.6 - i * 1.1, method, color='white',
                                fontsize=10, ha='right', va='center')
        bar_labels.append(label)

        time_label = ax_compare.text(0.85, 3.6 - i * 1.1, '', color='white',
                                     fontsize=9, ha='left', va='center')
        bar_labels.append(time_label)

    # Info panel
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.set_xticks([])
    ax_info.set_yticks([])

    info_text = ax_info.text(0.5, 0.5, '', color='white', fontsize=11,
                             ha='center', va='center', family='monospace',
                             linespacing=1.8)

    # Title
    fig.suptitle('DENDRITE', color='#238636', fontsize=28, fontweight='bold', y=0.96)
    subtitle = fig.text(0.5, 0.92, 'GPU-Accelerated Battery Simulation',
                        color='#8b949e', fontsize=14, ha='center')

    # Animation state
    frames_total = 300  # 10 seconds at 30fps

    def diffusion_step(c):
        """One step of 2D diffusion."""
        c_new = c.copy()
        c_new[1:-1, 1:-1] = c[1:-1, 1:-1] + alpha * (
            c[1:-1, 2:] + c[1:-1, :-2] +
            c[2:, 1:-1] + c[:-2, 1:-1] -
            4 * c[1:-1, 1:-1]
        )
        return c_new

    def animate(frame):
        nonlocal c

        # Run diffusion (multiple steps per frame for visible change)
        for _ in range(3):
            c = diffusion_step(c)

        # Update main plot
        im.set_array(c)

        # Animate comparison bars (grow over first 2 seconds)
        bar_progress = min(1.0, frame / 60)
        max_width = 0.7

        for i, (bar, norm, time) in enumerate(zip(bars, normalized, times)):
            # Dendrite bar grows fast, others grow proportionally slower
            if i == 0:  # Dendrite
                width = max_width * bar_progress
                time_str = f'{time:.1f} ms' if bar_progress > 0.5 else ''
            else:
                # Other bars grow slower, capped at visual max
                width = min(max_width, max_width * bar_progress * min(norm / 50, 1))
                time_str = f'{time:.0f} ms' if bar_progress > 0.5 else ''

            bar.set_width(width)
            bar_labels[i * 2 + 1].set_text(time_str)

        # Update info text based on frame
        phase = frame // 100

        if phase == 0:
            info = (
                "RTX 3090 Performance:\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                "2D Diffusion:  89% peak\n"
                "Butler-Volmer: 85% peak\n"
                "Spherical:     76% peak"
            )
        elif phase == 1:
            info = (
                "Speedup vs CPU:\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                "133x faster than NumPy\n"
                "197x faster than CuPy\n"
                "(naive GPU is slower!)"
            )
        else:
            info = (
                "Pure C/CUDA kernels\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                "No Python runtime\n"
                "Hand-tuned for bandwidth\n"
                "github.com/Infatoshi/dendrite"
            )

        info_text.set_text(info)

        return [im, info_text] + bars

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=frames_total,
                                   interval=33, blit=False)

    return fig, anim


def create_spherical_diffusion_viz():
    """Create spherical diffusion visualization (electrode particles)."""

    fig = plt.figure(figsize=(16, 9), facecolor='#0d1117')

    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1],
                          left=0.08, right=0.92, bottom=0.1, top=0.85,
                          wspace=0.25, hspace=0.3)

    ax_particles = fig.add_subplot(gs[:, 0])
    ax_single = fig.add_subplot(gs[0, 1])
    ax_kernel = fig.add_subplot(gs[1, 1])

    for ax in [ax_particles, ax_single, ax_kernel]:
        ax.set_facecolor('#0d1117')
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    # Multiple particles visualization
    n_particles = 16
    nr = 32  # radial points (one warp)

    # Initial concentration profiles (varied)
    particles = []
    for i in range(n_particles):
        r = np.linspace(0, 1, nr)
        # Different initial profiles
        phase = i * 0.3
        c = 0.5 + 0.4 * np.cos(np.pi * r + phase)
        particles.append(c)
    particles = np.array(particles)

    # Show particles as colored rows
    im_particles = ax_particles.imshow(particles, cmap=cmap_rg, aspect='auto',
                                        vmin=0, vmax=1, interpolation='nearest')
    ax_particles.set_xlabel('Radial Position (r/R)', color='white', fontsize=11)
    ax_particles.set_ylabel('Particle Index', color='white', fontsize=11)
    ax_particles.set_title('10,000 Electrode Particles (showing 16)',
                           color='white', fontsize=14, fontweight='bold')
    ax_particles.tick_params(colors='white')

    # Single particle detail
    ax_single.set_xlim(0, 1)
    ax_single.set_ylim(0, 1)
    line, = ax_single.plot([], [], color='#238636', linewidth=2)
    ax_single.set_xlabel('r/R', color='white', fontsize=10)
    ax_single.set_ylabel('Li Concentration', color='white', fontsize=10)
    ax_single.set_title('Single Particle Profile', color='white', fontsize=12, fontweight='bold')
    ax_single.tick_params(colors='white')
    ax_single.grid(True, alpha=0.2, color='white')

    # Kernel info
    ax_kernel.set_xlim(0, 1)
    ax_kernel.set_ylim(0, 1)
    ax_kernel.set_xticks([])
    ax_kernel.set_yticks([])

    kernel_text = ax_kernel.text(0.5, 0.5, '', color='white', fontsize=11,
                                  ha='center', va='center', family='monospace',
                                  linespacing=1.8)

    # Title
    fig.suptitle('DENDRITE - Spherical Diffusion', color='#238636',
                 fontsize=24, fontweight='bold', y=0.95)
    fig.text(0.5, 0.90, 'Lithium diffusion in electrode particles during charge/discharge',
             color='#8b949e', fontsize=12, ha='center')

    # Diffusion parameters
    D_s = 1e-14  # Solid diffusivity
    R_p = 5e-6   # Particle radius
    dr = R_p / (nr - 1)
    dt = 0.1 * dr**2 / D_s  # CFL condition
    alpha = D_s * dt / dr**2

    r = np.linspace(0, 1, nr)

    def spherical_diffusion_step(c):
        """One step of spherical diffusion for all particles."""
        c_new = c.copy()

        # Interior points with spherical geometry correction
        for i in range(1, nr - 1):
            r_i = (i + 0.5) / nr
            r_plus = (r_i + 0.5/nr) / r_i
            r_minus = (r_i - 0.5/nr) / r_i

            c_new[:, i] = c[:, i] + alpha * (
                r_plus * (c[:, i+1] - c[:, i]) -
                r_minus * (c[:, i] - c[:, i-1])
            )

        # Boundary conditions
        c_new[:, 0] = c_new[:, 1]  # Zero flux at center
        c_new[:, -1] = c_new[:, -2] - 0.001  # Surface flux (discharge)

        return np.clip(c_new, 0, 1)

    frames_total = 300

    def animate(frame):
        nonlocal particles

        # Run diffusion steps
        for _ in range(5):
            particles = spherical_diffusion_step(particles)

        # Update particle heatmap
        im_particles.set_array(particles)

        # Update single particle line
        line.set_data(r, particles[0])

        # Update kernel info
        phase = frame // 100

        if phase == 0:
            info = (
                "Warp Shuffle Optimization:\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "32 radial points = 1 warp\n"
                "__shfl_sync for neighbors\n"
                "No shared memory needed"
            )
        elif phase == 1:
            info = (
                "Speedup:\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "Shared memory: 36% peak\n"
                "Warp shuffle:  85% peak\n"
                "2.4x improvement"
            )
        else:
            info = (
                "Battery Physics:\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "Fast charging bottleneck\n"
                "Li plating prevention\n"
                "BMS state estimation"
            )

        kernel_text.set_text(info)

        return [im_particles, line, kernel_text]

    anim = animation.FuncAnimation(fig, animate, frames=frames_total,
                                   interval=33, blit=False)

    return fig, anim


def create_combined_viz():
    """Create combined visualization showing all kernels."""

    fig = plt.figure(figsize=(16, 9), facecolor='#0d1117')

    # 3-panel layout
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1],
                          left=0.05, right=0.95, bottom=0.08, top=0.82,
                          wspace=0.15, hspace=0.25)

    ax_diff = fig.add_subplot(gs[0, 0])
    ax_bv = fig.add_subplot(gs[0, 1])
    ax_sph = fig.add_subplot(gs[0, 2])
    ax_bar = fig.add_subplot(gs[1, :])

    for ax in [ax_diff, ax_bv, ax_sph, ax_bar]:
        ax.set_facecolor('#0d1117')
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    # === 2D Diffusion ===
    nx, ny = 48, 48
    x = np.linspace(-2.5, 2.5, nx)
    y = np.linspace(-2.5, 2.5, ny)
    X, Y = np.meshgrid(x, y)
    c_diff = np.exp(-(X**2 + Y**2))

    im_diff = ax_diff.imshow(c_diff, cmap=cmap_rg, vmin=0, vmax=1,
                              interpolation='bilinear', origin='lower')
    ax_diff.set_title('2D Diffusion\n89% peak', color='white', fontsize=12, fontweight='bold')
    ax_diff.set_xticks([])
    ax_diff.set_yticks([])

    # === Butler-Volmer ===
    eta = np.linspace(-0.3, 0.3, 100)
    i0 = 1.0
    alpha_bv = 0.5
    F_RT = 38.94  # F/(RT) at 298K
    j_bv = i0 * (np.exp(alpha_bv * F_RT * eta) - np.exp(-(1-alpha_bv) * F_RT * eta))

    ax_bv.axhline(0, color='#30363d', linewidth=1)
    ax_bv.axvline(0, color='#30363d', linewidth=1)
    line_bv, = ax_bv.plot(eta, j_bv, color='#238636', linewidth=2)
    ax_bv.fill_between(eta, j_bv, 0, where=(j_bv > 0), color='#238636', alpha=0.3)
    ax_bv.fill_between(eta, j_bv, 0, where=(j_bv < 0), color='#da3633', alpha=0.3)
    ax_bv.set_xlim(-0.3, 0.3)
    ax_bv.set_ylim(-15, 15)
    ax_bv.set_xlabel('Overpotential (V)', color='white', fontsize=9)
    ax_bv.set_ylabel('Current (A/m²)', color='white', fontsize=9)
    ax_bv.set_title('Butler-Volmer\n85% peak', color='white', fontsize=12, fontweight='bold')
    ax_bv.tick_params(colors='white', labelsize=8)

    # === Spherical Diffusion ===
    nr = 32
    n_show = 12
    r = np.linspace(0, 1, nr)
    particles = np.array([0.5 + 0.4 * np.cos(np.pi * r + i * 0.4) for i in range(n_show)])

    im_sph = ax_sph.imshow(particles, cmap=cmap_rg, aspect='auto', vmin=0, vmax=1)
    ax_sph.set_xlabel('Radial position', color='white', fontsize=9)
    ax_sph.set_ylabel('Particle', color='white', fontsize=9)
    ax_sph.set_title('Spherical Diffusion\n76% peak', color='white', fontsize=12, fontweight='bold')
    ax_sph.tick_params(colors='white', labelsize=8)

    # === Comparison bar chart ===
    methods = ['Dendrite\n(CUDA)', 'NumPy\n(CPU)', 'CuPy\n(naive GPU)']
    times = [2.63, 351, 518]
    colors_bar = ['#238636', '#6e7681', '#da3633']

    ax_bar.set_xlim(0, 600)
    ax_bar.set_ylim(-0.5, 2.5)
    ax_bar.set_xlabel('Time (ms) - lower is better', color='white', fontsize=11)
    ax_bar.set_title('10K particles, 100s simulation', color='#8b949e', fontsize=10)
    ax_bar.tick_params(colors='white')
    ax_bar.set_yticks([0, 1, 2])
    ax_bar.set_yticklabels(methods, fontsize=10)
    ax_bar.invert_yaxis()

    bars = []
    bar_texts = []
    for i, (t, c) in enumerate(zip(times, colors_bar)):
        bar = ax_bar.barh(i, 0.1, color=c, height=0.6)[0]
        bars.append(bar)
        txt = ax_bar.text(5, i, '', color='white', fontsize=11, va='center', fontweight='bold')
        bar_texts.append(txt)

    # Speedup annotations (will appear later)
    speedup_texts = []

    # Title
    fig.suptitle('DENDRITE', color='#238636', fontsize=32, fontweight='bold', y=0.95)
    fig.text(0.5, 0.88, 'GPU-Accelerated Battery Simulation  |  133x faster than NumPy  |  Pure C/CUDA',
             color='#8b949e', fontsize=13, ha='center')

    # Diffusion parameters
    alpha_d = 0.15

    def diffusion_step_2d(c):
        c_new = c.copy()
        c_new[1:-1, 1:-1] = c[1:-1, 1:-1] + alpha_d * (
            c[1:-1, 2:] + c[1:-1, :-2] + c[2:, 1:-1] + c[:-2, 1:-1] - 4 * c[1:-1, 1:-1]
        )
        return c_new

    def spherical_step(p):
        p_new = p.copy()
        for i in range(1, nr - 1):
            r_i = (i + 0.5) / nr
            p_new[:, i] = p[:, i] + 0.1 * ((p[:, i+1] - p[:, i]) - (p[:, i] - p[:, i-1]))
        p_new[:, -1] = p_new[:, -2] - 0.002
        p_new[:, 0] = p_new[:, 1]
        return np.clip(p_new, 0, 1)

    frames_total = 600  # 20 seconds at 30fps

    def animate(frame):
        nonlocal c_diff, particles

        # Update diffusion
        for _ in range(2):
            c_diff = diffusion_step_2d(c_diff)
        im_diff.set_array(c_diff)

        # Update spherical
        for _ in range(3):
            particles = spherical_step(particles)
        im_sph.set_array(particles)

        # Animate bars (first 3 seconds)
        bar_progress = min(1.0, frame / 90)

        for i, (bar, t) in enumerate(zip(bars, times)):
            width = t * bar_progress
            bar.set_width(width)
            if bar_progress > 0.3:
                bar_texts[i].set_text(f'{t:.1f} ms' if t < 10 else f'{t:.0f} ms')
                bar_texts[i].set_x(width + 8)

        # Add speedup labels after bars complete
        if frame == 100 and not speedup_texts:
            txt1 = ax_bar.text(400, 1, '133x slower', color='#f0883e', fontsize=10, va='center')
            txt2 = ax_bar.text(540, 2, '197x slower', color='#f85149', fontsize=10, va='center')
            speedup_texts.extend([txt1, txt2])

        return [im_diff, im_sph] + bars + bar_texts

    anim = animation.FuncAnimation(fig, animate, frames=frames_total,
                                   interval=33, blit=False)

    return fig, anim


if __name__ == '__main__':
    import os

    # Output directory
    out_dir = '/home/infatoshi/cuda/search/battery_kernels/dendrite'
    os.makedirs(out_dir, exist_ok=True)

    print("Generating combined visualization...")
    fig, anim = create_combined_viz()

    # Save as GIF
    gif_path = os.path.join(out_dir, 'dendrite.gif')
    print(f"Saving to {gif_path}...")

    # Use pillow writer for GIF
    anim.save(gif_path, writer='pillow', fps=30, dpi=100)
    print(f"Saved: {gif_path}")

    # Also save a static frame for fallback
    fig.savefig(os.path.join(out_dir, 'dendrite_static.png'),
                facecolor='#0d1117', dpi=150, bbox_inches='tight')
    print("Saved static frame")

    plt.close()
    print("Done!")
