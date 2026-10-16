import numpy as np
import matplotlib.pyplot as plt

def generate_sphere_vectors(n):
    np.random.seed(66)
    vecs = np.random.randn(n, 3)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

def pool_vectors(vecs, k):
    n = len(vecs)
    new_vecs = np.zeros_like(vecs)
    for i in range(n):
        candidates = list(range(n))
        candidates.remove(i)
        chosen_idx = np.random.choice(candidates, k, replace=False)
        pool_group = np.vstack([vecs[i], vecs[chosen_idx]])
        new_vecs[i] = np.mean(pool_group, axis=0)
    return new_vecs

def plot_3d_sphere(ax, vecs, title):
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='lightgray', alpha=0.15, edgecolor='none')

    ax.scatter(vecs[:, 0], vecs[:, 1], vecs[:, 2],
               color='teal', s=20, alpha=0.85, edgecolors='white', linewidth=0.3)

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_title(title, fontsize=15, pad=0, fontweight='bold')

    ax.view_init(elev=15, azim=45)


def main():
    n = 100
    k_values = [2, 4, 8]

    V_original = generate_sphere_vectors(n)
    V_pooled_list = [pool_vectors(V_original, k) for k in k_values]

    fig = plt.figure(figsize=(16, 4))

    ax1 = fig.add_subplot(141, projection='3d')
    plot_3d_sphere(ax1, V_original, f"Original vectors (n={n})")

    ax2 = fig.add_subplot(142, projection='3d')
    plot_3d_sphere(ax2, V_pooled_list[0], f"Pooled (k={k_values[0]})")

    ax3 = fig.add_subplot(143, projection='3d')
    plot_3d_sphere(ax3, V_pooled_list[1], f"Pooled (k={k_values[1]})")

    ax4 = fig.add_subplot(144, projection='3d')
    plot_3d_sphere(ax4, V_pooled_list[2], f"Pooled (k={k_values[2]})")

    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.0, top=0.85, wspace=0.02)

    plt.savefig("collapse_simulation.pdf", format='pdf', bbox_inches='tight', pad_inches=0, dpi=300)

    print("collapse_simulation.pdf")
    plt.show()


if __name__ == "__main__":
    main()