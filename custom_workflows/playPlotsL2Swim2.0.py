import numpy as np
import matplotlib.pyplot as plt

def plot_energy(simple_avg_energy, cfd_avg_energy, tolerance_label="0.1 m", outpath="averageEffort.png"):
    plt.figure(figsize=(6, 4))
    labels = ["Avg Effort Rectangular", "Avg Effort CFD"]
    values = [simple_avg_energy, cfd_avg_energy]
    plt.bar(labels, values, alpha=0.8)
    plt.ylabel("Control Effort (PWM)^2 * s")
    plt.title(f"Average Control Effort (tolerance {tolerance_label})")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_stop_times(simple_success_times, cfd_success_times, tolerance_label="0.1 m", outpath="averageStop.png"):
    if len(simple_success_times) == 0 or len(cfd_success_times) == 0:
        print("No success times to plot.")
        return

    s = np.asarray(simple_success_times, dtype=float)
    c = np.asarray(cfd_success_times, dtype=float)

    simple_avg_stop = float(s.mean())
    cfd_avg_stop = float(c.mean())

    fig, ax = plt.subplots(figsize=(6, 4))

    # Bars for averages
    xs = [0, 1]
    ax.bar(xs, [simple_avg_stop, cfd_avg_stop], alpha=0.7)

    # Draw each boxplot separately to avoid ragged-array coercion
    ax.boxplot(s, positions=[xs[0]], widths=0.35)
    ax.boxplot(c, positions=[xs[1]], widths=0.35)

    ax.set_xticks(xs)
    ax.set_xticklabels(["Avg Stop Time Rectangular", "Avg Stop Time CFD"])
    ax.set_ylabel("Stop time (s)")
    ax.set_title(f"Average Stop Times (tolerance {tolerance_label})")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

    # plt.figure(figsize=(6,4))
    # plt.bar(["Successes", "Failures"], [successes, failures], color=["green", "red"])
    # plt.ylabel("Count")
    # plt.title(f"Num. of Successes vs Failures, {success_radius_m} Tolerance")
    # plt.tight_layout()
    # plt.savefig("successFailures.png")

def main():
    # --- inputs (edit as needed) ---
    tolerance_label = "0.1 m"

    simple_avg_energy = 2.2234
    cfd_avg_energy = 3.6964

    simple_success_times = [2.2000010013580322, 1.658332109451294, 1.8249986171722412, 1.9083318710327148, 1.4416656494140625, 2.125, 2.2583351135253906, 1.7416653633117676, 1.5249989032745361, 1.566665530204773, 1.6749987602233887, 1.5999988317489624, 1.29166579246521, 1.7166653871536255, 1.7583320140838623, 1.749998688697815, 1.5833321809768677, 1.7333320379257202, 1.8333319425582886, 2.5500056743621826, 1.9249985218048096, 1.7416653633117676, 2.0499989986419678, 1.6916654109954834, 1.3583323955535889, 1.3333324193954468, 1.6083321571350098, 1.6916654109954834, 1.2083325386047363, 1.2833324670791626, 1.7166653871536255, 1.0833326578140259, 1.658332109451294, 1.2583324909210205, 2.141666889190674, 1.3249990940093994, 1.4666656255722046, 1.2999991178512573]

    cfd_success_times = [1.4416656494140625, 1.4416656494140625, 1.4166656732559204, 1.2416658401489258, 1.2083325386047363, 1.4416656494140625, 1.3333324193954468, 1.2499991655349731, 1.7249987125396729, 1.3249990940093994, 1.3249990940093994, 1.5583322048187256, 1.2333325147628784, 1.3499990701675415, 1.4583323001861572, 1.4583323001861572, 1.5416655540466309, 1.3333324193954468, 1.1499992609024048, 1.4583323001861572, 1.3249990940093994, 1.4249989986419678, 1.0916659832000732, 0.9999994039535522, 1.4499989748001099, 1.4583323001861572, 1.5166655778884888, 1.683332085609436, 1.5249989032745361, 1.4333323240280151, 1.29166579246521, 1.3666657209396362, 1.4166656732559204, 1.4166656732559204, 0.8666661977767944, 1.016666054725647, 1.3249990940093994, 1.3999990224838257, 1.2333325147628784, 1.316665768623352, 1.0999993085861206, 0.9416661262512207, 1.5499988794326782, 0.9833327531814575, 0.8166662454605103, 1.4416656494140625, 1.3249990940093994, 1.3999990224838257, 1.1583325862884521, 1.2416658401489258, 1.749998688697815, 1.2583324909210205, 0.8749995231628418]

    # --- plots ---
    plot_energy(simple_avg_energy, cfd_avg_energy, tolerance_label, outpath="averageEnergy.png")
    plot_stop_times(simple_success_times, cfd_success_times, tolerance_label, outpath="averageStop.png")

if __name__ == "__main__":
    main()
