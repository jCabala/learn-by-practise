import matplotlib.pyplot as plt

# MM1 Cache Size vs. D1 Miss Rate
mm1_values = [
    (64, 0.1614),
    (128, 0.1459),
    (256, 0.1381),
    (512, 0.1342),
    (1024, 0.1322),
    (2048, 0.1011),
    (4096, 0.0347),
    (8192, 0.0094),
]

mm2_values = [
    (64, 0.1069),
    (128, 0.0718),
    (256, 0.0543),
    (512, 0.0455),
    (1024, 0.0411),
    (2048, 0.0388),
    (4096, 0.0325),
    (8192, 0.0084),
]

mm3_values = [
    (64, 0.1152),
    (128, 0.0800),
    (256, 0.0624),
    (512, 0.0165),
    (1024, 0.0107),
    (2048, 0.0074),
    (4096, 0.0041),
    (8192, 0.0020),
]

mm4_values = [
    (64, 0.3248),
    (128, 0.3246),
    (256, 0.3246),
    (512, 0.3246),
    (1024, 0.3246),
    (2048, 0.3246),
    (4096, 0.1847),
    (8192, 0.1148),
]

def plot_all():
    plt.figure(figsize=(10, 6))

    # Plot MM1
    sizes_mm1, miss_rates_mm1 = zip(*mm1_values)
    plt.plot(sizes_mm1, miss_rates_mm1, marker='o', label='MM1')

    # Plot MM2
    sizes_mm2, miss_rates_mm2 = zip(*mm2_values)
    plt.plot(sizes_mm2, miss_rates_mm2, marker='o', label='MM2')

    # Plot MM3
    sizes_mm3, miss_rates_mm3 = zip(*mm3_values)
    plt.plot(sizes_mm3, miss_rates_mm3, marker='o', label='MM3')

    # Plot MM4
    sizes_mm4, miss_rates_mm4 = zip(*mm4_values)
    plt.plot(sizes_mm4, miss_rates_mm4, marker='o', label='MM4')

    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Cache Size (Bytes)')
    plt.ylabel('D1 Miss Rate')
    plt.title('Cache Size vs. D1 Miss Rate for MM1-MM4')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('cache_vs_miss_rate.png')

if __name__ == "__main__":
    plot_all()