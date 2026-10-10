import numpy as np
import matplotlib.pyplot as plt

suhu_u = np.arange(10, 40.5, 0.5)
mhs_u = np.arange(0, 51, 1)
kipas_u = np.arange(0, 101, 1)


def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    kiri = np.where(b > a, (x - a) / (b - a + 1e-12), 1.0)
    kanan = np.where(d > c, (d - x) / (d - c + 1e-12), 1.0)
    return np.clip(np.minimum(np.minimum(kiri, 1.0), kanan), 0, 1)


def trimf(x, a, b, c):
    return trapmf(x, a, b, b, c)


SUHU = {
    "Dingin": lambda x: trapmf(x, 10, 10, 15, 20),
    "Normal": lambda x: trimf(x, 18, 25, 32),
    "Panas": lambda x: trapmf(x, 28, 34, 40, 40),
}

MHS = {
    "Sedikit": lambda x: trapmf(x, 0, 0, 10, 25),
    "Sedang": lambda x: trimf(x, 10, 25, 40),
    "Banyak": lambda x: trapmf(x, 25, 40, 50, 50),
}

KIPAS = {
    "Lambat": lambda x: trapmf(x, 0, 0, 20, 45),
    "Sedang": lambda x: trimf(x, 30, 50, 70),
    "Cepat": lambda x: trapmf(x, 55, 80, 100, 100),
}

RULES = [
    ("Dingin", "Sedikit", "Lambat"),
    ("Dingin", "Sedang", "Lambat"),
    ("Dingin", "Banyak", "Sedang"),
    ("Normal", "Sedikit", "Lambat"),
    ("Normal", "Sedang", "Sedang"),
    ("Normal", "Banyak", "Cepat"),
    ("Panas", "Sedikit", "Sedang"),
    ("Panas", "Sedang", "Cepat"),
    ("Panas", "Banyak", "Cepat"),
]


def fuzzifikasi(suhu, mhs):
    return (
        {k: float(f(suhu)) for k, f in SUHU.items()},
        {k: float(f(mhs)) for k, f in MHS.items()},
    )


def inferensi(mu_suhu, mu_mhs):
    agregat = np.zeros_like(kipas_u, dtype=float)
    aktif = []

    for i, (s, m, out) in enumerate(RULES, start=1):
        alpha = min(mu_suhu[s], mu_mhs[m])

        if alpha > 0:
            aktif.append((i, s, m, out, alpha))

        potong = np.minimum(alpha, KIPAS[out](kipas_u))
        agregat = np.maximum(agregat, potong)

    return agregat, aktif


def defuzzifikasi(agregat):
    total = agregat.sum()
    return float(np.sum(kipas_u * agregat) / total) if total > 0 else 0.0


def label(v):
    mu = {k: float(f(v)) for k, f in KIPAS.items()}
    return max(mu, key=mu.get).upper()


def hitung(suhu, mhs):
    suhu = min(max(suhu, suhu_u[0]), suhu_u[-1])
    mhs = min(max(mhs, mhs_u[0]), mhs_u[-1])

    mu_suhu, mu_mhs = fuzzifikasi(suhu, mhs)
    agregat, aktif = inferensi(mu_suhu, mu_mhs)
    hasil = defuzzifikasi(agregat)

    print("\nFUZZIFIKASI")
    print(f"Suhu {suhu} °C      : " +
          ", ".join(f"{k}={v:.2f}" for k, v in mu_suhu.items()))
    print(f"Mahasiswa {mhs:g} org: " +
          ", ".join(f"{k}={v:.2f}" for k, v in mu_mhs.items()))

    print("\nRULE AKTIF")
    for i, s, m, out, a in aktif:
        print(f"R{i}: IF suhu={s} AND mhs={m} THEN kipas={out} (α={a:.2f})")

    print(f"\nKecepatan kipas = {hasil:.2f} % ({label(hasil)})")
    return hasil, agregat


def plot_hasil(suhu, mhs, hasil, agregat):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    for nama, f in SUHU.items():
        ax[0, 0].plot(suhu_u, f(suhu_u), label=nama)
    ax[0, 0].axvline(suhu, color="k", ls="--")
    ax[0, 0].set_title("Suhu Ruangan (°C)")
    ax[0, 0].legend()

    for nama, f in MHS.items():
        ax[0, 1].plot(mhs_u, f(mhs_u), label=nama)
    ax[0, 1].axvline(mhs, color="k", ls="--")
    ax[0, 1].set_title("Jumlah Mahasiswa Hadir")
    ax[0, 1].legend()

    for nama, f in KIPAS.items():
        ax[1, 0].plot(kipas_u, f(kipas_u), label=nama)
    ax[1, 0].set_title("Kecepatan Kipas (%)")
    ax[1, 0].legend()

    ax[1, 1].fill_between(kipas_u, agregat, alpha=0.4)
    ax[1, 1].axvline(
        hasil, color="r", ls="--", label=f"Centroid = {hasil:.1f}%"
    )
    ax[1, 1].set_title("Agregasi & Defuzzifikasi")
    ax[1, 1].legend()

    plt.tight_layout()
    plt.show()


def main():
    while True:
        mhs = float(input("Jumlah mahasiswa hadir (0-50) : "))
        suhu = float(input("Suhu ruangan (10-40 °C)       : "))
        
        hasil, agregat = hitung(suhu, mhs)
        plot_hasil(suhu, mhs, hasil, agregat)


if __name__ == "__main__":
    main()