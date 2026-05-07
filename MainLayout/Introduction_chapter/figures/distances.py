import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

# ==========================
# Cosmology definition
# ==========================
H0 = 70.0          # Hubble constant (km/s/Mpc)
Om0 = 0.3          # Matter density parameter

cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

# ==========================
# Redshift range
# ==========================
z = np.linspace(0.001, 3, 500)

# ==========================
# Distance calculations
# ==========================
Dc = cosmo.comoving_distance(z).value          # Mpc
Da = cosmo.angular_diameter_distance(z).value  # Mpc
Dl = cosmo.luminosity_distance(z).value        # Mpc

# ==========================
# Plotting
# ==========================
plt.figure(figsize=(8, 6))

plt.plot(z, Dc, label="Comoving Distance $D_C$", linewidth=2)
plt.plot(z, Da, "--", label="Angular Diameter Distance $D_A$", linewidth=2)
plt.plot(z, Dl, ":", label="Luminosity Distance $D_L$", linewidth=2)

plt.xlabel("Redshift $z$")
plt.ylabel("Distance (Mpc)")
plt.title("Cosmological Distance Measures")
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.savefig('distances.pdf')
plt.show()
