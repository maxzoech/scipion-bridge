import scipion_bridge as B

class CTF(B.Struct):
    voltage_kv: float              # Accelerating voltage (typically 300.0 or 200.0 kV)
    amplitude_contrast: float      # Amplitude contrast fraction (typically 0.07 to 0.10)
    spherical_aberration_mm: float # Spherical aberration (Cs) of the objective lens in mm (e.g., 2.7)
    defocus_u: float               # Defocus along the major axis (usually in Angstroms)
    defocus_v: float               # Defocus along the minor axis (usually in Angstroms)
    defocus_angle: float           # Astigmatism angle between the U axis and X axis (degrees)
    phase_shift: float             # Phase shift (in degrees), usually 0.0 unless using a Volta Phase Plate

class Particle(B.Struct):
    pixels: B.Array
    ctf: CTF


def test_static_size_set():

    set_schema = B.Set[CTF].schema()
    set_schema.print_tree()

    CTF.print_schema()
    


if __name__ == "__main__":
    test_static_size_set()