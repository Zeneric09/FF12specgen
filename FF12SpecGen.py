import os
import concurrent.futures
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import multiprocessing
from functools import partial
from tqdm import tqdm

# ==============================
# ⚙️ CONFIGURATION
# ==============================
DEFAULT_SPEC_INTENSITY = 0.225
MAX_WORKERS = max(1, min(multiprocessing.cpu_count(), 16))
# ==============================


def safe_ask_spec_intensity(default_value=DEFAULT_SPEC_INTENSITY):
    """Prompt user for spec intensity (GUI if available, otherwise console)."""
    try:
        import tkinter as tk

        class IntensityDialog(tk.Tk):
            def __init__(self):
                super().__init__()
                self.title("FF12 SpecGen - Spec Intensity")
                self.resizable(False, False)
                self.value = default_value

                tk.Label(self, text="Enter spec intensity (higher = shinier):").pack(padx=10, pady=10)

                self.entry = tk.Entry(self)
                self.entry.insert(0, str(default_value))
                self.entry.pack(padx=10, pady=5)
                self.entry.focus_set()

                ok_btn = tk.Button(self, text="OK", width=10, command=self.on_ok)
                ok_btn.pack(pady=10)

            def on_ok(self):
                try:
                    val = float(self.entry.get())
                    self.value = max(0.1, min(val, 5.0))
                except Exception:
                    self.value = default_value
                self.destroy()

        dialog = IntensityDialog()
        dialog.mainloop()

        # Ensure Tkinter fully cleaned up before multithreading starts
        import tkinter
        tkinter._default_root = None

        return dialog.value

    except Exception:
        try:
            val = float(input(f"⚙️ Enter spec intensity multiplier (default {default_value}): ").strip())
            return max(0.1, min(val, 5.0))
        except Exception:
            print(f"⚠️ Invalid input. Using default {default_value}")
            return default_value


def preprocess_diffuse(diffuse_img):
    gray = ImageOps.grayscale(diffuse_img)
    enhancer = ImageEnhance.Contrast(gray)
    enhanced_gray = enhancer.enhance(1.6)
    return enhanced_gray.convert("RGB")


def generate_spec_from_reference(diffuse_img, reference_spec_img, spec_intensity=DEFAULT_SPEC_INTENSITY):
    """Generates a specular map based on a reference and diffuse texture, with safe math for high intensity values."""
    diffuse_pre = preprocess_diffuse(diffuse_img)
    diffuse_np = np.array(diffuse_pre, dtype=np.float32) / 255.0

    ref_np = np.array(reference_spec_img.convert("RGB").resize(diffuse_pre.size, Image.LANCZOS),
                      dtype=np.float32) / 255.0

    # Compute luminance safely
    diff_lum = np.mean(diffuse_np, axis=-1)
    ref_lum = np.mean(ref_np, axis=-1)

    diff_lum_safe = np.clip(diff_lum, 1e-4, None)
    ratio = ref_lum / diff_lum_safe
    ratio_mean = np.clip(np.mean(ratio), 0.5, 2.0)

    spec_base = np.clip(diffuse_np * ratio_mean, 0, 1)
    spec_np = np.clip(0.6 * spec_base + 0.4 * ref_np, 0, 1)

    # Apply intensity tweak safely (handles NaN, Inf, overflow)
    spec_np = spec_np * spec_intensity
    spec_np = np.nan_to_num(spec_np, nan=0.0, posinf=1.0, neginf=0.0)
    spec_np = np.clip(spec_np, 0, 1)

    if not np.isfinite(spec_np).all():
        print(f"⚠️ Warning: invalid pixel values detected (intensity={spec_intensity})")

    # Optional brightness-based metal mask enhancement
    brightness = np.mean(diffuse_np, axis=-1)
    metal_mask = (brightness < 0.4) & (diffuse_np[..., 2] < 0.3)
    spec_np[metal_mask] = np.clip(spec_np[metal_mask] * 1.25, 0, 1)

    return Image.fromarray((spec_np * 255).astype(np.uint8))


def find_matching_spec(diffuse_path):
    folder = os.path.dirname(diffuse_path)
    base_name = os.path.splitext(os.path.basename(diffuse_path))[0]
    possible_spec_name = f"{base_name}_spec.png"
    candidate = os.path.join(folder, possible_spec_name)
    return candidate if os.path.exists(candidate) else None


def process_texture(in_path, base_folder, output_folder, spec_intensity):
    rel_path = os.path.relpath(in_path, base_folder)
    out_path = os.path.join(output_folder, os.path.splitext(rel_path)[0] + "_spec.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        diffuse_img = Image.open(in_path)
        matching_spec_path = find_matching_spec(in_path)

        if not matching_spec_path:
            print(f"⚠️ Skipped (no ref spec): {rel_path}")
            return False

        local_ref = Image.open(matching_spec_path)
        spec_img = generate_spec_from_reference(diffuse_img, local_ref, spec_intensity)
        spec_img.save(out_path)
        print(f"💾 Saved spec: {rel_path}")
        return True

    except Exception as e:
        print(f"❌ Failed {rel_path}: {e}")
        return False


def collect_diffuse_textures(base_folder):
    diffuse_paths = []
    for root, _, files in os.walk(base_folder):
        if "output" in root.lower():
            continue
        for fname in files:
            if fname.lower().endswith(".png") and "_spec" not in fname.lower():
                diffuse_paths.append(os.path.join(root, fname))
    return diffuse_paths


def batch_generate_spec(base_folder, output_folder, spec_intensity):
    os.makedirs(output_folder, exist_ok=True)
    all_textures = collect_diffuse_textures(base_folder)
    total = len(all_textures)
    print(f"\n🧮 Processing {total} textures using {MAX_WORKERS} threads...")
    print(f"🎚️ Spec intensity: {spec_intensity}")

    func = partial(
        process_texture,
        base_folder=base_folder,
        output_folder=output_folder,
        spec_intensity=spec_intensity,
    )

    processed = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for ok in tqdm(executor.map(func, all_textures),
                       total=total, desc="Generating Specs", dynamic_ncols=True, leave=True):
            if ok:
                processed += 1
            else:
                failed += 1

    print("\n✨ All done!")
    print(f"✅ Successfully processed: {processed}")
    print(f"⚠️ Skipped or failed: {failed}")
    print(f"🖼️ Output folder: {os.path.abspath(output_folder)}\n")

    input("Press any key to exit...")


if __name__ == "__main__":
    spec_intensity = safe_ask_spec_intensity()

    # Prevent Tcl threading errors before spawning threads
    try:
        import tkinter
        tkinter._default_root = None
    except Exception:
        pass

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(current_dir, "output")
    batch_generate_spec(current_dir, output_folder, spec_intensity)
