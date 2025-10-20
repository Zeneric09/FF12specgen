import os
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import multiprocessing
from functools import partial
from tqdm import tqdm
import cv2

# ==============================
# ⚙️ CONFIGURATION
# ==============================
DEFAULT_SPEC_INTENSITY = 0.5
MAX_WORKERS = max(1, min(multiprocessing.cpu_count(), 16))
# ==============================


def get_user_settings(default_intensity=DEFAULT_SPEC_INTENSITY):
    try:
        val = input(f"Enter spec intensity (default {default_intensity}): ").strip()
        val = float(val) if val else default_intensity
        fallback_input = input("Generate diffuse-based specs when no reference found? (y/n, default y): ").strip().lower()
        fallback = not fallback_input.startswith("n")
        return max(0.1, min(val, 5.0)), fallback
    except Exception:
        print(f"⚠️ Invalid or no input. Using defaults (intensity={default_intensity}, fallback=True)")
        return default_intensity, True


def preprocess_diffuse(diffuse_img):
    gray = ImageOps.grayscale(diffuse_img)
    enhancer = ImageEnhance.Contrast(gray)
    enhanced_gray = enhancer.enhance(1.6)
    return enhanced_gray.convert("RGB")


def adaptive_metal_mask(diffuse_np):
    mean = np.mean(diffuse_np, axis=-1)
    sat = np.std(diffuse_np, axis=-1) / (mean + 1e-5)
    return (sat < 0.15) & (mean < 0.5)


def generate_spec_from_reference(diffuse_img, reference_spec_img, spec_intensity=DEFAULT_SPEC_INTENSITY):
    diffuse_pre = preprocess_diffuse(diffuse_img)
    diffuse_np = np.array(diffuse_pre, dtype=np.float32) / 255.0
    ref_np = np.array(reference_spec_img.convert("RGB").resize(diffuse_pre.size, Image.LANCZOS),
                      dtype=np.float32) / 255.0

    diff_lum = np.mean(diffuse_np, axis=-1)
    ref_lum = np.mean(ref_np, axis=-1)
    ratio = np.clip(np.mean(ref_lum / np.clip(diff_lum, 1e-3, None)), 0.5, 2.0)

    spec_base = np.clip(diffuse_np * ratio, 0, 1)
    blend_ratio = np.clip(np.mean(np.abs(ref_np - diffuse_np)), 0.3, 0.7)
    spec_np = np.clip(blend_ratio * ref_np + (1 - blend_ratio) * spec_base, 0, 1)

    inv = np.clip(1.0 - np.mean(diffuse_np, axis=-1), 0, 1)
    spec_np *= np.power(inv, 0.75)[..., None]

    gray = np.mean(diffuse_np, axis=-1)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    spec_np += np.sqrt(gx ** 2 + gy ** 2)[..., None] * 0.15

    r, g, b = ref_np[..., 0], ref_np[..., 1], ref_np[..., 2]
    skin_mask = (b > 0.7) & (b > r * 2) & (b > g * 2)
    matte_mask = (b > 0.3) & (b < 0.7) & (r < 0.4) & (g < 0.4)
    yellow_shiny_mask = (r > 0.6) & (g > 0.5) & (b < 0.35) & ((r + g) > 1.2)
    metal_mask = (r > 0.5) & (g > 0.45) & (b > 0.5) & (np.abs(r - b) < 0.2)

    spec_np[skin_mask] *= [0.4, 0.38, 0.35]
    spec_np[matte_mask] *= 0.15
    spec_np[yellow_shiny_mask] *= [1.8, 1.75, 1.5]
    spec_np[metal_mask] *= [1.3, 1.25, 1.2]

    if np.any(skin_mask):
        skin_blur = cv2.GaussianBlur(spec_np, (11, 11), 0)
        spec_np[skin_mask] = np.clip(skin_blur[skin_mask] * 0.8, 0, 1)

    if np.any(yellow_shiny_mask):
        shiny_gray = np.mean(spec_np, axis=-1)
        blur = cv2.GaussianBlur(shiny_gray, (5, 5), 0)
        highpass = np.clip(shiny_gray - blur, 0, 1)[..., None]
        spec_np[yellow_shiny_mask] = np.clip(
            spec_np[yellow_shiny_mask] + highpass[yellow_shiny_mask] * 1.2, 0, 1
        )

    if np.any(metal_mask):
        spec_np[metal_mask] = np.clip(spec_np[metal_mask] * 1.2, 0, 1)

    spec_np[adaptive_metal_mask(diffuse_np)] *= [1.1, 1.05, 0.9]
    spec_np = np.clip(spec_np * spec_intensity, 0, 1)
    return Image.fromarray((spec_np * 255).astype(np.uint8))


def generate_spec_from_diffuse(diffuse_img, spec_intensity=DEFAULT_SPEC_INTENSITY):
    diffuse_pre = preprocess_diffuse(diffuse_img)
    diffuse_np = np.array(diffuse_pre, dtype=np.float32) / 255.0
    gray = np.mean(diffuse_np, axis=-1)

    blur_small = cv2.GaussianBlur(gray, (3, 3), 0)
    blur_large = cv2.GaussianBlur(gray, (13, 13), 0)
    detail = np.clip((blur_small - blur_large) * 4.0 + 0.5, 0, 1)
    inv = np.clip(1.0 - detail, 0, 1)
    spec_np = np.power(inv, 0.75)

    avg = np.mean(diffuse_np, axis=-1)
    spec_np = np.clip(np.power(1.2 - avg, 0.8) * spec_np, 0, 1)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    edges = np.sqrt(gx ** 2 + gy ** 2)
    spec_np = np.clip(spec_np + edges * 0.15, 0, 1)

    metal_mask = adaptive_metal_mask(diffuse_np)
    spec_np[metal_mask] *= 1.3

    tint = diffuse_np ** 0.8
    tint /= (np.mean(tint, axis=-1, keepdims=True) + 1e-5)
    spec_rgb = np.clip(spec_np[..., None] * tint, 0, 1)
    spec_rgb = np.clip(spec_rgb * spec_intensity, 0, 1)
    return Image.fromarray((spec_rgb * 255).astype(np.uint8))


def find_matching_spec(diffuse_path):
    folder = os.path.dirname(diffuse_path)
    base_name = os.path.splitext(os.path.basename(diffuse_path))[0]
    candidate = os.path.join(folder, f"{base_name}_spec.png")
    return candidate if os.path.exists(candidate) else None


def process_texture(in_path, base_folder, output_folder, spec_intensity, use_fallback_generation):
    rel_path = os.path.relpath(in_path, base_folder)
    out_path = os.path.join(output_folder, os.path.splitext(rel_path)[0] + "_spec.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        diffuse_img = Image.open(in_path)
        matching_spec_path = find_matching_spec(in_path)
        if matching_spec_path:
            local_ref = Image.open(matching_spec_path)
            spec_img = generate_spec_from_reference(diffuse_img, local_ref, spec_intensity)
            print(f"💾 Using reference spec for: {rel_path}")
        elif use_fallback_generation:
            print(f"⚠️ No ref spec found. Generating from diffuse: {rel_path}")
            spec_img = generate_spec_from_diffuse(diffuse_img, spec_intensity)
        else:
            print(f"⏭️ Skipped (no ref spec): {rel_path}")
            return False

        spec_img.save(out_path)
        return True
    except Exception as e:
        print(f"❌ Failed {rel_path}: {e}")
        return False


def collect_diffuse_textures(base_folder):
    diffuse_paths = []
    for root, _, files in os.walk(base_folder):
        if "output" in root.lower():
            continue
        for f in files:
            if f.lower().endswith(".png") and "_spec" not in f.lower():
                diffuse_paths.append(os.path.join(root, f))
    return diffuse_paths


def batch_generate_spec(base_folder, output_folder, spec_intensity, use_fallback_generation):
    import concurrent.futures
    os.makedirs(output_folder, exist_ok=True)
    all_textures = collect_diffuse_textures(base_folder)
    total = len(all_textures)
    print(f"\n🧮 Processing {total} textures using {MAX_WORKERS} threads...")
    print(f"🎚️ Spec intensity: {spec_intensity}")
    print(f"🎨 Fallback diffuse-based generation: {'ON' if use_fallback_generation else 'OFF'}")

    func = partial(process_texture,
                   base_folder=base_folder,
                   output_folder=output_folder,
                   spec_intensity=spec_intensity,
                   use_fallback_generation=use_fallback_generation)

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
    # 🔧 Headless mode — no Tkinter GUI
    spec_intensity, use_fallback_generation = get_user_settings()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(current_dir, "output")
    batch_generate_spec(current_dir, output_folder, spec_intensity, use_fallback_generation)
