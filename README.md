Summary of FF12SpecGen.py:

A Python utility script that batch-generates specular texture maps for Final Fantasy 12 TZA using diffuse textures and existing reference spec maps.

Key features:

Prompts the user (via Tkinter GUI or console) for a specular intensity multiplier.

Preprocesses diffuse textures (grayscale + contrast enhancement).

Generates new spec maps by blending diffuse and reference spec data with safe math and brightness-based metal masking.

Processes textures in parallel using multithreading for speed.

Automatically skips missing references and saves results to an “output” folder.

Provides real-time progress via tqdm and final success/failure statistics.

In short: it’s a multithreaded image-processing tool that automates generating specular maps for texture sets, with adjustable intensity and safety against numeric errors.


PYTHON MODULE REQUIREMENTS:

pip install pillow numpy tqdm

pip install tkinter


Put the script and your textures in a directory like this:

/MyTextures

 ├─ FF12SpecGen.py
 
 ├─ wood.png
 
 ├─ wood_spec.png      ← reference spec map
 
 ├─ metal.png
 
 ├─ metal_spec.png
 
 └─ ...

Each diffuse texture (e.g. wood.png) should have a matching spec reference (e.g. wood_spec.png).
