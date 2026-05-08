import cv2
import numpy as np
import sys

f1 = "/mnt/zone/A/scannetpp_release_realtrain/data/2b5ef64cad/iphone/rgb/frame_000438.png"
f2 = "/tmp/src_check.png"

img1 = cv2.imread(f1)
img2 = cv2.imread(f2)

if img1 is None or img2 is None:
    print(f"Error: {f1 if img1 is None else f2} not found")
    sys.exit(1)

if img1.shape != img2.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

m1 = np.mean(img1, axis=(0,1))
m2 = np.mean(img2, axis=(0,1))
l1 = np.mean(0.299*img1[:,:,2] + 0.587*img1[:,:,1] + 0.114*img1[:,:,0])
l2 = np.mean(0.299*img2[:,:,2] + 0.587*img2[:,:,1] + 0.114*img2[:,:,0])
diff = np.abs(m1 - m2)

print(f"Extracted Mean: {m1}, Lum: {l1:.2f}")
print(f"Source    Mean: {m2}, Lum: {l2:.2f}")
print(f"Abs Diff: {diff}")
v = "likely normal" if np.mean(diff) < 15 else "likely washed out/color shifted"
print(f"Verdict: {v}")
