"""Download the 28 public test assets (Khronos glTF sample models, real artist and CAD-exported meshes) to data/.
python scripts/get_data.py"""
import urllib.request
from pathlib import Path

NEW = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/{m}/glTF-Binary/{m}.glb"
OLD = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/{m}/glTF-Binary/{m}.glb"
MODELS = {NEW: ["ToyCar", "CarConcept", "CesiumMilkTruck", "DamagedHelmet", "Lantern", "WaterBottle", "BoomBox", "Avocado",
                "Duck", "AntiqueCamera", "Corset", "SheenChair", "GlamVelvetSofa", "ChairDamaskPurplegold", "IridescenceLamp",
                "ABeautifulGame", "Fox", "BrainStem", "MosquitoInAmber", "DragonAttenuation", "PotOfCoals", "ChronographWatch",
                "CommercialRefrigerator", "SunglassesKhronos"],
          OLD: ["Buggy", "2CylinderEngine", "GearboxAssy", "ReciprocatingSaw"]}
D = Path(__file__).resolve().parents[1] / "data"
D.mkdir(exist_ok=True)
for url, ms in MODELS.items():
    for m in ms:
        out = D / f"{m}.glb"
        if not out.exists():
            print("downloading", m); urllib.request.urlretrieve(url.format(m=m), out)
print("data ready")
