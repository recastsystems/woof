"""Wave maps through the production Rust renderer and normal render tree."""
from ..data import invoke
import json
from pathlib import Path

def run(request):
    from woof.ocean.door import render_ocean
    document = json.loads(Path(request).read_text())
    if "comparisons" in document:
        out = Path(document["output"])
        out.mkdir(parents=True, exist_ok=True)
        workdirs = []
        for i, row in enumerate(document["comparisons"]):
            row = {**row, "operation": "products"}
            recipe = out / f"comparison-{i:02}.json"
            recipe.write_text(json.dumps(row, indent=2) + "\n")
            workdirs.append(invoke(recipe, operation="products"))
        scale_recipe = out / "shared-scales-request.json"
        scale_recipe.write_text(json.dumps({"operation":"product-scales", "threads":document.get("threads",20), "output":str(out / "scales"), "workdirs":[str(p) for p in workdirs]}, indent=2) + "\n")
        invoke(scale_recipe, operation="product-scales")
        for workdir in workdirs:
            render_ocean(workdir / "render-request.json")
        (out / "comparison-receipt.json").write_text(json.dumps({"status":"complete", "workdirs":[str(p) for p in workdirs], "shared_scales":str(out / "scales/shared-scales.json")}, indent=2) + "\n")
        return out
    out = invoke(request, operation="products")
    render_ocean(out / "render-request.json")
    return out
