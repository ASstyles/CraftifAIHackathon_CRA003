import os
import zipfile

def package_pipegen_export():
    zip_path = "pipegen_pipeline/craftifai_pipegen_export_CRA003.zip"
    src_dir = "pipegen_pipeline/official_pipegen_build"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(src_dir):
            for f in files:
                if f.endswith(".plan") or f.endswith(".onnx"):
                    continue
                full_path = os.path.join(root, f)
                arcname = os.path.relpath(full_path, src_dir)
                z.write(full_path, os.path.join("official_pipegen_build", arcname))
        
        z.write("pipegen_pipeline/requirement.txt", "requirement.txt")
        z.write("pipegen_pipeline/actual_measured_benchmark.json", "actual_measured_benchmark.json")
        z.write("pipegen_pipeline/speed_report.md", "speed_report.md")
        z.write("pipegen_pipeline/config_infer_primary.txt", "config_infer_primary.txt")

    size_kb = os.path.getsize(zip_path) / 1024
    print(f"Created {zip_path} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    package_pipegen_export()
