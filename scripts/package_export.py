import os
import zipfile

def package_pipegen_export():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    zip_path = os.path.join(repo_root, "pipegen_pipeline", "craftifai_pipegen_export_CRA003.zip")
    src_dir = os.path.join(repo_root, "pipegen_pipeline", "official_pipegen_build")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        # Include official PipeGen build directory
        for root, dirs, files in os.walk(src_dir):
            for f in files:
                # Exclude hardware-specific .plan binaries (they only run on Ampere CC 8.6)
                if f.endswith(".plan"):
                    continue
                full_path = os.path.join(root, f)
                arcname = os.path.relpath(full_path, src_dir)
                z.write(full_path, os.path.join("official_pipegen_build", arcname))
        
        # Include documentation, benchmarks, requirements, and guides
        z.write(os.path.join(repo_root, "pipegen_pipeline", "requirement.txt"), "requirement.txt")
        z.write(os.path.join(repo_root, "pipegen_pipeline", "actual_measured_benchmark.json"), "actual_measured_benchmark.json")
        z.write(os.path.join(repo_root, "pipegen_pipeline", "speed_report.md"), "speed_report.md")
        z.write(os.path.join(repo_root, "pipegen_pipeline", "config_infer_primary.txt"), "config_infer_primary.txt")
        z.write(os.path.join(repo_root, "pipegen_pipeline", "MODELS_README.md"), "MODELS_README.md")
        
        # Include scripts for compilation, setup, and execution
        z.write(os.path.join(repo_root, "scripts", "compile_engines.sh"), "scripts/compile_engines.sh")
        z.write(os.path.join(repo_root, "scripts", "setup_models.sh"), "scripts/setup_models.sh")
        z.write(os.path.join(repo_root, "scripts", "run_pipegen_end_to_end.sh"), "scripts/run_pipegen_end_to_end.sh")

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Created {zip_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    package_pipegen_export()
