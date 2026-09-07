import os
import pathlib

def parse_knapsack_pisinger_xiang(folder_path, experiment_name="Parsed", random_seed=42, notes="", output_folder=None, file_prefix="instance"):
    """
    Parse all files in folder_path that contain one or more 0-1 knapsack instances
    Format per instance:
        n W
        w_1 p_1
        ...
        w_n p_n
    If output_folder is given, writes one YAML file per instance:
        {file_prefix}_{idx}.yaml
    Returns a list of YAML strings (one per instance).
    """
    instances = []

    for fname in sorted(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        if not os.path.isfile(fpath):
            continue

        with open(fpath, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        i = 0
        while i < len(lines):
            parts = lines[i].split()
            if len(parts) < 2:
                break
            try:
                n = int(parts[0])
                W = int(parts[1])
            except ValueError:
                break
            i += 1

            items = []
            for _ in range(n):
                if i >= len(lines):
                    raise ValueError(f"Unexpected end of file in {fname} while reading items")
                wp = lines[i].split()
                if len(wp) < 2:
                    raise ValueError(f"Bad item line in {fname}: {lines[i]}")
                print(f'wp={wp}')
                w = int(wp[0])
                p = int(wp[1])
                items.append({"weight": w, "value": p})
                i += 1

            instances.append({"capacity": W, "items": items})

    yaml_docs = []
    for inst in instances:
        doc_lines = [
            f'experiment_name: "{experiment_name}"',
            f"random_seed: {random_seed}",
            f'notes: "{notes}"',
            "number_knapsack: 1",
            f"capacity: {inst['capacity']}",
            "items:",
        ]
        for it in inst["items"]:
            doc_lines.append(f"- {{weight: {it['weight']}, value: {it['value']}}}")
        yaml_docs.append("\n".join(doc_lines))

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        for idx, content in enumerate(yaml_docs, start=1):
            out_path = os.path.join(output_folder, f"{file_prefix}_{idx}.yaml")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)

    return yaml_docs


if __name__ == "__main__":
    import sys
    config_dir = pathlib.Path(__file__).resolve().parents[3] / "config"
    folder = str(config_dir / "Unparsed" / "xiang_instances_01_KP")
    out_dir = str(config_dir / "ParsedXiang")
    docs = parse_knapsack_pisinger_xiang(folder, output_folder=out_dir)

