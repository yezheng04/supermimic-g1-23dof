"""Length-bucketed scheduling without importing CUDA into CPU workers."""


def batches(rows, batch_size=64, frame_budget=131072):
    if batch_size < 1 or frame_budget < 1:
        raise ValueError("Batch size and frame budget must be positive")
    ordered = sorted(rows, key=lambda r: (int(r["source_frames"]), r["name"]))
    current = []
    for row in ordered:
        frames = int(row["source_frames"])
        if frames < 2:
            raise ValueError(f"Motion has fewer than two source frames: {row['name']}")
        if current and (len(current) >= batch_size or frames * (len(current) + 1) > frame_budget):
            yield current
            current = []
        current.append(row)
    if current:
        yield current
