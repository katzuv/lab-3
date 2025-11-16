from pathlib import Path


def get_edited_data_path(path: Path, start_index: int) -> Path:
    data = path.read_text()
    for replacement in (
        ("\t\t", ","),
        ("\t", ","),
        ("	count B	count C	count D", ""),
        ("count A", "ticks"),
        ("#", "line"),
        ("time(s)", "time"),
    ):
        data = data.replace(replacement[0], replacement[1])

    header = data.splitlines()[0]
    data = f"{header}\n" + "\n".join(
        line.removesuffix(",") for line in data.splitlines()[start_index:]
    )  # Trim data before sine.

    new_path = path.with_suffix(".csv")
    new_path.write_text(data)

    return new_path
