import os


def get_all_files(directory: str):
    files: list[str] = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            print(filename)
            files.append(os.path.join(root, filename))
    return files


if __name__ == "__main__":
    directory = "./data/"
    all_files = get_all_files(directory)
    for file in all_files:
        print(file)
