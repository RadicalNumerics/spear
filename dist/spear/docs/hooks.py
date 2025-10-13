import logging
import os

log = logging.getLogger("mkdocs")

# Global variable to store the paths of created __init__.py files
created_files = []


def on_startup(*args, **kwargs):
    log.info("Creating missing __init__.py files in spear package")
    for subdir, dirs, files in os.walk("spear"):
        if "__init__.py" not in files:
            init_file_path = os.path.join(subdir, "__init__.py")
            with open(init_file_path, "w"):
                pass  # empty file
            created_files.append(init_file_path)
    log.info(f"{len(created_files)} __init__.py files created")

    with open("README.md") as f:
        readme_content = f.read()
    with open("docs/overrides/fancy-first-page.txt") as f:
        fancy_first_page_content = f.read()

    full_content = fancy_first_page_content + readme_content
    log.info("Replacing relative paths from docs/assets[...] to ./assets[...]")
    full_content = full_content.replace('src="docs/assets', 'src="./assets')
    with open("docs/index.md", "w") as f:
        f.write(full_content)

    log.info("Generated docs/index.md by prepending fancy-first-page.txt to README.md")


def on_shutdown(*args, **kwargs):
    log.info(f"Removing {len(created_files)} created __init__.py files")
    for file_path in created_files:
        if os.path.exists(file_path):
            os.remove(file_path)
