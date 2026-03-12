import glob, re

files = glob.glob('blender_importer/**/*.py', recursive=True)
for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    def replacer(match):
        indent = match.group(1)
        return indent + 'except Exception as e:\\n' + indent + '  print(f"Warning in ' + filepath.replace('\\\\', '/') + ': {e}")'

    new_content, count = re.subn(r'([ \t]*)except Exception:\r?\n\1[ \t]+pass', replacer, content)

    if count > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Replaced {count} instances in {filepath}')
