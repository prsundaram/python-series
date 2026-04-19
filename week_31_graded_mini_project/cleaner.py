import nbformat

nb = nbformat.read("solution.ipynb", as_version=4)

# remove invalid fields
for cell in nb.cells:
    if "outputs" in cell:
        for output in cell["outputs"]:
            output.pop("jetTransient", None)

nbformat.write(nb, "clean_solution.ipynb")