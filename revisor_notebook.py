"""Revisor de notebooks de la Practica 3. SOLO AVISA: nunca modifica el notebook.

Uso, desde la raiz del repo:
    python revisor_notebook.py notebooks/R3_p3_storage.ipynb

Da dos listas:
    ERRORES -> incumplen el enunciado y pueden quitar nota. Hay que arreglarlos.
    AVISOS  -> codigo que puede costar explicar, o cosas que conviene revisar.
               Un aviso no es un fallo: si sabes explicar esa linea, puedes dejarla.

No comprueba si los resultados estan bien ni si entiendes el codigo.
"""
import ast
import json
import re
import sys

MAX_LINE = 100
MAX_CELL_LINES = 40
MAX_FUNCTION_LINES = 25
MAX_CHAINED_CALLS = 3

# Librerias que ya aparecen en el enunciado y en la Practica 2
USUAL_IMPORTS = ["numpy", "pandas", "matplotlib", "sklearn", "json", "time"]

# Herramientas que suelen costar mas de explicar
HARD_FUNCTIONS = ["map", "filter", "reduce", "eval", "exec", "getattr", "setattr"]
HARD_PANDAS = ["assign", "pipe", "transform", "agg", "aggregate", "apply", "melt",
               "pivot_table", "explode", "stack", "unstack", "rename_axis", "reindex"]
HARD_SKLEARN = ["GridSearchCV", "RandomizedSearchCV", "make_scorer", "BaseEstimator",
                "TransformedTargetRegressor", "PolynomialFeatures"]

FOLD_BUILDERS = ["KFold", "StratifiedKFold", "GroupKFold", "StratifiedGroupKFold",
                 "ShuffleSplit", "train_test_split"]
BAD_ENCODERS = ["OrdinalEncoder", "LabelEncoder"]
FORBIDDEN_COLUMNS = ["row_id", "feature_group", "config_group", "fold_id", "iops"]
SPANISH_WORDS = ["para", "porque", "los", "las", "una", "que", "este", "esta", "con",
                 "aqui", "aquí", "del", "por", "como", "pero"]


def call_name(node):
    """Name of the function or method being called."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def keyword_text(node, name):
    """Source text of a keyword argument, or None if it is not there."""
    for keyword in node.keywords:
        if keyword.arg == name:
            return ast.unparse(keyword.value)
    return None


def chained_calls(node):
    """How many method calls are chained, like a().b().c()."""
    count = 0
    while isinstance(node, ast.Call):
        count = count + 1
        if not isinstance(node.func, ast.Attribute):
            break
        node = node.func.value
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
    return count


def check_code_tree(tree, add_error, add_warning):
    """Look at the structure of one code cell."""
    long_chains = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            add_warning("usa lambda: mejor una funcion normal o un bucle")
        if isinstance(node, ast.IfExp):
            add_warning("if de una sola linea (a if x else b): mejor un if normal")
        if isinstance(node, ast.NamedExpr):
            add_warning("usa el operador :=")
        if isinstance(node, ast.ClassDef):
            add_warning("define una clase (" + node.name + "): en esta practica no hace falta")
        if isinstance(node, ast.Try):
            add_warning("usa try/except: puede esconder errores")
        if isinstance(node, ast.Global):
            add_warning("usa global")

        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            n_ifs = 0
            for generator in node.generators:
                n_ifs = n_ifs + len(generator.ifs)
            if len(node.generators) > 1 or n_ifs > 0:
                add_warning("comprehension con varios for o con if: mejor un bucle normal")
            for child in ast.walk(node):
                if child is not node and isinstance(child, (ast.ListComp, ast.DictComp, ast.SetComp)):
                    add_warning("comprehension dentro de otra: mejor bucles normales")

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            size = node.end_lineno - node.lineno + 1
            if size > MAX_FUNCTION_LINES:
                add_warning("la funcion " + node.name + " tiene " + str(size) + " lineas: partela")
            if node.decorator_list:
                add_warning("la funcion " + node.name + " usa un decorador")
            if node.args.vararg or node.args.kwarg:
                add_warning("la funcion " + node.name + " usa *args o **kwargs")
            for child in ast.walk(node):
                if child is not node and isinstance(child, ast.FunctionDef):
                    add_warning("funcion dentro de otra funcion (" + child.name + ")")

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            else:
                roots = [(node.module or "").split(".")[0]]
            for root in roots:
                if root not in USUAL_IMPORTS:
                    add_warning("importa " + root + ": comprueba que lo habeis visto")

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base = ast.unparse(node.left)
            power = ast.unparse(node.right).lower()
            if base == "10" and "mae" in power:
                add_warning("eleva 10 a un MAE: solo vale si lo llamas factor multiplicativo, nunca error en ns")

        if isinstance(node, ast.Assign):
            names = " ".join(ast.unparse(target) for target in node.targets).upper()
            if "FEATURE" in names or "NOMINAL" in names or "NUMERIC" in names:
                for child in ast.walk(node.value):
                    if isinstance(child, ast.Constant) and child.value in FORBIDDEN_COLUMNS:
                        add_error("la columna " + str(child.value) + " no puede entrar al modelo")

        if isinstance(node, ast.Call):
            name = call_name(node)
            if name in HARD_FUNCTIONS:
                add_warning("usa " + name + "(): mejor un bucle normal")
            if name in HARD_PANDAS:
                add_warning("usa ." + name + "() de pandas: comprueba que sabes explicarlo")
            if name in HARD_SKLEARN:
                add_warning("usa " + name + ": comprueba que lo habeis visto")
            if name == "fit_transform":
                add_warning("fit_transform fuera de un Pipeline: comprueba que no se ajusta antes del fold")
            if name in FOLD_BUILDERS:
                add_error("usa " + name + ": en la P3 los folds se leen de los CSV oficiales")
            if name in BAD_ENCODERS:
                add_error("usa " + name + ": las categoricas van con one-hot, nunca con enteros")
            if name == "OneHotEncoder":
                if keyword_text(node, "drop") is None:
                    add_error("OneHotEncoder sin drop=: hay que declarar el nivel de referencia")
                if keyword_text(node, "handle_unknown") is None:
                    add_warning("OneHotEncoder sin handle_unknown: declara que pasa con un nivel no visto")
            if name == "LogisticRegression":
                if keyword_text(node, "C") != "np.inf":
                    add_error("LogisticRegression sin C=np.inf: la penalizacion queda por defecto")
                if keyword_text(node, "class_weight") != "None":
                    add_error("LogisticRegression sin class_weight=None escrito")
            chain = chained_calls(node)
            if chain > MAX_CHAINED_CALLS:
                long_chains.append(chain)
    if long_chains:
        add_warning(str(max(long_chains)) + " metodos encadenados en una linea: separalos en pasos")


def check_code_text(source, add_error, add_warning):
    """Look at the plain text of one code cell."""
    lines = source.split("\n")
    if len(lines) > MAX_CELL_LINES:
        add_warning("la celda tiene " + str(len(lines)) + " lineas: partela en varias")
    for number, line in enumerate(lines, start=1):
        if len(line) > MAX_LINE:
            add_warning("linea " + str(number) + " muy larga (" + str(len(line)) + " caracteres)")
        if "TODO" in line:
            add_warning("linea " + str(number) + " todavia tiene un TODO")
    if re.search(r"[\"'][A-Za-z]:[\\/]|[\"']/(Users|home)/", source):
        add_error("ruta absoluta: las rutas tienen que ser relativas")


def check_language(text, add_warning):
    """Rough check: several common Spanish words in the same cell."""
    words = set(re.findall(r"[a-záéíóúñ]+", text.lower()))
    found = [word for word in SPANISH_WORDS if word in words]
    if len(found) >= 3:
        add_warning("parece estar en espanol (" + ", ".join(found[:4]) + "): lo entregable va en ingles")


def check_markdown(source, add_warning):
    """Look at the text of one Markdown cell."""
    if "TODO" in source:
        add_warning("todavia tiene un TODO")
    if re.search(r"significan", source, re.IGNORECASE):
        add_warning("menciona 'significant': la variacion entre folds no demuestra nada, revisa la frase")
    pattern = r"(scikit-learn|sklearn)[^.]{0,50}(chose|chooses|stores?|holds?|keeps?)[^.]{0,50}reference class"
    if re.search(pattern, source, re.IGNORECASE):
        add_warning("parece decir que sklearn guarda una clase de referencia: revisa la frase")
    if re.search(r"times more likely", source, re.IGNORECASE):
        add_warning("dice 'times more likely': un odds ratio no es un cociente de probabilidades (tema 7)")
    if re.search(r"\bcaus(es|ed|al)\b", source, re.IGNORECASE):
        add_warning("habla de causa: un coeficiente describe una asociacion, no una causa (tema 7)")
    check_language(source, add_warning)


def strip_notebook_commands(source):
    """Blank the !commands and %magics, which are not Python."""
    clean = []
    for line in source.split("\n"):
        if line.lstrip().startswith(("!", "%")):
            clean.append("")
        else:
            clean.append(line)
    return "\n".join(clean)


def check_first_cell(cells, errors):
    if not cells or cells[0]["cell_type"] != "code":
        errors.append("Celda 1: la primera celda tiene que ser de codigo, con la semilla")
        return
    source = "".join(cells[0]["source"])
    if "PROJECT_RANDOM_STATE = 2627" not in source or "print" not in source:
        errors.append("Celda 1: tiene que tener PROJECT_RANDOM_STATE = 2627 y un print")


def check_execution(code_cells, errors, warnings):
    """Stored outputs: errors, and whether it was run top to bottom."""
    counts = [cell.get("execution_count") for cell in code_cells]
    if any(count is None for count in counts):
        warnings.append("General: hay celdas sin ejecutar. Solo puntua lo que aparece en las salidas")
    else:
        if counts != list(range(1, len(counts) + 1)):
            warnings.append("General: las celdas no estan en orden. Reinicia y ejecuta todo")
    for cell in code_cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append("General: hay una salida con error (" + output.get("ename", "") + ")")


def review(path):
    with open(path, encoding="utf-8") as notebook_file:
        notebook = json.load(notebook_file)
    cells = notebook["cells"]
    errors = []
    warnings = []
    check_first_cell(cells, errors)

    code_cells = []
    for position, cell in enumerate(cells, start=1):
        source = "".join(cell["source"])
        label = "Celda " + str(position) + ": "

        def add_error(message):
            errors.append(label + message)

        def add_warning(message):
            warnings.append(label + message)

        if cell["cell_type"] == "markdown":
            check_markdown(source, add_warning)
        elif cell["cell_type"] == "code":
            code_cells.append(cell)
            check_code_text(source, add_error, add_warning)
            comments = " ".join(re.findall(r"#(.*)", source))
            check_language(comments, add_warning)
            try:
                tree = ast.parse(strip_notebook_commands(source))
            except SyntaxError:
                add_error("no se ha podido leer el codigo (error de sintaxis)")
                continue
            check_code_tree(tree, add_error, add_warning)
    check_execution(code_cells, errors, warnings)
    # The same message can appear several times in one cell
    return list(dict.fromkeys(errors)), list(dict.fromkeys(warnings))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 2:
        print("Uso: python revisor_notebook.py notebooks/mi_notebook.ipynb")
        return 2
    errors, warnings = review(sys.argv[1])
    print("Notebook:", sys.argv[1])
    print("\nERRORES (hay que arreglarlos):", len(errors))
    for message in errors:
        print("  -", message)
    print("\nAVISOS (revisalos: si sabes explicarlo, puedes dejarlo):", len(warnings))
    for message in warnings:
        print("  -", message)
    print("\nEl notebook no se ha modificado. Si cambias codigo, reinicia, ejecuta todo")
    print("y comprueba que los resultados no han cambiado.")
    if errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
