// Run Python scripts in Pyodide under Node.js with a pyAUSAXS Pyodide wheel installed, the same way the website runs them.
//
// Usage: node run_in_pyodide.mjs <pyausaxs wheel> <script.py>...
//
// The repository's tests/files folder is mounted at tests/files, relative to the working directory, so the example
// scripts find their data. Each script runs in a fresh namespace with the backend settings reset, as a page running
// several examples in one Pyodide instance must also do; the first failure exits with a nonzero status.

import { loadPyodide } from "pyodide";
import { readFileSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const [wheel, ...scripts] = process.argv.slice(2);
if (!wheel || scripts.length === 0) {
    console.error("usage: node run_in_pyodide.mjs <pyausaxs wheel> <script.py>...");
    process.exit(2);
}

const here = dirname(fileURLToPath(import.meta.url));
const pyodide = await loadPyodide();

const home = "/home/pyodide";
pyodide.FS.mkdirTree(`${home}/tests/files`);
pyodide.mountNodeFS(`${home}/tests/files`, resolve(here, "..", "files"));

const wheelPath = `/tmp/${basename(wheel)}`;
pyodide.FS.writeFile(wheelPath, readFileSync(wheel));
await pyodide.loadPackage("micropip");
await pyodide.runPythonAsync(`
import micropip
await micropip.install("emfs:${wheelPath}")
`);

let failed = false;
for (const script of scripts) {
    const code = readFileSync(script, "utf8");
    console.log(`=== ${script}`);
    try {
        await pyodide.loadPackagesFromImports(code);
        await pyodide.runPythonAsync(`
import os, sys, warnings
import pyausaxs
pyausaxs.settings.reset() # the backend settings are global, so one script's would otherwise carry over into the next
os.chdir("${home}")
if "matplotlib" in sys.modules or ${code.includes("matplotlib") ? "True" : "False"}:
    import matplotlib
    matplotlib.use("Agg") # plt.show() becomes a no-op
    warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")
`);
        const start = performance.now();
        await pyodide.runPythonAsync(code, { globals: pyodide.toPy({ __name__: "__main__" }) });
        console.log(`=== passed (${((performance.now() - start) / 1000).toFixed(1)} s)`);
    } catch (e) {
        console.error(`=== FAILED: ${e}`);
        failed = true;
        break;
    }
}
process.exit(failed ? 1 : 0);
