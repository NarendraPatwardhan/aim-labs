package main

import (
	"encoding/json"
	"fmt"
	"os"
)

// Notebook represents the top-level structure of a .ipynb file.
type Notebook struct {
	Cells         []Cell         `json:"cells"`
	Metadata      map[string]any `json:"metadata"`
	NBFormat      int            `json:"nbformat"`
	NBFormatMinor int            `json:"nbformat_minor"`
}

// Cell represents a single cell in a notebook.
type Cell struct {
	CellType       string         `json:"cell_type"`
	ExecutionCount *int           `json:"execution_count,omitempty"`
	Metadata       map[string]any `json:"metadata"`
	Outputs        []Output       `json:"outputs,omitempty"`
	Source         []string       `json:"source"`
}

// Output represents the output of a code cell.
type Output struct {
	OutputType     string         `json:"output_type"`
	Name           string         `json:"name,omitempty"`
	Text           []string       `json:"text,omitempty"`
	Data           map[string]any `json:"data,omitempty"`
	ExecutionCount *int           `json:"execution_count,omitempty"`
}

// main function orchestrates the notebook generation.
func main() {
	// Initialize a new notebook structure
	notebook := Notebook{
		NBFormat:      4,
		NBFormatMinor: 5,
		Metadata: map[string]any{
			"kernelspec": map[string]string{
				"display_name": "Python 3 (ipykernel)",
				"language":     "python",
				"name":         "python3",
			},
			"language_info": map[string]any{
				"name":    "python",
				"version": "3.9.7",
			},
		},
	}

	// A slice to hold all the cells we generate
	var allCells []Cell

	// Generate and append cells for each feature in order
	allCells = append(allCells, createIntroCells()...)
	allCells = append(allCells, createMarkdownCells()...)
	allCells = append(allCells, createLatexCells()...)
	allCells = append(allCells, createCodeExecutionCells()...)
	allCells = append(allCells, createRichMediaCells()...)
	allCells = append(allCells, createMagicCommandsCells()...)
	allCells = append(allCells, createShellCommandCells()...)
	allCells = append(allCells, createInteractiveWidgetCells()...)
	allCells = append(allCells, createKernelCells()...)
	allCells = append(allCells, createSharingCells()...)
	allCells = append(allCells, createTabCompletionCells()...)
	allCells = append(allCells, createDebuggingCells()...)
	allCells = append(allCells, createConclusionCells()...)

	notebook.Cells = allCells

	// Marshal the notebook struct into pretty-printed JSON
	outputBytes, err := json.MarshalIndent(notebook, "", "  ")
	if err != nil {
		fmt.Printf("Error marshalling JSON: %v\n", err)
		return
	}

	// Write the JSON to a .ipynb file
	outputFilePath := "001_features_showcase.ipynb"
	err = os.WriteFile(outputFilePath, outputBytes, 0644)
	if err != nil {
		fmt.Printf("Error writing to file %s: %v\n", outputFilePath, err)
		return
	}

	fmt.Printf("Successfully generated notebook and saved to %s\n", outputFilePath)
}

// Helper functions to create cells for each feature

func createIntroCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"# A Tour of Jupyter Notebook Features\n\n",
				"This notebook demonstrates the key features that make Jupyter Notebooks a powerful tool for interactive computing, data science, and documentation.",
			},
		},
	}
}

func createMarkdownCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## 1. Markdown Rendering\n\n",
				"Jupyter allows you to write rich text using Markdown. This is perfect for documenting your code, explaining your methodology, and structuring your analysis. Double-click this cell to see the raw Markdown code.",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"### Examples:\n\n",
				"- **Bold** and *Italic* text.\n",
				"- `Code snippets` can be highlighted.\n",
				"- Numbered and bulleted lists.\n",
				"- [Links to websites](https://jupyter.org)\n",
				"- Blockquotes:\n",
				"> To be, or not to be, that is the question.",
			},
		},
	}
}

func createLatexCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## 2. LaTeX Equation Support\n\n",
				"For scientific and academic work, the ability to render complex mathematical equations is essential. Jupyter uses MathJax to render LaTeX equations directly in Markdown cells.",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"### Examples:\n\n",
				"- **Inline equation:** Use single dollar signs for inline math, like the famous mass-energy equivalence formula: $E = mc^2$.\n\n",
				"- **Display equation:** Use double dollar signs to display an equation on its own line, centered:\n\n",
				"$$\\frac{1}{\\pi} = \\frac{2\\sqrt{2}}{9801} \\sum_{k=0}^{\\infty} \\frac{(4k)!(1103+26390k)}{(k!)^4 396^{4k}}$$\n",
			},
		},
	}
}

func createCodeExecutionCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## 3. Interactive Code Execution\n\n",
				"This is the core feature. You can write and execute code in any supported language (like Python) in a cell. The output appears directly below. Select the cell below and press `Shift+Enter` to run it.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"import time\n\n",
				"for i in range(5):\n",
				"    print(f'Counting... {i+1}')\n",
				"    time.sleep(0.5)\n\n",
				"print('Done!')",
			},
		},
	}
}

func createRichMediaCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## 4. Rich Media & Visualization\n\n",
				"Notebooks can render rich media output directly inline, including plots, tables, and images. Run the cell below to generate a plot using the `matplotlib` library.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"# Ensure you have matplotlib installed: !pip install matplotlib\n",
				"import matplotlib.pyplot as plt\n\n",
				"x = [i for i in range(10)]\n",
				"y = [i**2 for i in x]\n\n",
				"plt.figure(figsize=(8, 5))\n",
				"plt.plot(x, y)\n",
				"plt.title('A Simple Quadratic Plot')\n",
				"plt.xlabel('x values')\n",
				"plt.ylabel('y values (x^2)')\n",
				"plt.grid(True)\n",
				"plt.show()",
			},
		},
	}
}

func createMagicCommandsCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## 5. Magic Commands\n\n",
				"Magic commands are special directives, prefixed with `%` or `%%`, that are not part of the Python language but provide powerful extensions to the notebook environment.",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"### `%lsmagic` - List all available magics",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"%lsmagic",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"### `%%timeit` - Measure execution time\n\n",
				"Use `%%timeit` at the top of a cell to automatically run the code multiple times and get a precise measurement of its execution time.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"%%timeit\n",
				"total = 0\n",
				"for i in range(10000):\n",
				"    total += i",
			},
		},
	}
}

func createShellCommandCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## 6. Shell Command Integration\n\n",
				"You can run any shell command by prefixing it with an exclamation mark (`!`). This is extremely useful for file management, package installation, or checking system information without leaving your notebook.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"# The command below lists files in the current directory.\n",
				"# On Windows, you might use `!dir` instead.\n",
				"!ls -la",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"You can even assign the output of a command to a Python variable:",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"files = !ls\n",
				"print(f'Found {len(files)} files/folders.')\n",
				"print(files)",
			},
		},
	}
}

func createInteractiveWidgetCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## 7. Interactive Widgets (ipywidgets)\n\n",
				"You can create interactive controls like sliders, dropdowns, and text boxes to manipulate your code and visualizations in real time. This is fantastic for building simple UIs and dashboards.\n\n",
				"**Note:** You may need to install the library first by running `!pip install ipywidgets` in a cell.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"from ipywidgets import interact\n\n",
				"def plot_power(power):\n",
				"    import numpy as np\n",
				"    import matplotlib.pyplot as plt\n",
				"    x = np.linspace(0, 2, 100)\n",
				"    y = x ** power\n",
				"    plt.figure(figsize=(6,4))\n",
				"    plt.plot(x, y)\n",
				"    plt.title(f'Plot of y = x^{power}')\n",
				"    plt.show()\n\n",
				"# The interact function automatically creates a slider for the 'power' argument.\n",
				"interact(plot_power, power=(0.1, 5.0, 0.1));",
			},
		},
	}
}

func createKernelCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## 8. Multi-Language Support (Kernels)\n\n",
				"While this notebook is running a Python kernel, Jupyter is language-agnostic. It supports kernels for dozens of languages, including **R, Julia, Scala, and SQL**.\n\n",
				"This feature is managed through the Jupyter UI. You can switch the kernel for a notebook by going to the **Kernel > Change kernel** menu.\n\n",
				"Because this depends on your local Jupyter setup, we can't demonstrate it in a code cell, but it's a fundamental part of the Jupyter architecture.",
			},
		},
	}
}

func createSharingCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## 9. Easy Sharing and Exporting\n\n",
				"Notebooks (`.ipynb` files) are just JSON files, making them easy to share and version control. You can also export them to a variety of static formats.\n\n",
				"### Manual Export\n",
				"In the Jupyter interface, you can use the **File > Download as** menu to export to:\n",
				"- HTML\n",
				"- PDF (requires a LaTeX installation)\n",
				"- Markdown\n",
				"- Python Script (`.py`)\n\n",
				"### Programmatic Export with `nbconvert`\n",
				"You can also use the command-line tool `jupyter nbconvert` to automate this process. For example, to convert this notebook to HTML, you would run the following command in your terminal:\n\n",
				"```bash\n",
				"jupyter nbconvert --to html 001_features_showcase.ipynb\n",
				"```\n\n",
				"This is very powerful for automatically generating reports from your notebooks.",
			},
		},
	}
}

func createTabCompletionCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## 10. Tab Completion and Help\n\n",
				"Jupyter provides powerful introspection features to make coding easier.\n\n",
				"### Tab Completion\n",
				"In the cell below, place your cursor after `random.rand` and press `Tab`. Jupyter will show you the available functions that start with that prefix (like `randint` and `randrange`).\n",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"import random\n\n",
				"# Place cursor after 'rand' and press Tab\n",
				"random.rand",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"### Getting Help\n",
				"You can get instant help on any object, function, or method by appending a question mark `?` and running the cell. This will open a help pane at the bottom of the screen with the docstring.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"import pandas as pd\n\n",
				"# Run this cell to get help on the read_csv function\n",
				"pd.read_csv?",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"Using two question marks (`??`) will show the full source code if it's available.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"def my_simple_function():\n",
				"    \"\"\"This is a simple function.\"\"\"\n",
				"    return 42\n\n",
				"# Run this cell to see the source code\n",
				"my_simple_function??",
			},
		},
	}
}

func createDebuggingCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## 11. Debugging Capabilities\n\n",
				"Jupyter has a built-in interactive debugger.\n\n",
				"First, let's run a cell that will produce an error.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"def buggy_function(x):\n",
				"    y = 'hello'\n",
				"    # This will cause a TypeError\n",
				"    return x + y\n\n",
				"buggy_function(5)",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"After the error occurs in the cell above, run the cell below containing the `%debug` magic command. This will open an interactive debugging prompt (`ipdb`) right here in the notebook. You can inspect variables, step through the code, and figure out what went wrong.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"# Run this cell AFTER the cell above produces an error\n",
				"%debug",
			},
		},
	}
}

func createConclusionCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## 12. Notebook Extensions\n\n",
				"The Jupyter ecosystem is highly extensible. You can install community-developed extensions to add features like:\n\n",
				"- A table of contents\n",
				"- Code auto-formatting (e.g., Black or YAPF)\n",
				"- Spellchecking for markdown cells\n",
				"- Variable inspectors\n\n",
				"These are typically managed through the `jupyter_contrib_nbextensions` package. This allows you to tailor your notebook environment to your exact needs.",
			},
		},
	}
}
