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
				"version": "3.9.7", // Example version
			},
		},
	}

	var allCells []Cell

	// Generate and append cells for each section of the tutorial
	allCells = append(allCells, createMatplotlibIntro()...)
	allCells = append(allCells, createAnatomyOfPlot()...)
	allCells = append(allCells, createBasicPlotting()...)
	allCells = append(allCells, createCustomization()...)
	allCells = append(allCells, createSubplots()...)
	allCells = append(allCells, createOtherPlotTypes()...)
	allCells = append(allCells, createSavingPlots()...)
	allCells = append(allCells, createInteractivity()...)
	allCells = append(allCells, createConclusion()...)

	notebook.Cells = allCells

	outputBytes, err := json.MarshalIndent(notebook, "", "  ")
	if err != nil {
		fmt.Printf("Error marshalling JSON: %v\n", err)
		return
	}

	outputFilePath := "002_matplotlib_deep_dive.ipynb"
	err = os.WriteFile(outputFilePath, outputBytes, 0644)
	if err != nil {
		fmt.Printf("Error writing to file %s: %v\n", outputFilePath, err)
		return
	}

	fmt.Printf("Successfully generated Matplotlib tutorial and saved to %s\n", outputFilePath)
}

// Helper functions to create cells for each section

func createMatplotlibIntro() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"# Matplotlib: A Deep Dive Tutorial\n\n",
				"Welcome to this comprehensive guide to Matplotlib, the foundational plotting library for Python. This notebook will take you from the basic building blocks of a plot to more advanced customization and interactivity.\n\n",
				"We will cover:\n",
				"1. The Anatomy of a Plot (Figures and Axes)\n",
				"2. Basic Plotting (Line, Scatter, Bar)\n",
				"3. Customizing Your Plots\n",
				"4. Working with Multiple Plots (Subplots)\n",
				"5. Common Statistical Plot Types\n",
				"6. Saving Your Plots\n",
				"7. Creating Interactive Plots",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"# First, let's import the necessary libraries\n",
				"import matplotlib.pyplot as plt\n",
				"import numpy as np\n\n",
				"# This magic command ensures that plots are displayed inline in the notebook\n",
				"%matplotlib inline",
			},
		},
	}
}

func createAnatomyOfPlot() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## 1. The Anatomy of a Plot\n\n",
				"Understanding Matplotlib's structure is key. The two most important components are the **Figure** and the **Axes**.\n\n",
				"- **Figure**: The top-level container for everything. It's the overall window or page that everything is drawn on.\n",
				"- **Axes**: This is what we think of as 'the plot'. It's the region of the image with the data space. A figure can contain one or more axes.\n\n",
				"The best practice is to explicitly create a Figure and one or more Axes and then call methods on them. This is known as the **Object-Oriented API**.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"# Create a Figure and a single Axes object\n",
				"fig, ax = plt.subplots()\n\n",
				"# Now, 'ax' is our plotting area. We can call methods on it.\n",
				"ax.plot([1, 2, 3, 4], [10, 20, 25, 30])\n\n",
				"# Display the plot\n",
				"plt.show()",
			},
		},
	}
}

func createBasicPlotting() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## 2. Basic Plotting Commands\n\n",
				"Let's explore the most common plot types.",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"### Line Plots\n",
				"Used to visualize data that changes over a continuous interval.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"x = np.linspace(0, 10, 100) # 100 points from 0 to 10\n",
				"y = np.sin(x)\n\n",
				"fig, ax = plt.subplots()\n",
				"ax.plot(x, y)\n",
				"plt.show()",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"### Scatter Plots\n",
				"Ideal for showing the relationship between two individual data points.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"x = np.random.randn(50)\n",
				"y = np.random.randn(50)\n",
				"colors = np.random.rand(50)\n",
				"sizes = 1000 * np.random.rand(50)\n\n",
				"fig, ax = plt.subplots()\n",
				"ax.scatter(x, y, c=colors, s=sizes, alpha=0.5)\n",
				"plt.show()",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"### Bar Charts\n",
				"Used to compare quantities for different categories.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"categories = ['A', 'B', 'C', 'D']\n",
				"values = [23, 45, 55, 12]\n\n",
				"fig, ax = plt.subplots()\n",
				"ax.bar(categories, values)\n",
				"plt.show()",
			},
		},
	}
}

func createCustomization() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## 3. Customizing Your Plots\n\n",
				"A default plot is good, but a great plot is customized for clarity and aesthetics. This includes labels, titles, colors, and legends.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"x = np.linspace(0, 2 * np.pi, 50)\n\n",
				"fig, ax = plt.subplots(figsize=(10, 6)) # Control figure size\n\n",
				"# Plotting two lines on the same axes\n",
				"ax.plot(x, np.sin(x), color='blue', linestyle='--', marker='o', label='Sine')\n",
				"ax.plot(x, np.cos(x), color='red', linestyle='-', marker='x', label='Cosine')\n\n",
				"# Adding labels and title\n",
				"ax.set_title('Sine and Cosine Waves', fontsize=16)\n",
				"ax.set_xlabel('X-axis (radians)', fontsize=12)\n",
				"ax.set_ylabel('Y-axis (value)', fontsize=12)\n\n",
				"# Setting axis limits\n",
				"ax.set_xlim(0, 2 * np.pi)\n",
				"ax.set_ylim(-1.5, 1.5)\n\n",
				"# Adding a grid\n",
				"ax.grid(True, linestyle=':', alpha=0.6)\n\n",
				"# Adding a legend\n",
				"ax.legend(loc='upper right')\n\n",
				"plt.show()",
			},
		},
	}
}

func createSubplots() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## 4. Working with Multiple Plots (Subplots)\n\n",
				"Often, you need to display multiple plots in a single figure to compare them. The `plt.subplots()` function is perfect for this, creating a grid of axes.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"# Create a 2x2 grid of subplots\n",
				"fig, axs = plt.subplots(2, 2, figsize=(12, 10))\n\n",
				"# The 'axs' object is a 2D numpy array of Axes objects.\n",
				"# We can access them by index.\n\n",
				"# Top-left plot\n",
				"axs[0, 0].plot(np.random.randn(50).cumsum(), 'k--') # k-- is black dashed line\n",
				"axs[0, 0].set_title('Top-Left')\n\n",
				"# Top-right plot\n",
				"axs[0, 1].scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))\n",
				"axs[0, 1].set_title('Top-Right')\n\n",
				"# Bottom-left plot\n",
				"axs[1, 0].bar(['A', 'B', 'C'], [3, 4, 2])\n",
				"axs[1, 0].set_title('Bottom-Left')\n\n",
				"# Bottom-right plot\n",
				"axs[1, 1].hist(np.random.randn(1000), bins=20, color='steelblue')\n",
				"axs[1, 1].set_title('Bottom-Right')\n\n",
				"# Add a single title for the entire figure\n",
				"fig.suptitle('A Figure with Four Subplots', fontsize=20)\n\n",
				"plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle\n",
				"plt.show()",
			},
		},
	}
}

func createOtherPlotTypes() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## 5. Common Statistical Plot Types",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"### Histograms\n",
				"Excellent for visualizing the distribution of a single numerical variable.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"data = np.random.normal(0, 1, 10000) # 10000 points from a normal distribution\n\n",
				"fig, ax = plt.subplots()\n",
				"ax.hist(data, bins=50, density=True, alpha=0.7, color='g')\n",
				"ax.set_title('Histogram of a Normal Distribution')\n",
				"plt.show()",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"### Box Plots\n",
				"A great way to summarize the distribution of data, showing the median, quartiles, and potential outliers.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"data1 = np.random.normal(0, 1, 100)\n",
				"data2 = np.random.normal(2, 1.5, 100)\n",
				"data3 = np.random.uniform(-2, 5, 100)\n",
				"data_to_plot = [data1, data2, data3]\n\n",
				"fig, ax = plt.subplots()\n",
				"ax.boxplot(data_to_plot, patch_artist=True, tick_labels=['Normal', 'Wide Normal', 'Uniform'])\n",
				"ax.set_title('Comparison of Distributions with Box Plots')\n",
				"plt.show()",
			},
		},
	}
}

func createSavingPlots() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## 6. Saving Your Plots\n\n",
				"You can easily save your figures to a file using `plt.savefig()`.\n\n",
				"Common formats include `.png`, `.jpg`, `.pdf`, and `.svg` (for scalable vector graphics).",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"x = np.linspace(0, 10, 100)\n",
				"y = np.cos(x)\n\n",
				"fig, ax = plt.subplots()\n",
				"ax.plot(x, y)\n",
				"ax.set_title('A Saved Plot')\n\n",
				"# Save the figure before showing it\n",
				"# dpi controls the resolution (dots per inch)\n",
				"fig.savefig('my_saved_plot.png', dpi=300, bbox_inches='tight')\n\n",
				"print('Plot has been saved as my_saved_plot.png')",
			},
		},
	}
}

func createInteractivity() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## 7. Creating Interactive Plots\n\n",
				"Using the `ipywidgets` library, you can create interactive controls that are linked to your plotting functions. This allows you to explore parameters and see their effects in real time.\n\n",
				"**Note:** You may need to install the library first: `!pip install ipywidgets`",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"from ipywidgets import interact, FloatSlider\n\n",
				"def plot_sine_wave(amplitude, frequency, phase):\n",
				"    \"\"\"Plots a sine wave with adjustable parameters.\"\"\"\n",
				"    x = np.linspace(0, 2 * np.pi, 200)\n",
				"    y = amplitude * np.sin(frequency * x + phase)\n",
				"    \n",
				"    fig, ax = plt.subplots(figsize=(8, 5))\n",
				"    ax.plot(x, y)\n",
				"    ax.set_ylim(-5.5, 5.5)\n",
				"    ax.set_title(f'y = {amplitude:.1f} * sin({frequency:.1f}x + {phase:.1f})')\n",
				"    ax.grid(True)\n",
				"    plt.show()\n\n",
				"# The interact function creates a UI for each function argument.\n",
				"# We can define the widgets and their ranges.\n",
				"interact(\n",
				"    plot_sine_wave, \n",
				"    amplitude=FloatSlider(min=0.5, max=5.0, step=0.1, value=1.0),\n",
				"    frequency=FloatSlider(min=1.0, max=10.0, step=0.5, value=1.0),\n",
				"    phase=FloatSlider(min=0, max=2*np.pi, step=0.1, value=0)\n",
				");",
			},
		},
	}
}

func createConclusion() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"## Conclusion\n\n",
				"You've now seen the core components of Matplotlib, from creating simple plots to customizing them, arranging them in grids, and even making them interactive.\n\n",
				"This library is incredibly deep. From here, you can explore:\n",
				"- **3D plotting** with `mplot3d`.\n",
				"- **Animations**.\n",
				"- Higher-level libraries like **Seaborn** and **Plotly** that are built on top of Matplotlib and simplify the creation of complex statistical plots.",
			},
		},
	}
}
