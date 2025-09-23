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

	// Generate and append cells for each section in order
	allCells = append(allCells, createIntroAndSetupCells()...)
	allCells = append(allCells, createMonteCarloCells()...)
	allCells = append(allCells, createCellularAutomataCells()...)
	allCells = append(allCells, createFractalCells()...)
	allCells = append(allCells, createConclusionCells()...)

	notebook.Cells = allCells

	// Marshal the notebook struct into pretty-printed JSON
	outputBytes, err := json.MarshalIndent(notebook, "", "  ")
	if err != nil {
		fmt.Printf("Error marshalling JSON: %v\n", err)
		return
	}

	// Write the JSON to a .ipynb file
	outputFilePath := "004_gpu_acceleration.ipynb"
	err = os.WriteFile(outputFilePath, outputBytes, 0644)
	if err != nil {
		fmt.Printf("Error writing to file %s: %v\n", outputFilePath, err)
		return
	}

	fmt.Printf("Successfully generated notebook and saved to %s\n", outputFilePath)
}

// Helper functions to create cells for each section

func createIntroAndSetupCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"# GPU Acceleration with PyTorch: A Tensor Operations Primer"},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"This notebook explores PyTorch's power as an efficient library for tensor operations, which are fundamental to computational modeling in AI and beyond. Rather than focusing on neural networks, we will demonstrate how PyTorch enables high-performance scientific computing through three compelling examples."},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"Tensors are multi-dimensional arrays that generalize scalars (0D), vectors (1D), and matrices (2D). They are ideal for numerical computing because they provide a structured, contiguous memory layout that enables efficient access and operations. Unlike Python lists, tensors support vectorized operations—applying functions to entire arrays at once without explicit loops—which reduces overhead and leverages hardware optimizations. Additionally, features like broadcasting allow tensors of different shapes to be combined implicitly, making code concise and less error-prone."},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"Many real-world problems involve massive datasets or computations. CPUs, with their handful of powerful cores, excel at sequential tasks but bottleneck on parallel workloads. GPUs, originally designed for graphics rendering, have thousands of smaller cores that can execute the same operation on many data elements simultaneously. This massive parallelism is perfect for tensor operations. Without GPUs, large-scale modeling would be impractically slow; with them, we can achieve 10-100x speedups. This phenomenon, where GPU advancements for gaming accidentally empowered parallel computing, is often called the \"hardware lottery\" that propelled modern AI forward."},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"We'll explore these concepts through non-NN use cases: Monte Carlo methods, cellular automata, and fractal generation. We will use only base PyTorch operations and functional modules, steering clear of any `nn` modules to keep the focus on raw tensor capabilities."},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"A key note on GPU performance: The first execution of an operation on the GPU can be slower because PyTorch performs just-in-time (JIT) compilation of CUDA kernels and allocates memory. This setup overhead happens only once. To demonstrate true performance, we will include a \"warmup\" pass—running the computation once without timing it—before measuring."},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"## Section 1: PyTorch Basics – Tensors and Devices"},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"Let's start with a quick setup. We will import the essentials and identify the available compute devices."},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"import torch\n",
				"import numpy as np\n",
				"import matplotlib.pyplot as plt\n",
				"import time  # For timing comparisons\n",
				"\n",
				"# For animation\n",
				"from matplotlib.animation import FuncAnimation\n",
				"from IPython.display import HTML\n",
				"plt.rcParams['animation.html'] = 'jshtml' # Configure for notebook display",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"# Setup our compute devices\n",
				"cpu_device = torch.device(\"cpu\")\n",
				"gpu_device = None\n",
				"if torch.cuda.is_available():\n",
				"    gpu_device = torch.device(\"cuda\")\n",
				"    print(f\"GPU is available: {torch.cuda.get_device_name(0)}\")\n",
				"else:\n",
				"    print(\"GPU not available, all operations will run on CPU.\")",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"PyTorch lets you move tensors to a GPU with `.to(device)`, and operations on that tensor will automatically parallelize. For example, let's time matrix multiplication, a core operation in nearly all scientific computing:"},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"def matmul_example(dev):\n",
				"    A = torch.randn(2000, 2000, device=dev)\n",
				"    B = torch.randn(2000, 2000, device=dev)\n",
				"    C = torch.matmul(A, B)  # Trigger computation\n",
				"    if dev.type == 'cuda':\n",
				"        torch.cuda.synchronize()  # Ensure GPU finishes\n",
				"    return C\n",
				"\n",
				"# Warmup if GPU is available\n",
				"if gpu_device:\n",
				"    _ = matmul_example(dev=gpu_device)\n",
				"\n",
				"# Time on CPU\n",
				"start_cpu = time.time()\n",
				"_ = matmul_example(dev=cpu_device)\n",
				"print(f\"CPU Time: {time.time() - start_cpu:.4f}s\")\n",
				"\n",
				"# Time on GPU (post-warmup), if available\n",
				"if gpu_device:\n",
				"    start_gpu = time.time()\n",
				"    _ = matmul_example(dev=gpu_device)\n",
				"    print(f\"GPU Time: {time.time() - start_gpu:.4f}s\")",
			},
		},
	}
}

func createMonteCarloCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"## Section 2: Monte Carlo Methods – Parallel Sampling"},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"Monte Carlo methods rely on repeated random sampling to obtain numerical results, especially when deterministic solutions are infeasible. A classic example is approximating π by randomly scattering points in a unit square and finding the proportion that falls within an inscribed quarter-circle. This proportion approaches π/4 as the number of points increases."},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"This problem is perfectly suited for parallelization, as each random sample is independent. In PyTorch, we can generate all samples simultaneously in a large tensor and perform the calculations without any loops."},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"def approximate_pi(num_samples=10000000, dev=cpu_device):\n",
				"    # Generate random points: two tensors for x and y coords\n",
				"    x = torch.rand(num_samples, device=dev)\n",
				"    y = torch.rand(num_samples, device=dev)\n",
				"    \n",
				"    # Element-wise operations: compute distance squared\n",
				"    inside = (x**2 + y**2) <= 1.0  # Boolean tensor\n",
				"    \n",
				"    # Reduction: count trues in parallel\n",
				"    num_inside = torch.sum(inside.float())\n",
				"    \n",
				"    pi_approx = (num_inside / num_samples) * 4\n",
				"    if dev.type == 'cuda':\n",
				"        torch.cuda.synchronize()\n",
				"    return pi_approx.item()\n",
				"\n",
				"# Warmup on GPU\n",
				"if gpu_device:\n",
				"    _ = approximate_pi(dev=gpu_device)\n",
				"\n",
				"# Time on CPU\n",
				"start_cpu = time.time()\n",
				"pi_cpu = approximate_pi(dev=cpu_device)\n",
				"print(f\"CPU π ≈ {pi_cpu:.6f}, Time: {time.time() - start_cpu:.4f}s\")\n",
				"\n",
				"# Time on GPU (post-warmup)\n",
				"if gpu_device:\n",
				"    start_gpu = time.time()\n",
				"    pi_gpu = approximate_pi(dev=gpu_device)\n",
				"    print(f\"GPU π ≈ {pi_gpu:.6f}, Time: {time.time() - start_gpu:.4f}s\")",
			},
		},
		{ // RESTORED CHALLENGE
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"As the code demonstrates, more samples yield a more accurate result at the cost of computation time. You can explore this trade-off by re-running the cell above with different values for `num_samples`. For instance, try a smaller value like `100000` and a much larger one like `100000000` to see the effect on both the π estimate and the execution time."},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"### Visualization: The Random Samples"},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"num_viz = 10000\n",
				"x_viz = torch.rand(num_viz)\n",
				"y_viz = torch.rand(num_viz)\n",
				"inside_viz = (x_viz**2 + y_viz**2) <= 1.0\n",
				"\n",
				"plt.figure(figsize=(6, 6))\n",
				"plt.scatter(x_viz[inside_viz], y_viz[inside_viz], color='blue', s=1, label='Inside')\n",
				"plt.scatter(x_viz[~inside_viz], y_viz[~inside_viz], color='red', s=1, label='Outside')\n",
				"plt.gca().set_aspect('equal')\n",
				"plt.title(f\"Monte Carlo π Approximation (Samples: {num_viz})\")\n",
				"plt.legend()\n",
				"plt.show()",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"### Visualization: Convergence of π"},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"The accuracy of the approximation improves with more samples. We can visualize this convergence, which illustrates the Law of Large Numbers."},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"sample_sizes = torch.logspace(3, 7, 20).int()\n",
				"pi_estimates = []\n",
				"\n",
				"for n in sample_sizes:\n",
				"    # Use CPU for this visualization as individual runs are fast\n",
				"    pi_val = approximate_pi(num_samples=n.item(), dev=cpu_device)\n",
				"    pi_estimates.append(pi_val)\n",
				"\n",
				"plt.figure(figsize=(10, 6))\n",
				"plt.plot(sample_sizes, pi_estimates, 'o-', label='π Estimate')\n",
				"plt.axhline(y=np.pi, color='r', linestyle='--', label='Actual π')\n",
				"plt.xscale('log')\n",
				"plt.xlabel('Number of Samples (log scale)')\n",
				"plt.ylabel('Estimated Value of π')\n",
				"plt.title('Convergence of Monte Carlo π Approximation')\n",
				"plt.legend()\n",
				"plt.grid(True, which=\"both\", ls=\"--\")\n",
				"plt.show()",
			},
		},
	}
}

func createCellularAutomataCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"## Section 3: Cellular Automata – Grid-Based Simulations"},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"Cellular automata are models consisting of a grid of cells, each evolving over time according to fixed rules based on its neighbors. They demonstrate how complex, emergent behaviors can arise from simple local interactions. Conway's Game of Life is a famous example."},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source: []string{
				"The rules for the Game of Life are:\n",
				"1.  **Underpopulation:** A live cell with fewer than 2 live neighbors dies.\n",
				"2.  **Survival:** A live cell with 2 or 3 live neighbors stays alive.\n",
				"3.  **Overpopulation:** A live cell with more than 3 live neighbors dies.\n",
				"4.  **Reproduction:** A dead cell with exactly 3 live neighbors becomes alive.",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"This process can be efficiently implemented using a 2D convolution to count the neighbors of every cell simultaneously. The update rules are then applied as a set of parallel, element-wise tensor operations."},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"import torch.nn.functional as F\n",
				"\n",
				"def game_of_life(steps=100, grid_size=100, dev=cpu_device, initial_prob=0.5):\n",
				"    grid = (torch.rand(grid_size, grid_size, device=dev) > initial_prob).float()\n",
				"    kernel = torch.ones(3, 3, device=dev); kernel[1,1] = 0\n",
				"    grids = [grid.clone().cpu().numpy()]\n",
				"    for _ in range(steps):\n",
				"        neighbors = F.conv2d(grid.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1).squeeze()\n",
				"        rule1 = (neighbors == 3) # Reproduction\n",
				"        rule2 = ((neighbors == 2) & (grid == 1)) # Survival\n",
				"        grid = (rule1 | rule2).float()\n",
				"        grids.append(grid.clone().cpu().numpy())\n",
				"    if dev.type == 'cuda': torch.cuda.synchronize()\n",
				"    return grids\n",
				"\n",
				"if gpu_device: _ = game_of_life(dev=gpu_device)\n",
				"start_cpu = time.time()\n",
				"grids_cpu = game_of_life(dev=cpu_device)\n",
				"print(f\"CPU Time: {time.time() - start_cpu:.4f}s\")\n",
				"if gpu_device:\n",
				"    start_gpu = time.time()\n",
				"    grids_gpu = game_of_life(dev=gpu_device)\n",
				"    print(f\"GPU Time: {time.time() - start_gpu:.4f}s\")",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"### Visualization: The Evolution of Life"},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"fig = plt.figure(figsize=(6, 6))\n",
				"im = plt.imshow(grids_cpu[0], cmap='binary')\n",
				"plt.axis('off')\n",
				"plt.title(\"Game of Life Evolution\")\n",
				"\n",
				"def animate(i):\n",
				"    im.set_data(grids_cpu[i])\n",
				"    plt.gca().set_title(f'Step {i}')\n",
				"    return [im]\n",
				"\n",
				"anim = FuncAnimation(fig, animate, frames=len(grids_cpu), interval=50, blit=True)\n",
				"plt.close()\n",
				"HTML(anim.to_jshtml())",
			},
		},
		{ // RESTORED CHALLENGE
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"The `initial_prob` parameter in the `game_of_life` function sets the initial density of live cells, which can dramatically alter the system's evolution. To observe this, you can modify the function call to experiment with different densities. A sparse world (e.g., `initial_prob=0.8`, for a 20% density) often dies out quickly, while a very dense world (e.g., `initial_prob=0.2`, for an 80% density) might collapse into stable patterns or chaos. Try these values and see how the animation changes."},
		},
	}
}

func createFractalCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"## Section 4: Fractal Generation – Iterative Computations"},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"Fractals are geometric shapes that exhibit self-similarity at every scale. The Mandelbrot set is a famous fractal in the complex plane, defined by iterating the function `z = z² + c` for every point `c` on the plane. If the value of `z` remains bounded, the point `c` is in the set."},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"Generating this set is computationally intensive, as the iteration must be performed for every pixel in the final image. However, since the calculation for each point is independent, the entire grid of points can be processed in parallel on a GPU."},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"def mandelbrot(width=800, height=600, x_center=-0.75, y_center=0, zoom=1, max_iter=100, dev=cpu_device):\n",
				"    x_width = 3.5 / zoom\n",
				"    y_height = 2.0 / zoom * height / width\n",
				"    x = torch.linspace(x_center - x_width / 2, x_center + x_width / 2, width, device=dev)\n",
				"    y = torch.linspace(y_center - y_height / 2, y_center + y_height / 2, height, device=dev)\n",
				"    real, imag = torch.meshgrid(x, y, indexing='xy')\n",
				"    c = real + imag * 1j\n",
				"    z = torch.zeros_like(c)\n",
				"    escaped = torch.zeros_like(real, dtype=torch.int)\n",
				"    for i in range(max_iter):\n",
				"        z = z**2 + c\n",
				"        not_escaped = torch.abs(z) <= 2\n",
				"        escaped[not_escaped] = i + 1\n",
				"    if dev.type == 'cuda': torch.cuda.synchronize()\n",
				"    return escaped.cpu().numpy()\n",
				"\n",
				"if gpu_device: _ = mandelbrot(dev=gpu_device)\n",
				"start_cpu = time.time()\n",
				"fractal_cpu = mandelbrot(dev=cpu_device)\n",
				"print(f\"CPU Time: {time.time() - start_cpu:.4f}s\")\n",
				"if gpu_device:\n",
				"    start_gpu = time.time()\n",
				"    fractal_gpu = mandelbrot(dev=gpu_device)\n",
				"    print(f\"GPU Time: {time.time() - start_gpu:.4f}s\")",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"### Visualization: The Mandelbrot Set"},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"plt.figure(figsize=(10, 7))\n",
				"plt.imshow(fractal_cpu, cmap='hot', extent=[-2.5, 1.0, -1.5, 1.5])\n",
				"plt.title(\"Mandelbrot Set\")\n",
				"plt.colorbar(label='Iteration Count for Escape')\n",
				"plt.show()",
			},
		},
		{ // RESTORED CHALLENGE
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"The level of detail in the fractal's boundary is controlled by the `max_iter` parameter. More iterations resolve finer patterns but increase the computational cost. Feel free to experiment by changing this value in the `mandelbrot` function call. A lower value like `50` will render faster but with less detail, while a higher value like `250` will reveal more intricate structures at the cost of longer execution time."},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"### Visualization: Zooming into the Fractal"},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"The most fascinating property of fractals is their infinite complexity. We can zoom into a region on the boundary of the set to reveal intricate, self-repeating patterns. This requires high resolution and many iterations, a task well-suited for a GPU."},
		},
		{
			CellType: "code",
			Metadata: map[string]any{},
			Source: []string{
				"# Coordinates for a visually interesting region ('Seahorse Valley')\n",
				"x_center, y_center = -0.745, 0.186\n",
				"zoom = 200\n",
				"max_iter_zoom = 500\n",
				"\n",
				"# Use the faster device for this intensive computation\n",
				"compute_device = gpu_device if gpu_device else cpu_device\n",
				"\n",
				"fractal_zoom = mandelbrot(\n",
				"    width=1200, height=800, \n",
				"    x_center=x_center, y_center=y_center, \n",
				"    zoom=zoom, max_iter=max_iter_zoom, \n",
				"    dev=compute_device\n",
				")\n",
				"\n",
				"plt.figure(figsize=(12, 8))\n",
				"plt.imshow(fractal_zoom, cmap='magma')\n",
				"plt.title(f'Mandelbrot Set (Zoomed to {zoom}x)')\n",
				"plt.axis('off')\n",
				"plt.show()",
			},
		},
	}
}

func createConclusionCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"## Conclusion: The Foundation for AI"},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"Through these examples, we've seen PyTorch tensors excel at Monte Carlo sampling, grid-based simulations, and iterative fractal generation. All three leverage parallel operations like element-wise math, reductions, and convolutions, which are accelerated dramatically by GPUs."},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{},
			Source:   []string{"This is the fundamental principle that enables modern AI. Neural networks are composed of large tensors of weights, and their core operations—like matrix multiplications during a forward pass and gradient calculations during backpropagation—are precisely the kinds of parallel tasks that GPUs are designed to handle. Without GPU-accelerated tensor libraries like PyTorch, training today's large-scale models would be computationally infeasible."},
		},
	}
}

