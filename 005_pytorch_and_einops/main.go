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
	Attachments    map[string]any `json:"attachments,omitempty"`
}

// Output represents the output of a code cell.
type Output struct {
	OutputType     string         `json:"output_type"`
	Name           string         `json:"name,omitempty"`
	Text           []string       `json:"text,omitempty"`
	Data           map[string]any `json:"data,omitempty"`
	ExecutionCount *int           `json:"execution_count,omitempty"`
}

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
				"version": "3.9.7",
			},
		},
	}

	var allCells []Cell

	allCells = append(allCells, createLectureIntroCells()...)
	allCells = append(allCells, createAutogradCells()...)
	allCells = append(allCells, createTwoPhilosophiesCells()...)
	allCells = append(allCells, createEinopsBasicsTutorialCells()...)
	allCells = append(allCells, createEinopsDLTutorialCells()...)
	allCells = append(allCells, createTransformerIntroCells()...)
	allCells = append(allCells, createTransformerEncoderCells()...)
	allCells = append(allCells, createTransformerDecoderCells()...)
	allCells = append(allCells, createEncoderDecoderCells()...)

	notebook.Cells = allCells

	outputBytes, err := json.MarshalIndent(notebook, "", "  ")
	if err != nil {
		fmt.Printf("Error marshalling JSON: %v\n", err)
		return
	}

	outputFilePath := "005_pytorch_and_einops.ipynb"
	err = os.WriteFile(outputFilePath, outputBytes, 0644)
	if err != nil {
		fmt.Printf("Error writing to file %s: %v\n", outputFilePath, err)
		return
	}

	fmt.Printf("Successfully generated notebook and saved to %s\n", outputFilePath)
}

func createLectureIntroCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Source: []string{
				"# Deep Dive Tutorial: From Autograd to Transformers\n\n",
				"Welcome to the second part of our exploration into PyTorch. In the previous session, we established PyTorch as a powerful library for GPU-accelerated tensor computations. Today, we will build upon that foundation to understand how PyTorch enables the creation and training of complex neural networks.\n\n",
				"We will cover:\n\n",
				"1.  **The Magic Behind PyTorch: Autograd and Computation Graphs:** How PyTorch automatically calculates gradients, the bedrock of modern neural network training.\n",
				"2.  **Two Philosophies of Model Building:** We'll compare the standard Object-Oriented (`nn.Module`) approach with the flexible functional (`torch.func`) paradigm.\n",
				"3.  **Introducing `einops`:** A powerful and elegant library for tensor manipulation that will make your code more readable, reliable, and expressive.\n",
				"4.  **Putting It All Together: Building a Transformer:** We will use our knowledge and `einops` to construct a Transformer, the architecture behind models like ChatGPT and AlphaFold.",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source: []string{
				"!pip install einops graphviz -q",
			},
			Metadata: map[string]any{},
		},
	}
}

func createAutogradCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Source: []string{
				"## 1. The Magic Behind PyTorch: Autograd\n\n",
				"At the heart of any deep learning framework is the ability to perform **automatic differentiation**, or **Autograd**. When we train a neural network, we need to adjust its parameters (weights and biases) to minimize a loss function. This adjustment is typically done using an optimization algorithm like Gradient Descent, which requires computing the gradient of the loss function with respect to every parameter in the model.\n\n",
				"PyTorch automates this entire process with `torch.autograd`. Here’s the core idea:\n\n",
				"*   **Tracking Operations:** PyTorch keeps track of every operation performed on tensors that have their `requires_grad` attribute set to `True`.\n",
				"*   **Building a DAG:** As operations are performed, PyTorch dynamically builds a **Directed Acyclic Graph (DAG)**. In this graph, the leaves are the input tensors (and model parameters), and the root is the output tensor (typically, the loss). The nodes in between represent the mathematical operations.\n",
				"*   **Backpropagation with `.backward()`:** When you call `.backward()` on the final output tensor (e.g., `loss.backward()`), PyTorch traverses this graph backward from the root. It uses the **chain rule** of calculus to compute the gradients at each step and accumulates them in the `.grad` attribute of the leaf tensors (i.e., your model's parameters).\n\n",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "markdown",
			Source:   []string{"### Autograd's Computation Graph (DAG)\n\nHere is a simplified view of the graph created during a forward pass and traversed during the backward pass."},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source: []string{
				"import graphviz\n\n",
				"dot_source = \"\"\"\n",
				"digraph Autograd_DAG {\n",
				"    rankdir=TB;\n",
				"    node [shape=box, style=\"rounded,filled\", fillcolor=\"lightblue\"];\n",
				"    edge [fontsize=10];\n\n",
				"    subgraph cluster_forward {\n",
				"        label = \"Forward Pass: Building the Graph\";\n",
				"        style=filled;\n",
				"        color=lightgrey;\n",
				"        node [fillcolor=\"azure\"];\n",
				"        X [label=\"Input Tensor X\"];\n",
				"        W [label=\"Parameters W\\nrequires_grad=True\", style=\"rounded,filled\", fillcolor=\"lightpink\"];\n",
				"        MatMul [label=\"z = X @ W\"];\n",
				"        LossFn [label=\"Loss = L(z)\"];\n",
				"        X -> MatMul;\n",
				"        W -> MatMul;\n",
				"        MatMul -> LossFn;\n",
				"    }\n\n",
				"    subgraph cluster_backward {\n",
				"        label = \"Backward Pass: Traversing the Graph\";\n",
				"        style=filled;\n",
				"        color=lightgrey;\n",
				"        node [fillcolor=\"honeydew\"];\n",
				"        Grad_Z [label=\"Compute dL/dz\"];\n",
				"        Grad_W [label=\"Compute dL/dW (Chain Rule)\"];\n",
				"        W_Update [label=\"Accumulate in W.grad\", style=\"rounded,filled\", fillcolor=\"lightpink\"];\n",
				"        LossFn -> Grad_Z [label=\"loss.backward()\"];\n",
				"        Grad_Z -> Grad_W;\n",
				"        Grad_W -> W_Update;\n",
				"    }\n",
				"}\n",
				"\"\"\"\n",
				"graph = graphviz.Source(dot_source)\n",
				"graph",
			},
			Metadata: map[string]any{},
		},
	}
}

func createTwoPhilosophiesCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Source: []string{
				"## 2. Two Philosophies of Model Building\n\n",
				"PyTorch offers two primary ways to structure your deep learning models: the traditional object-oriented approach and a more recent functional approach. Let's explore both by building a simple Multi-Layer Perceptron (MLP) to predict housing prices using the California Housing dataset.",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source: []string{
				"import torch\n",
				"import torch.nn as nn\n",
				"import torch.optim as optim\n",
				"import torch.nn.functional as F\n",
				"from torch.func import grad\n\n",
				"# For loading real data\n",
				"from sklearn.datasets import fetch_california_housing\n",
				"from sklearn.model_selection import train_test_split\n",
				"from sklearn.preprocessing import StandardScaler\n\n",
				"# --- 0. Data Loading and Preprocessing ---\n",
				"print(\"--- Loading and preparing data ---\")\n",
				"housing = fetch_california_housing()\n",
				"X, y = housing.data, housing.target\n",
				"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
				"scaler = StandardScaler()\n",
				"X_train = scaler.fit_transform(X_train)\n",
				"X_test = scaler.transform(X_test)\n",
				"X_train = torch.tensor(X_train, dtype=torch.float32)\n",
				"y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)\n",
				"X_test = torch.tensor(X_test, dtype=torch.float32)\n",
				"y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)\n\n",
				"# --- Hyperparameters ---\n",
				"INPUT_FEATURES = X_train.shape[1]\n",
				"HIDDEN_FEATURES = 64\n",
				"OUTPUT_FEATURES = 1\n",
				"LEARNING_RATE = 0.001\n",
				"EPOCHS = 20\n\n",
				"## 1. The nn.Module Approach (Stateful Objects)\n",
				"print(\"\\n--- Training with nn.Module ---\")\n\n",
				"class SimpleMLP(nn.Module):\n",
				"    def __init__(self):\n",
				"        super().__init__()\n",
				"        self.layer1 = nn.Linear(INPUT_FEATURES, HIDDEN_FEATURES)\n",
				"        self.activation = nn.ReLU()\n",
				"        self.layer2 = nn.Linear(HIDDEN_FEATURES, OUTPUT_FEATURES)\n\n",
				"    def forward(self, x):\n",
				"        x = self.layer1(x)\n",
				"        x = self.activation(x)\n",
				"        x = self.layer2(x)\n",
				"        return x\n\n",
				"model = SimpleMLP()\n",
				"loss_fn_mse = nn.MSELoss()\n",
				"optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)\n\n",
				"for epoch in range(EPOCHS):\n",
				"    model.train()\n",
				"    y_pred = model(X_train)\n",
				"    loss = loss_fn_mse(y_pred, y_train)\n",
				"    optimizer.zero_grad()\n",
				"    loss.backward()\n",
				"    optimizer.step()\n",
				"    if (epoch + 1) % 5 == 0:\n",
				"        print(f\"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}\")\n\n",
				"model.eval()\n",
				"with torch.no_grad():\n",
				"    y_test_pred = model(X_test)\n",
				"    test_loss = loss_fn_mse(y_test_pred, y_test)\n",
				"    print(f\"Final Test MSE (nn.Module): {test_loss.item():.4f}\")\n\n",
				"## 2. The torch.func Approach (Stateless Functions)\n",
				"print(\"\\n--- Training with torch.func ---\")\n\n",
				"def init_params():\n",
				"    params = {\n",
				"        'w1': torch.randn(HIDDEN_FEATURES, INPUT_FEATURES) * (2 / INPUT_FEATURES)**0.5,\n",
				"        'b1': torch.zeros(HIDDEN_FEATURES),\n",
				"        'w2': torch.randn(OUTPUT_FEATURES, HIDDEN_FEATURES) * (2 / HIDDEN_FEATURES)**0.5,\n",
				"        'b2': torch.zeros(OUTPUT_FEATURES)\n",
				"    }\n",
				"    return params\n\n",
				"def mlp_fn(params, x):\n",
				"    x = F.linear(x, params['w1'], params['b1'])\n",
				"    x = F.relu(x)\n",
				"    x = F.linear(x, params['w2'], params['b2'])\n",
				"    return x\n\n",
				"def compute_loss(params, x, y):\n",
				"    y_pred = mlp_fn(params, x)\n",
				"    return F.mse_loss(y_pred, y)\n\n",
				"grad_fn = grad(compute_loss)\n",
				"functional_params = init_params()\n\n",
				"for epoch in range(EPOCHS):\n",
				"    grads = grad_fn(functional_params, X_train, y_train)\n",
				"    with torch.no_grad():\n",
				"        for key in functional_params:\n",
				"            functional_params[key] -= LEARNING_RATE * grads[key]\n",
				"    if (epoch + 1) % 5 == 0:\n",
				"        loss = compute_loss(functional_params, X_train, y_train)\n",
				"        print(f\"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}\")\n\n",
				"with torch.no_grad():\n",
				"    y_test_pred = mlp_fn(functional_params, X_test)\n",
				"    test_loss = F.mse_loss(y_test_pred, y_test)\n",
				"    print(f\"Final Test MSE (torch.func): {test_loss.item():.4f}\")",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "markdown",
			Source: []string{
				"### Key Takeaway\n\n",
				"Both methods achieve the same goal. 👍\n\n",
				"*   **`nn.Module`** is the standard, convenient choice for most projects. It bundles state (parameters) and logic (the `forward` method) together in an object.\n\n",
				"*   **`torch.func`** decouples state and logic. This provides greater flexibility and is essential for advanced techniques like meta-learning or custom per-sample gradient computations, where you need to manipulate model parameters in more complex ways.",
			},
			Metadata: map[string]any{},
		},
	}
}

func createEinopsBasicsTutorialCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Source: []string{
				"# Einops tutorial, part 1: basics\n",
				"## Welcome to einops-land!\n\n",
				"We don't write \n",
				"```python\n",
				"y = x.transpose(0, 2, 3, 1)\n",
				"```\n",
				"We write comprehensible code\n",
				"```python\n",
				"y = rearrange(x, 'b c h w -> b h w c')\n",
				"```\n\n\n",
				"`einops` supports widely used tensor packages (such as `numpy`, `pytorch`, `jax`, `tensorflow`), and extends them.\n\n",
				"## What's in this tutorial?\n\n",
				"- fundamentals: reordering, composition and decomposition of axes\n",
				"- operations: `rearrange`, `reduce`, `repeat`\n",
				"- how much you can do with a single operation!",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "markdown",
			Source:   []string{"## Preparations\n\nTo run this notebook, you will need to download the resources from the einops repository."},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source: []string{
				"import numpy as np\n",
				"from IPython import get_ipython\n",
				"from IPython.display import display_html\n",
				"from PIL.Image import fromarray\n",
				"from einops import rearrange, reduce, repeat\n\n",
				"# The following code is inlined from the original tutorial's utils.py\n",
				"def display_np_arrays_as_images():\n",
				"    def np_to_png(a):\n",
				"        if 2 <= len(a.shape) <= 3:\n",
				"            return fromarray(np.array(np.clip(a, 0, 1) * 255, dtype=\"uint8\"))._repr_png_()\n",
				"        else:\n",
				"            return fromarray(np.zeros([1, 1], dtype=\"uint8\"))._repr_png_()\n\n",
				"    def np_to_text(obj, p, cycle):\n",
				"        if len(obj.shape) < 2:\n",
				"            print(repr(obj))\n",
				"        if 2 <= len(obj.shape) <= 3:\n",
				"            pass\n",
				"        else:\n",
				"            print(f\"<array of shape {obj.shape}>\")\n\n",
				"    # This will only work in IPython environments\n",
				"    try:\n",
				"        ipy = get_ipython()\n",
				"        if ipy is None:\n",
				"            return\n",
				"        ipy.display_formatter.formatters[\"image/png\"].for_type(np.ndarray, np_to_png)\n",
				"        ipy.display_formatter.formatters[\"text/plain\"].for_type(np.ndarray, np_to_text)\n",
				"    except ImportError: pass\n\n",
				"_style_inline = \"\"\"<style>\n",
				".einops-answer {\n",
				"    color: transparent;\n",
				"    padding: 5px 15px;\n",
				"    background-color: #def;\n",
				"}\n",
				".einops-answer:hover { color: blue; }\n",
				"</style>\"\"\"\n\n",
				"def guess(x):\n",
				"    display_html(\n",
				"        _style_inline + f\"<h4>Answer is: <span class='einops-answer'>{tuple(x)}</span> (hover to see)</h4>\",\n",
				"        raw=True,\n",
				"    )\n\n",
				"# Now run the functions\n",
				"display_np_arrays_as_images()\n\n",
				"# Download resources\n",
				"!mkdir -p resources\n",
				"!wget https://raw.githubusercontent.com/arogozhnikov/einops/main/docs/resources/test_images.npy -P resources/",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "markdown",
			Source:   []string{"## Load a batch of images to play with"},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source: []string{
				"ims = np.load(\"./resources/test_images.npy\", allow_pickle=False)\n",
				"print(ims.shape, ims.dtype)",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source:   []string{"# display the first image\nims[0]"},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source:   []string{"# rearrange, as the name suggests, rearranges elements\nrearrange(ims[0], \"h w c -> w h c\")"},
			Metadata: map[string]any{},
		},
		{
			CellType: "markdown",
			Source:   []string{"## Composition of axes"},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source:   []string{"# einops allows seamlessly composing batch and height to a new height dimension\nrearrange(ims, \"b h w c -> (b h) w c\")"},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source:   []string{"# resulting dimensions are computed very simply\nrearrange(ims, \"b h w c -> h (b w) c\").shape"},
			Metadata: map[string]any{},
		},
		{
			CellType: "markdown",
			Source:   []string{"## Decomposition of axis"},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source:   []string{"# decomposition is the inverse process - represent an axis as a combination of new axes\nrearrange(ims, \"(b1 b2) h w c -> b1 b2 h w c \", b1=2).shape"},
			Metadata: map[string]any{},
		},
		{
			CellType: "markdown",
			Source:   []string{"## Meet einops.reduce"},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source:   []string{"# average over batch\nreduce(ims, \"b h w c -> h w c\", \"mean\")"},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source:   []string{"# max-pooling with a kernel 2x2\nreduce(ims, \"b (h h1) (w w1) c -> b h w c\", \"max\", h1=2, w1=2)"},
			Metadata: map[string]any{},
		},
		{
			CellType: "markdown",
			Source:   []string{"## Repeating elements"},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source:   []string{"# repeat along a new axis. New axis can be placed anywhere\nrepeat(ims[0], \"h w c -> h new_axis w c\", new_axis=5).shape"},
			Metadata: map[string]any{},
		},
	}
}

func createEinopsDLTutorialCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Source: []string{
				"# Einops tutorial, part 2: deep learning\n\n",
				"Previous part of tutorial provides visual examples with numpy.\n\n",
				"## What's in this tutorial?\n\n",
				"- working with deep learning packages\n",
				"- important cases for deep learning models\n",
				"- `einops.asnumpy` and `einops.layers`",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source: []string{
				"import torch\n",
				"import numpy as np\n",
				"from einops import rearrange, reduce\n",
				"x_np = np.random.RandomState(42).normal(size=[10, 32, 100, 200])\n",
				"x = torch.from_numpy(x_np)\n",
				"x.requires_grad = True",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "markdown",
			Source: []string{
				"## Backpropagation\n\n",
				"- Gradients are a corner stone of deep learning\n",
				"- You can back-propagate through einops operations",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source: []string{
				"y0 = x\n",
				"y1 = reduce(y0, \"b c h w -> b c\", \"max\")\n",
				"y2 = rearrange(y1, \"b c -> c b\")\n",
				"y3 = reduce(y2, \"c b -> \", \"sum\")\n",
				"y3.backward()\n",
				"print(reduce(x.grad, \"b c h w -> \", \"sum\"))",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "markdown",
			Source: []string{
				"## Common building blocks of deep learning\n\n",
				"Let's check how some familiar operations can be written with `einops`\n\n",
				"**Flattening** is common operation, frequently appears at the boundary\n",
				"between convolutional layers and fully connected layers",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source:   []string{"y = rearrange(x, \"b c h w -> b (c h w)\")\ny.shape"},
			Metadata: map[string]any{},
		},
		{
			CellType: "markdown",
			Source:   []string{"**space-to-depth**"},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source:   []string{"y = rearrange(x, \"b c (h h1) (w w1) -> b (h1 w1 c) h w\", h1=2, w1=2)\ny.shape"},
			Metadata: map[string]any{},
		},
		{
			CellType: "markdown",
			Source:   []string{"**depth-to-space** (notice that it's reverse of the previous)"},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source:   []string{"y = rearrange(x, \"b (h1 w1 c) h w -> b c (h h1) (w w1)\", h1=2, w1=2)\ny.shape"},
			Metadata: map[string]any{},
		},
		{
			CellType: "markdown",
			Source:   []string{"## Layers\n\nFor frameworks that prefer operating with layers, `einops` layers are available.\n"},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source: []string{
				"from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, ReLU\n",
				"from einops.layers.torch import Rearrange\n\n",
				"# A simple LeNet-style model for image classification\n",
				"model_with_einops = Sequential(\n",
				"    # Assuming input is (batch, 3, 32, 32)\n",
				"    Conv2d(3, 6, kernel_size=5), # -> (batch, 6, 28, 28)\n",
				"    MaxPool2d(kernel_size=2),   # -> (batch, 6, 14, 14)\n",
				"    Conv2d(6, 16, kernel_size=5),# -> (batch, 16, 10, 10)\n",
				"    MaxPool2d(kernel_size=2),   # -> (batch, 16, 5, 5)\n",
				"    # Flatten the feature map\n",
				"    Rearrange('b c h w -> b (c h w)'), \n",
				"    Linear(16*5*5, 120), \n",
				"    ReLU(),\n",
				"    Linear(120, 10), \n",
				")",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source: []string{
				"# Let's check that the model works as expected\n",
				"dummy_input = torch.randn(1, 3, 32, 32)\n",
				"output = model_with_einops(dummy_input)\n",
				"print(model_with_einops)\n",
				"print(\"\\nOutput shape:\", output.shape)",
			},
			Metadata: map[string]any{},
		},
	}
}

func createTransformerIntroCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Source: []string{
				"## 3. Putting It All Together: Building a Transformer\n\n",
				"The Transformer architecture has become the foundation of modern AI. Its core component is the **Multi-Head Self-Attention (MHSA)** mechanism. Before writing the code, let's build a strong intuition for how `einops` and `einsum` make implementing attention remarkably elegant.\n\n",
				"### Bridging Theory and Code: Attention with `einops` and `einsum`\n\n",
				"At its heart, attention is a mechanism to compute a weighted sum of values, where the weights are dynamically computed based on the similarity between a \"query\" and all \"keys\".\n\n",
				"**The Steps of Attention:**\n\n",
				"1.  **Similarity Score:** We compute the dot product between each Query (Q) and all Keys (K). This is a perfect use case for `torch.einsum`, which expresses this complex batch matrix multiplication concisely:\n    *   `torch.einsum('b h i d, b h j d -> b h i j', q, k)`\n    *   **Translation:** For each item in the **b**atch and each **h**ead, multiply the query token `i` (of dimension **d**) with each key token `j` (of dimension **d**) to produce a similarity matrix of shape (`b`, `h`, `i`, `j`).\n\n",
				"2.  **Scaling:** We scale the scores by dividing by the square root of the head dimension (`sqrt(d_k)`). This stabilizes the gradients during training.\n\n",
				"3.  **Softmax:** The scores are converted into probabilities (weights that sum to 1).\n\n",
				"4.  **Weighted Sum:** We multiply the attention scores with the Values (V). `einsum` again makes this clear:\n    *   `torch.einsum('b h i j, b h j d -> b h i d', attn, v)`\n    *   **Translation:** For each item in the **b**atch and each **h**ead, multiply the attention scores (`i`, `j`) with each value token `j` (of dimension **d**) to produce a new weighted representation for token `i`.\n\n",
				"**The Role of `rearrange` in Multi-Head Attention:**\n\n",
				"Multi-head attention requires splitting a single large Q, K, or V tensor into multiple smaller \"heads\". `rearrange` handles this decomposition and the reverse composition elegantly:\n\n",
				"*   **Splitting:** `rearrange(qkv, 'b n (h d) -> b h n d', h=num_heads)`\n    *   This takes a tensor of shape `(batch, sequence_len, heads * head_dim)` and splits the last dimension into two new ones: `h` and `d`.\n*   **Combining:** `rearrange(out, 'b h n d -> b n (h d)')`\n    *   This performs the inverse operation, merging the `h` and `d` dimensions back together.\n\n",
				"This combination of `einsum` for the core logic and `rearrange` for structuring the data is what makes `einops` so powerful for building Transformers.",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "markdown",
			Source:   []string{"### Scaled Dot-Product Attention"},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source: []string{
				"dot_source = '''\n",
				"digraph G {\n",
				"    rankdir=TB;\n",
				"    node [shape=box, style=filled, fillcolor=\"lightblue\"];\n",
				"    Q [label=\"Query\"]; K [label=\"Key\"]; V [label=\"Value\"];\n",
				"    MatMul1 [label=\"MatMul\"];\n",
				"    TransposeK [label=\"Transpose\"];\n",
				"    Scale [label=\"Scale by 1/√d_k\"];\n",
				"    OptionalMask [label=\"Optional Mask\", style=dashed];\n",
				"    Softmax [fillcolor=\"lightyellow\"];\n",
				"    MatMul2 [label=\"MatMul\"];\n",
				"    Output [shape=ellipse, fillcolor=\"lightgreen\"];\n",
				"    Q -> MatMul1; K -> TransposeK -> MatMul1;\n",
				"    MatMul1 -> Scale -> OptionalMask -> Softmax -> MatMul2;\n",
				"    V -> MatMul2 -> Output;\n",
				"    labelloc=\"t\"; label=\"Scaled Dot-Product Attention\";\n",
				"}\n",
				"'''\n",
				"graphviz.Source(dot_source)",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "markdown",
			Source:   []string{"### Multi-Head Attention"},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source: []string{
				"dot_source = '''\n",
				"digraph G {\n",
				"    rankdir=TB;\n",
				"    node [shape=box, style=rounded];\n",
				"    subgraph cluster_input {\n",
				"        label = \"Input Projection\"; style=filled; color=lightgrey;\n",
				"        Input [label=\"Input x\"];\n",
				"        LinearQ [label=\"Linear_Q\"]; LinearK [label=\"Linear_K\"]; LinearV [label=\"Linear_V\"];\n",
				"        Input -> LinearQ -> Q;\n",
				"        Input -> LinearK -> K;\n",
				"        Input -> LinearV -> V;\n",
				"    }\n",
				"    subgraph cluster_heads {\n",
				"        label = \"Parallel Attention Heads\"; style=filled; color=lightgrey;\n",
				"        node [style=filled, fillcolor=\"lightblue\"];\n",
				"        Head1 [label=\"Head 1\\nScaled Dot-Product\"];\n",
				"        Head2 [label=\"Head 2\\nScaled Dot-Product\"];\n",
				"        HeadN [label=\"...\\nHead n\"];\n",
				"    }\n",
				"    Q -> Head1; K -> Head1; V -> Head1;\n",
				"    Q -> Head2; K -> Head2; V -> Head2;\n",
				"    Q -> HeadN; K -> HeadN; V -> HeadN;\n",
				"    subgraph cluster_output {\n",
				"        label = \"Output Stage\"; style=filled; color=lightgrey;\n",
				"        Concat [label=\"Concatenate\"];\n",
				"        LinearOut [label=\"Final Linear Layer\"];\n",
				"        OutputMHA [label=\"Multi-Head Output\", shape=ellipse, style=filled, fillcolor=\"lightgreen\"];\n",
				"    }\n",
				"    Head1 -> Concat; Head2 -> Concat; HeadN -> Concat;\n",
				"    Concat -> LinearOut -> OutputMHA;\n",
				"}\n",
				"'''\n",
				"graphviz.Source(dot_source)",
			},
			Metadata: map[string]any{},
		},
	}
}

func createTransformerEncoderCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Source: []string{
				"### Architecture 1: Transformer Encoder\n\n",
				"The Encoder's job is to map an input sequence of symbol representations (x₁, ..., xₙ) to a sequence of continuous representations z = (z₁, ..., zₙ). It is composed of a stack of identical layers, each having two sub-layers: a multi-head self-attention mechanism and a simple, position-wise fully connected feed-forward network. Residual connections and layer normalization are used around each of the two sub-layers.",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source: []string{
				"dot_source = '''\n",
				"digraph G {\n",
				"    rankdir=TB;\n",
				"    node [shape=box, style=rounded];\n",
				"    X_in [label=\"Input x\"];\n",
				"    AddNorm1 [label=\"Add & Norm\", shape=circle];\n",
				"    AddNorm2 [label=\"Add & Norm\", shape=circle];\n",
				"    X_out [label=\"Output\"];\n",
				"    MHA [label=\"Multi-Head\\nAttention\", style=filled, fillcolor=\"lightpink\"];\n",
				"    FF [label=\"Feed Forward\", style=filled, fillcolor=\"skyblue\"];\n",
				"    LN1 [label=\"LayerNorm\"]; LN2 [label=\"LayerNorm\"];\n",
				"    X_in -> LN1 -> MHA -> AddNorm1;\n",
				"    X_in -> AddNorm1 [label=\" Residual\"];\n",
				"    AddNorm1 -> LN2 -> FF -> AddNorm2;\n",
				"    AddNorm1 -> AddNorm2 [label=\" Residual\"];\n",
				"    AddNorm2 -> X_out;\n",
				"    labelloc=\"t\";\n",
				"    label=\"Transformer Encoder Block\";\n",
				"}\n",
				"'''\n",
				"graphviz.Source(dot_source)",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source: []string{
				"import torch\n",
				"from torch import nn\n",
				"from einops import rearrange\n\n",
				"# 1. Multi-Head Self-Attention (MHSA)\n",
				"class Attention(nn.Module):\n",
				"    def __init__(self, dim, n_heads, head_dim):\n",
				"        super().__init__()\n",
				"        self.n_heads = n_heads\n",
				"        inner_dim = n_heads * head_dim\n",
				"        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)\n",
				"        self.scale = head_dim ** -0.5\n\n",
				"    def forward(self, x):\n",
				"        # x: (batch, sequence, dimension)\n",
				"        qkv = self.to_qkv(x).chunk(3, dim=-1)\n",
				"        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv)\n\n",
				"        # Scaled Dot-Product Attention using einsum\n",
				"        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale\n",
				"        attn = dots.softmax(dim=-1)\n",
				"        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)\n\n",
				"        # Reshape and return\n",
				"        out = rearrange(out, 'b h n d -> b n (h d)')\n",
				"        return out\n\n",
				"# 2. Feed-Forward Network\n",
				"class FeedForward(nn.Module):\n",
				"    def __init__(self, dim, hidden_dim):\n",
				"        super().__init__()\n",
				"        self.net = nn.Sequential(\n",
				"            nn.Linear(dim, hidden_dim),\n",
				"            nn.GELU(),\n",
				"            nn.Linear(hidden_dim, dim)\n",
				"        )\n",
				"    def forward(self, x):\n",
				"        return self.net(x)\n\n",
				"# 3. Transformer Encoder Block\n",
				"class Transformer(nn.Module):\n",
				"    def __init__(self, dim, n_heads, head_dim, mlp_dim):\n",
				"        super().__init__()\n",
				"        self.norm1 = nn.LayerNorm(dim)\n",
				"        self.attn = Attention(dim, n_heads, head_dim)\n",
				"        self.norm2 = nn.LayerNorm(dim)\n",
				"        self.ff = FeedForward(dim, mlp_dim)\n\n",
				"    def forward(self, x):\n",
				"        # Attention block with pre-normalization and residual connection\n",
				"        x = self.attn(self.norm1(x)) + x\n",
				"        # Feed-forward block with pre-normalization and residual connection\n",
				"        x = self.ff(self.norm2(x)) + x\n",
				"        return x\n\n",
				"# --- Example Usage ---\n",
				"batch_size = 1\n",
				"sequence_length = 10\n",
				"embedding_dim = 64\n",
				"num_heads = 8\n",
				"head_dimension = 8\n",
				"mlp_hidden_dim = 128\n\n",
				"input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)\n\n",
				"transformer_encoder = Transformer(\n",
				"    dim=embedding_dim,\n",
				"    n_heads=num_heads,\n",
				"    head_dim=head_dimension,\n",
				"    mlp_dim=mlp_hidden_dim\n",
				")\n\n",
				"output = transformer_encoder(input_tensor)\n",
				"print(\"Input Shape:\", input_tensor.shape)\n",
				"print(\"Output Shape:\", output.shape)",
			},
			Metadata: map[string]any{},
		},
	}
}

func createTransformerDecoderCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Source: []string{
				"### Architecture 2: Transformer Decoder\n\n",
				"The Decoder is also composed of a stack of identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head cross-attention over the output of the encoder stack. The self-attention sub-layer in the decoder stack is also modified to prevent positions from attending to subsequent positions (Masked Attention).",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source: []string{
				"dot_source = '''\n",
				"digraph G {\n",
				"    rankdir=TB;\n",
				"    node [shape=box, style=rounded];\n",
				"    Tgt_in [label=\"Target Input\"];\n",
				"    Context [label=\"Encoder Context\"];\n",
				"    AddNorm1 [label=\"Add & Norm\", shape=circle];\n",
				"    AddNorm2 [label=\"Add & Norm\", shape=circle];\n",
				"    AddNorm3 [label=\"Add & Norm\", shape=circle];\n",
				"    Tgt_out [label=\"Output\"];\n",
				"    MMHA [label=\"Masked Multi-Head\\nSelf-Attention\", style=filled, fillcolor=\"lightpink\"];\n",
				"    MHA [label=\"Multi-Head\\nCross-Attention\", style=filled, fillcolor=\"lightblue\"];\n",
				"    FF [label=\"Feed Forward\", style=filled, fillcolor=\"skyblue\"];\n",
				"    LN1 [label=\"LayerNorm\"]; LN2 [label=\"LayerNorm\"]; LN3 [label=\"LayerNorm\"];\n",
				"    Tgt_in -> LN1 -> MMHA -> AddNorm1;\n",
				"    Tgt_in -> AddNorm1 [label=\" Residual\"];\n",
				"    AddNorm1 -> LN2 -> MHA -> AddNorm2;\n",
				"    AddNorm1 -> AddNorm2 [label=\" Residual\"];\n",
				"    Context -> MHA;\n",
				"    AddNorm2 -> LN3 -> FF -> AddNorm3;\n",
				"    AddNorm2 -> AddNorm3 [label=\" Residual\"];\n",
				"    AddNorm3 -> Tgt_out;\n",
				"    labelloc=\"t\";\n",
				"    label=\"Transformer Decoder Block\";\n",
				"}\n",
				"'''\n",
				"graphviz.Source(dot_source)",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source: []string{
				"class MaskedAttention(nn.Module):\n",
				"    def __init__(self, dim, n_heads, head_dim):\n",
				"        super().__init__()\n",
				"        self.n_heads = n_heads\n",
				"        inner_dim = n_heads * head_dim\n",
				"        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)\n",
				"        self.scale = head_dim ** -0.5\n\n",
				"    def forward(self, x):\n",
				"        qkv = self.to_qkv(x).chunk(3, dim=-1)\n",
				"        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv)\n\n",
				"        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale\n\n",
				"        # Masking logic\n",
				"        mask = torch.ones_like(dots, dtype=torch.bool).triu_(1)\n",
				"        dots.masked_fill_(mask, float('-inf'))\n\n",
				"        attn = dots.softmax(dim=-1)\n",
				"        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)\n",
				"        out = rearrange(out, 'b h n d -> b n (h d)')\n",
				"        return out\n\n",
				"class CrossAttention(nn.Module):\n",
				"    def __init__(self, dim, n_heads, head_dim):\n",
				"        super().__init__()\n",
				"        self.n_heads = n_heads\n",
				"        inner_dim = n_heads * head_dim\n",
				"        self.to_q = nn.Linear(dim, inner_dim, bias=False)\n",
				"        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)\n",
				"        self.scale = head_dim ** -0.5\n\n",
				"    def forward(self, x, context):\n",
				"        # Q from decoder, K and V from encoder context\n",
				"        q = self.to_q(x)\n",
				"        k, v = self.to_kv(context).chunk(2, dim=-1)\n",
				"        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), (q, k, v))\n\n",
				"        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale\n",
				"        attn = dots.softmax(dim=-1)\n",
				"        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)\n",
				"        out = rearrange(out, 'b h n d -> b n (h d)')\n",
				"        return out\n\n",
				"# Decoder-Only block (e.g., for a GPT-like model without cross-attention)\n",
				"class DecoderOnlyBlock(nn.Module):\n",
				"    def __init__(self, dim, n_heads, head_dim, mlp_dim):\n",
				"        super().__init__()\n",
				"        self.norm1 = nn.LayerNorm(dim)\n",
				"        self.attn = MaskedAttention(dim, n_heads, head_dim)\n",
				"        self.norm2 = nn.LayerNorm(dim)\n",
				"        self.ff = FeedForward(dim, mlp_dim)\n\n",
				"    def forward(self, x):\n",
				"        x = self.attn(self.norm1(x)) + x\n",
				"        x = self.ff(self.norm2(x)) + x\n",
				"        return x\n",
			},
			Metadata: map[string]any{},
		},
	}
}

func createEncoderDecoderCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Source: []string{
				"### Architecture 3: Encoder-Decoder Transformer\n\n",
				"This is the full architecture, combining the Encoder and Decoder stacks. The output of the top encoder is transformed into a set of attention vectors K and V. These are used in each decoder's cross-attention layer, allowing the decoder to focus on appropriate places in the input sequence.",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source: []string{
				"dot_source = '''\n",
				"digraph G {\n",
				"    rankdir=TB;\n",
				"    graph [compound=true];\n",
				"    node [shape=box, style=rounded];\n",
				"    subgraph cluster_encoder {\n",
				"        label = \"Encoder Stack\";\n",
				"        rankdir=LR;\n",
				"        InputSeq [label=\"Input Sequence\"];\n",
				"        EmbIn [label=\"Embedding\"];\n",
				"        Enc1 [label=\"Encoder Block 1\", style=filled, fillcolor=\"lightpink\"];\n",
				"        EncN [label=\"Encoder Block N\", style=filled, fillcolor=\"lightpink\"];\n",
				"        ContextOut [label=\"Context (K, V)\"];\n",
				"        InputSeq -> EmbIn -> Enc1 -> EncN -> ContextOut;\n",
				"    }\n",
				"    subgraph cluster_decoder {\n",
				"        label = \"Decoder Stack\";\n",
				"        rankdir=LR;\n",
				"        OutputSeq [label=\"Output Sequence\"];\n",
				"        EmbOut [label=\"Embedding\"];\n",
				"        Dec1 [label=\"Decoder Block 1\", style=filled, fillcolor=\"lightblue\"];\n",
				"        DecN [label=\"Decoder Block N\", style=filled, fillcolor=\"lightblue\"];\n",
				"        OutputProbs [label=\"Linear + Softmax\"];\n",
				"        OutputSeq -> EmbOut -> Dec1 -> DecN -> OutputProbs;\n",
				"    }\n",
				"    ContextOut -> Dec1 [lhead=cluster_decoder, ltail=cluster_encoder];\n",
				"}\n",
				"'''\n",
				"graphviz.Source(dot_source)",
			},
			Metadata: map[string]any{},
		},
		{
			CellType: "code",
			Source: []string{
				"# Full Decoder Block for an Encoder-Decoder model\n",
				"class TransformerDecoderBlock(nn.Module):\n",
				"    def __init__(self, dim, n_heads, head_dim, mlp_dim):\n",
				"        super().__init__()\n",
				"        self.norm1 = nn.LayerNorm(dim)\n",
				"        self.masked_attn = MaskedAttention(dim, n_heads, head_dim)\n",
				"        self.norm2 = nn.LayerNorm(dim)\n",
				"        self.cross_attn = CrossAttention(dim, n_heads, head_dim)\n",
				"        self.norm3 = nn.LayerNorm(dim)\n",
				"        self.ff = FeedForward(dim, mlp_dim)\n\n",
				"    def forward(self, x, context):\n",
				"        # Masked self-attention\n",
				"        x = self.masked_attn(self.norm1(x)) + x\n",
				"        # Cross-attention with encoder context\n",
				"        x = self.cross_attn(self.norm2(x), context) + x\n",
				"        # Feed-forward\n",
				"        x = self.ff(self.norm3(x)) + x\n",
				"        return x\n\n",
				"# Full Encoder-Decoder Model\n",
				"class EncoderDecoder(nn.Module):\n",
				"    def __init__(self, num_encoder_layers, num_decoder_layers, dim, n_heads, head_dim, mlp_dim):\n",
				"        super().__init__()\n",
				"        self.encoder = nn.ModuleList([Transformer(dim, n_heads, head_dim, mlp_dim) for _ in range(num_encoder_layers)])\n",
				"        self.decoder = nn.ModuleList([TransformerDecoderBlock(dim, n_heads, head_dim, mlp_dim) for _ in range(num_decoder_layers)])\n\n",
				"    def forward(self, src, tgt):\n",
				"        # Encode the source sequence\n",
				"        encoded_src = src\n",
				"        for encoder_layer in self.encoder:\n",
				"            encoded_src = encoder_layer(encoded_src)\n\n",
				"        # Decode using the encoded source as context\n",
				"        decoded_tgt = tgt\n",
				"        for decoder_layer in self.decoder:\n",
				"            decoded_tgt = decoder_layer(decoded_tgt, encoded_src)\n",
				"        return decoded_tgt\n\n",
				"# --- Example Usage ---\n",
				"src_seq_len = 15\n",
				"tgt_seq_len = 20\n\n",
				"source_seq = torch.randn(batch_size, src_seq_len, embedding_dim)\n",
				"target_seq = torch.randn(batch_size, tgt_seq_len, embedding_dim)\n\n",
				"encoder_decoder = EncoderDecoder(\n",
				"    num_encoder_layers=3, num_decoder_layers=3,\n",
				"    dim=embedding_dim, n_heads=num_heads,\n",
				"    head_dim=head_dimension, mlp_dim=mlp_hidden_dim\n",
				")\n\n",
				"output = encoder_decoder(source_seq, target_seq)\n",
				"print(\"Source Shape:\", source_seq.shape)\n",
				"print(\"Target Shape:\", target_seq.shape)\n",
				"print(\"Final Output Shape:\", output.shape)",
			},
			Metadata: map[string]any{},
		},
	}
}

