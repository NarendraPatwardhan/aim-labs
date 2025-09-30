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
				"version": "3.10",
			},
		},
	}

	var allCells []Cell

	// Generate and append cells for each section in a logical order
	allCells = append(allCells, createIntroCells()...)
	allCells = append(allCells, createSetupCells()...)
	allCells = append(allCells, createPipelineIntroCells()...)
	allCells = append(allCells, createSentimentAnalysisCells()...)
	allCells = append(allCells, createTextGenerationCells()...)
	allCells = append(allCells, createZeroShotClassificationCells()...)
	allCells = append(allCells, createNamedEntityRecognitionCells()...)
	allCells = append(allCells, createSummarizationCells()...)
	allCells = append(allCells, createPipelineDeepDiveCells()...)
	allCells = append(allCells, createUnderTheHoodIntroCells()...)
	allCells = append(allCells, createTokenizerCells()...)
	allCells = append(allCells, createModelCells()...)
	allCells = append(allCells, createDatasetsLibraryCells()...)
	allCells = append(allCells, createFinetuningIntroCells()...)
	allCells = append(allCells, createFinetuningDataPrepCells()...)
	allCells = append(allCells, createFinetuningTrainingCells()...)
	allCells = append(allCells, createGradioCells()...)
	allCells = append(allCells, createAdvancedUseCasesIntroCells()...)
	allCells = append(allCells, createVisionCells()...)
	allCells = append(allCells, createAudioCells()...)
	allCells = append(allCells, createHubCells()...)
	allCells = append(allCells, createConclusionCells()...)

	notebook.Cells = allCells

	outputBytes, err := json.MarshalIndent(notebook, "", "  ")
	if err != nil {
		fmt.Printf("Error marshalling JSON: %v\n", err)
		return
	}

	outputFilePath := "006_huggingface_deep_dive.ipynb"
	err = os.WriteFile(outputFilePath, outputBytes, 0644)
	if err != nil {
		fmt.Printf("Error writing to file %s: %v\n", outputFilePath, err)
		return
	}

	fmt.Printf("Successfully generated final notebook and saved to %s\n", outputFilePath)
}

// --- MODIFIED HELPER FUNCTION ---

func createFinetuningTrainingCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "finetuning-training-md"},
			Source: []string{
				"### 4.2. Training with the `Trainer` API\n\n",
				"Now we're ready to train. Below, we'll define our `TrainingArguments` and instantiate the `Trainer`.\n\n",
				"**A Note on Logging and Arguments:**\n",
				"1.  **Disabling `wandb`**: You might be prompted for a \"wandb API key\". This is because the `Trainer` automatically integrates with the Weights & Biases logging tool if it's installed. We will add the argument **`report_to=\"none\"`** to disable this behavior.\n\n",
				"2.  **Compatibility**: We are using a simplified set of arguments for broader compatibility with different versions of the `transformers` library. Advanced features like evaluating and saving the model at the end of each epoch are available in newer versions via arguments like `evaluation_strategy`, `save_strategy`, and `load_best_model_at_end`.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "finetuning-training-code"},
			Source: []string{
				"from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer\n",
				"import numpy as np\n",
				"import evaluate\n\n",
				"# The warning about some weights not being initialized is NORMAL.\n",
				"# It's telling us that a new classification head is being added to the base model, which is exactly what we want to train.\n",
				"model = AutoModelForSequenceClassification.from_pretrained(\"distilbert-base-uncased\", num_labels=2)\n\n",
				"# Define training arguments\n",
				"training_args = TrainingArguments(\n",
				"    output_dir=\"./results\",\n",
				"    learning_rate=2e-5,\n",
				"    per_device_train_batch_size=16,\n",
				"    per_device_eval_batch_size=16,\n",
				"    num_train_epochs=3,\n",
				"    weight_decay=0.01,\n",
				"    report_to=\"none\", # Add this line to disable wandb logging\n",
				")\n\n",
				"# Define metrics computation\n",
				"metric = evaluate.load(\"accuracy\")\n\n",
				"def compute_metrics(eval_pred):\n",
				"    logits, labels = eval_pred\n",
				"    predictions = np.argmax(logits, axis=-1)\n",
				"    return metric.compute(predictions=predictions, references=labels)\n\n",
				"# Instantiate the Trainer\n",
				"trainer = Trainer(\n",
				"    model=model,\n",
				"    args=training_args,\n",
				"    train_dataset=tokenized_train,\n",
				"    eval_dataset=tokenized_test,\n",
				"    tokenizer=tokenizer,\n",
				"    compute_metrics=compute_metrics,\n",
				")\n\n",
				"# Start training!\n",
				"trainer.train()",
			},
		},
	}
}

// --- ALL OTHER HELPER FUNCTIONS (UNCHANGED FROM PREVIOUS RESPONSE) ---

// (The remaining functions `createIntroCells`, `createSetupCells`, etc., are unchanged.
// They are included here for completeness.)

func createIntroCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "intro-title"},
			Source: []string{
				"# A Deep Dive into the Hugging Face Ecosystem 🤖\n\n",
				"**Welcome, Master's students!** This notebook is your comprehensive guide to the Hugging Face ecosystem, the driving force behind the modern NLP revolution. We'll explore the key libraries and concepts that make it possible to leverage state-of-the-art transformer models with just a few lines of code.\n\n",
				"### What is Hugging Face?\n",
				"Hugging Face is a company and an open-source community dedicated to democratizing Artificial Intelligence. They provide tools that empower anyone to build, train, and deploy state-of-the-art models. Their ecosystem is built around a few core components:\n\n",
				"- **🤗 Transformers**: The flagship library providing thousands of pretrained models for a wide range of tasks in text, vision, and audio.\n",
				"- **🤗 Datasets**: A library for easily accessing and processing massive datasets with smart caching and memory-mapping.\n",
				"- **🤗 Tokenizers**: A high-performance library for the text preprocessing steps required by transformer models.\n",
				"- **The Hugging Face Hub**: A central platform (like GitHub for ML) for sharing models, datasets, and demos (Spaces).\n\n",
				"In this notebook, we will journey from the simplest, high-level abstractions down to the nitty-gritty of fine-tuning a model for a custom task. Let's begin!",
			},
		},
	}
}

func createSetupCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "setup-intro-md"},
			Source: []string{
				"## 1. Environment Setup\n\n",
				"First things first, let's install the necessary libraries and verify that our environment is correctly configured. We will use the `--upgrade` flag to ensure we have the latest versions, which is important for API consistency.\n\n",
				"**CRITICAL NOTE:** After running the installation, you **MUST** restart the runtime for the changes to take effect. In Google Colab, go to `Runtime > Restart runtime` in the menu.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "setup-install-code"},
			Source: []string{
				"# We use -q for a quiet installation and --upgrade to get the latest versions\n",
				"!pip install -q --upgrade transformers datasets evaluate accelerate torch",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "setup-gpu-check-md"},
			Source: []string{
				"### GPU Check\n",
				"Let's confirm that PyTorch can detect and use the T4 GPU provided by Colab. This is crucial for the training steps later on, as fine-tuning transformers on a CPU is incredibly slow.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "setup-gpu-check-code"},
			Source: []string{
				"import torch\n\n",
				"if torch.cuda.is_available():\n",
				"    device = torch.device(\"cuda\")\n",
				"    print(f\"GPU is available: {torch.cuda.get_device_name(0)}\")\n",
				"else:\n",
				"    device = torch.device(\"cpu\")\n",
				"    print(\"GPU not available, using CPU. Training will be very slow.\")",
			},
		},
	}
}

func createPipelineIntroCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "pipeline-intro-md"},
			Source: []string{
				"## 2. The `pipeline`: Your Gateway to Transformers\n\n",
				"The easiest way to start using a pretrained model is with the `pipeline()` function. It's a high-level API that encapsulates all the complex steps: preprocessing text, feeding it to the model, and postprocessing the output into a human-readable format.\n\n",
				"You simply instantiate a pipeline by specifying a task, and the library handles the rest, including downloading a suitable pretrained model from the Hub.",
			},
		},
	}
}

func createSentimentAnalysisCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "sentiment-md"},
			Source: []string{
				"### 2.1. Task: Sentiment Analysis\n",
				"Let's start with a classic: determining if a piece of text is positive or negative.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "sentiment-code"},
			Source: []string{
				"from transformers import pipeline\n\n",
				"# The first time you run this, it will download the default model for the task\n",
				"classifier = pipeline(\"sentiment-analysis\")\n\n",
				"results = classifier([\n",
				"    \"This course is incredibly informative and well-structured!\",\n",
				"    \"I'm not sure I understand everything, the pace is a bit too fast.\"\n",
				"])\n\n",
				"for result in results:\n",
				"    print(f\"Label: {result['label']}, Score: {result['score']:.4f}\")",
			},
		},
	}
}

func createTextGenerationCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "text-gen-md"},
			Source: []string{
				"### 2.2. Task: Text Generation\n",
				"Here, we'll use a large language model (LLM) to generate text based on a prompt. We'll use DistilGPT-2, a smaller, faster version of GPT-2, which is perfect for experimentation in Colab.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "text-gen-code"},
			Source: []string{
				"generator = pipeline(\"text-generation\", model=\"distilgpt2\")\n\n",
				"prompt = \"In a world where AI has become sentient, the first thing it did was\"\n\n",
				"outputs = generator(\n",
				"    prompt,\n",
				"    max_length=50, # The total length of the output text\n",
				"    num_return_sequences=2 # How many different completions to generate\n",
				")\n\n",
				"for i, output in enumerate(outputs):\n",
				"    print(f\"--- Option {i+1} ---\")\n",
				"    print(output['generated_text'])\n",
				"    print()",
			},
		},
	}
}

func createZeroShotClassificationCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "zero-shot-md"},
			Source: []string{
				"### 2.3. Task: Zero-Shot Classification\n",
				"This is one of the most powerful features. You can classify text into labels you define **on the fly**, without ever having to fine-tune the model on those specific labels. The model leverages its understanding of language to determine the best fit.\n\n",
				"It works by posing the problem as Natural Language Inference (NLI), where the model checks if a hypothesis (e.g., \"This text is about politics\") is entailed by the premise (the input text).",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "zero-shot-code"},
			Source: []string{
				"zero_shot_classifier = pipeline(\"zero-shot-classification\")\n\n",
				"sequence = \"The government announced a new tax policy for renewable energy.\"\n",
				"candidate_labels = [\"education\", \"politics\", \"business\", \"sports\"]\n\n",
				"result = zero_shot_classifier(sequence, candidate_labels)\n",
				"print(result)",
			},
		},
	}
}

func createNamedEntityRecognitionCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "ner-md"},
			Source: []string{
				"### 2.4. Task: Named Entity Recognition (NER)\n\n",
				"NER models identify and categorize named entities in text, such as persons, organizations, locations, and dates. This is fundamental for information extraction.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "ner-code"},
			Source: []string{
				"ner = pipeline(\"ner\", grouped_entities=True)\n",
				"text = \"My name is Clara and I live in Naples. I work for Google.\"\n\n",
				"entities = ner(text)\n",
				"print(entities)",
			},
		},
	}
}

func createSummarizationCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "summarization-md"},
			Source: []string{
				"### 2.5. Task: Summarization\n\n",
				"Summarization models create a shorter version of a text while preserving its main points. The default models are typically **abstractive**, meaning they can generate new phrases that are not in the original text, as opposed to **extractive** models which just copy key sentences.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "summarization-code"},
			Source: []string{
				"summarizer = pipeline(\"summarization\")\n\n",
				"article = \"\"\"\n",
				"The James Webb Space Telescope (JWST) has captured stunning new images of the Pillars of Creation, a stellar nursery located 6,500 light-years away in the Eagle Nebula. \n",
				"The images, taken with JWST's Near-Infrared Camera (NIRCam), reveal previously unseen details of the iconic gas and dust columns where new stars are forming. \n",
				"Compared to the Hubble Space Telescope's famous 1995 image, JWST's view pierces through much of the obscuring dust, unveiling hundreds of newly-formed stars that were previously hidden. \n",
				"Scientists are analyzing the data to better understand the processes of star formation and how the intense ultraviolet radiation from massive young stars shapes the surrounding nebula.\n",
				"\"\"\"\n\n",
				"summary = summarizer(article, max_length=60, min_length=25, do_sample=False)\n",
				"print(summary[0]['summary_text'])",
			},
		},
	}
}

func createPipelineDeepDiveCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "pipeline-deep-dive-md"},
			Source: []string{
				"### 2.6. Pipeline Deep Dive: Customization and More Examples\n\n",
				"The `pipeline` is powerful because it finds a great balance between ease-of-use and flexibility. It's the perfect tool for most application developers and for rapid prototyping.\n\n",
				"**Why is it so effective?**\n",
				"- **Abstraction**: It hides the complex tokenization, model inference, and decoding logic.\n",
				"- **Defaults**: It automatically selects a reasonable default model for a given task, so you don't have to search the Hub yourself initially.\n",
				"- **Optimization**: It handles details like batching inputs for greater efficiency.\n\n",
				"Beyond just the `task` name, you can customize a pipeline with several key arguments:\n",
				"- `model`: Specify any model checkpoint from the Hub that is compatible with the task. For example, `pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment')`.\n",
				"- `tokenizer`: You can provide a specific tokenizer, though this is usually loaded automatically with the model.\n",
				"- `device`: Manually assign the pipeline to a specific device. Use `device=0` for the first GPU, `device=1` for the second, and `device=-1` for the CPU. This is crucial for controlling hardware usage.\n\n",
				"Let's explore a few more diverse pipelines.",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "qa-pipeline-md"},
			Source: []string{
				"#### Example: Question Answering\n",
				"This pipeline extracts an answer to a question from a given context document. It doesn't generate text; it finds the span of text in the context that best answers the question.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "qa-pipeline-code"},
			Source: []string{
				"from transformers import pipeline\n\n",
				"qa_pipeline = pipeline(\"question-answering\")\n\n",
				"context = \"\"\"\n",
				"The city of Naples, in southern Italy, is the regional capital of Campania. It is the third-most populated city in Italy, after Rome and Milan. \n",
				"Famous for its rich history, art, culture, and gastronomy, its historic city centre is a UNESCO World Heritage Site. \n",
				"The nearby Mount Vesuvius is a well-known volcano famous for its eruption in AD 79 that destroyed the Roman city of Pompeii.\n",
				"\"\"\"\n",
				"question = \"What is the famous volcano near Naples?\"\n\n",
				"result = qa_pipeline(question=question, context=context)\n",
				"print(f\"Answer: '{result['answer']}' with score {result['score']:.4f}\")",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "translation-pipeline-md"},
			Source: []string{
				"#### Example: Translation\n",
				"This uses a sequence-to-sequence model to translate text from one language to another.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "translation-pipeline-code"},
			Source: []string{
				"# We must specify a model for translation as there's no universal default\n",
				"# 'Helsinki-NLP' provides hundreds of high-quality translation models\n",
				"translator = pipeline(\"translation_en_to_de\", model=\"Helsinki-NLP/opus-mt-en-de\")\n\n",
				"text = \"I am studying artificial intelligence at the university.\"\n",
				"translation = translator(text)\n\n",
				"print(f\"English: {text}\")\n",
				"print(f\"German: {translation[0]['translation_text']}\")",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "object-detection-pipeline-md"},
			Source: []string{
				"#### Example: Object Detection\n",
				"This computer vision pipeline identifies multiple objects in an image and returns their class labels along with bounding boxes.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "object-detection-pipeline-code"},
			Source: []string{
				"from PIL import Image\n",
				"import requests\n\n",
				"# Let's use a DETR (DEtection TRansformer) model\n",
				"object_detector = pipeline(\"object-detection\", model=\"facebook/detr-resnet-50\")\n\n",
				"url = \"https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg\"\n",
				"image = Image.open(requests.get(url, stream=True).raw)\n",
				"display(image)\n\n",
				"objects = object_detector(image)\n\n",
				"print(\"\\nDetected objects:\")\n",
				"for obj in objects:\n",
				"    print(f\"- Label: {obj['label']}, Score: {obj['score']:.2f}, Box: {obj['box']}\")",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "all-pipelines-list-md"},
			Source: []string{
				"### Complete List of Available Pipelines\n\n",
				"For your reference, here is a list of many of the available pipeline tasks you can explore:\n\n",
				"**Audio**\n",
				"- `audio-classification`\n",
				"- `automatic-speech-recognition`\n",
				"- `text-to-audio`\n",
				"- `zero-shot-audio-classification`\n\n",
				"**Computer Vision**\n",
				"- `depth-estimation`\n",
				"- `image-classification`\n",
				"- `image-segmentation`\n",
				"- `image-to-image`\n",
				"- `object-detection`\n",
				"- `video-classification`\n",
				"- `zero-shot-image-classification`\n",
				"- `zero-shot-object-detection`\n\n",
				"**Natural Language Processing**\n",
				"- `fill-mask`\n",
				"- `question-answering`\n",
				"- `summarization`\n",
				"- `table-question-answering`\n",
				"- `text-classification` (same as `sentiment-analysis`)\n",
				"- `text-generation`\n",
				"- `text2text-generation`\n",
				"- `token-classification` (same as `ner`)\n",

				"- `translation`\n",
				"- `zero-shot-classification`\n\n",
				"**Multimodal**\n",
				"- `document-question-answering`\n",
				"- `feature-extraction`\n",
				"- `image-to-text`\n",
				"- `visual-question-answering`",
			},
		},
	}
}

func createUnderTheHoodIntroCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "under-hood-intro-md"},
			Source: []string{
				"## 3. Under the Hood: Models, Tokenizers, and Datasets\n\n",
				"The `pipeline` is magical, but what's actually happening behind the scenes? To gain true mastery, we need to understand the three core components that every transformer-based workflow uses:\n\n",
				"1.  **Tokenizer**: Converts raw text into a numerical representation the model can understand.\n",
				"2.  **Model**: The neural network architecture (the \"brains\") that processes the numerical inputs and produces outputs (logits).\n",
				"3. **Dataset**: A standard representation for data that integrates with the other components.\n\n",
				"Let's replicate the sentiment analysis task from before, but this time, we'll perform each step manually.",
			},
		},
	}
}

func createTokenizerCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "tokenizer-md"},
			Source: []string{
				"### 3.1. The Tokenizer\n\n",
				"Transformers don't see words; they see numbers called **tokens**. The tokenizer's job is to perform this conversion. Modern tokenizers use a **subword tokenization** algorithm (like WordPiece for BERT or BPE for GPT). This allows them to handle any word, even ones they haven't seen before, by breaking them down into smaller, known pieces.\n\n",
				"A tokenizer returns a dictionary containing:\n",
				"- `input_ids`: The numerical representation of the tokens.\n",
				"- `attention_mask`: A binary tensor indicating which tokens the model should pay attention to (useful for batching sentences of different lengths).",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "tokenizer-code"},
			Source: []string{
				"from transformers import AutoTokenizer\n\n",
				"# We'll use the same model checkpoint as the default sentiment-analysis pipeline\n",
				"checkpoint = \"distilbert-base-uncased-finetuned-sst-2-english\"\n",
				"tokenizer = AutoTokenizer.from_pretrained(checkpoint)\n\n",
				"raw_inputs = [\n",
				"    \"This course is incredibly informative and well-structured!\",\n",
				"    \"I'm not sure I understand everything, the pace is a bit too fast.\"\n",
				"]\n\n",
				"# The tokenizer handles padding and truncation automatically to create a rectangular tensor\n",
				"inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors=\"pt\")\n\n",
				"print(inputs)",
			},
		},
	}
}

func createModelCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "model-md"},
			Source: []string{
				"### 3.2. The Model\n\n",
				"Now we load the model itself. The `AutoModel` classes are factory classes that instantiate the correct model architecture based on the provided checkpoint. Since we're doing sequence classification, we use `AutoModelForSequenceClassification`.\n\n",
				"The model takes the `input_ids` and `attention_mask` from the tokenizer and outputs **logits** — raw, unnormalized scores for each class. To turn these into probabilities, we apply the **Softmax** function.\n\n",
				"$$\\text{Softmax}(z_i) = \\frac{e^{z_i}}{\\sum_{j=1}^{K} e^{z_j}} \\quad \\text{for } i=1, \\ldots, K$$",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "model-code"},
			Source: []string{
				"from transformers import AutoModelForSequenceClassification\n",
				"import torch\n\n",
				"# Load the model and move it to the GPU\n",
				"model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)\n\n",
				"# Move the input tensors to the same device\n",
				"inputs = {k: v.to(device) for k, v in inputs.items()}\n\n",
				"# Perform the forward pass\n",
				"with torch.no_grad(): # Disable gradient calculation for inference\n",
				"    outputs = model(**inputs)\n\n",
				"print(\"Logits:\", outputs.logits)\n\n",
				"# Convert logits to probabilities\n",
				"predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)\n",
				"print(\"\\nProbabilities:\", predictions)\n\n",
				"# Get the predicted label IDs\n",
				"predicted_labels = torch.argmax(predictions, dim=-1)\n",
				"print(\"\\nPredicted Label IDs:\", predicted_labels)\n\n",
				"# You can map the IDs back to human-readable labels\n",
				"print(\"\\nLabels:\", [model.config.id2label[label_id] for label_id in predicted_labels.tolist()])",
			},
		},
	}
}

func createDatasetsLibraryCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "datasets-lib-md"},
			Source: []string{
				"### 3.3. The `datasets` Library\n\n",
				"Training a model requires a lot of data. The `datasets` library is the standard way to interact with datasets in the Hugging Face ecosystem. It provides:\n\n",
				"- **Access to thousands of datasets** from the Hub with one line of code: `load_dataset()`.\n",
				"- **Efficient data processing**: It uses Apache Arrow on the backend, which memory-maps data, allowing you to work with datasets far larger than your RAM.\n",
				"- **Powerful methods** like `.map()`, `.filter()`, and `.shuffle()` that are optimized for speed.\n\n",
				"Let's load the IMDB dataset, which is a common benchmark for sentiment analysis.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "datasets-lib-code"},
			Source: []string{
				"from datasets import load_dataset\n\n",
				"# Load the IMDB dataset\n",
				"imdb_dataset = load_dataset(\"imdb\")\n\n",
				"print(imdb_dataset)\n\n",
				"# Let's look at one example from the training set\n",
				"print(\"\\n--- Example ---\")\n",
				"example = imdb_dataset[\"train\"][100]\n",
				"print(\"Text:\", example[\"text\"][:200] + \"...\")\n",
				"print(\"Label:\", example[\"label\"], \"(0=Negative, 1=Positive)\")",
			},
		},
	}
}

func createFinetuningIntroCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "finetuning-intro-md"},
			Source: []string{
				"## 4. Fine-Tuning a Model for a Custom Task\n\n",
				"This is where the magic of **transfer learning** comes in. We rarely train a transformer model from scratch. Instead, we start with a **pretrained** model that has already learned the nuances of a language from a massive text corpus (like Wikipedia) and **fine-tune** it on our specific, smaller dataset.\n\n",
				"We will fine-tune the `distilbert-base-uncased` model on the IMDB dataset for sentiment analysis. To simplify the training loop, we'll use the **`Trainer` API**, a feature-complete training and evaluation framework provided by the `transformers` library.",
			},
		},
	}
}

func createFinetuningDataPrepCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "finetuning-data-prep-md"},
			Source: []string{
				"### 4.1. Data Preparation\n\n",
				"The first step is to tokenize our dataset. We'll write a function to tokenize the text and then apply it to the entire dataset using the highly efficient `.map()` method from the `datasets` library. For faster demonstration, we'll only use small subsets of the full dataset.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "finetuning-data-prep-code"},
			Source: []string{
				"from transformers import AutoTokenizer\n",
				"from datasets import load_dataset\n\n",
				"# Load tokenizer and dataset\n",
				"tokenizer = AutoTokenizer.from_pretrained(\"distilbert-base-uncased\")\n",
				"imdb = load_dataset(\"imdb\")\n\n",
				"# Create small subsets for quicker training\n",
				"train_subset = imdb[\"train\"].shuffle(seed=42).select(range(1000))\n",
				"test_subset = imdb[\"test\"].shuffle(seed=42).select(range(1000))\n\n",
				"# Tokenization function\n",
				"def tokenize_function(examples):\n",
				"    return tokenizer(examples[\"text\"], truncation=True)\n\n",
				"# Apply the function to the datasets\n",
				"tokenized_train = train_subset.map(tokenize_function, batched=True)\n",
				"tokenized_test = test_subset.map(tokenize_function, batched=True)",
			},
		},
	}
}

func createAdvancedUseCasesIntroCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "advanced-intro-md"},
			Source: []string{
				"## 5. Beyond Text: Vision and Audio\n\n",
				"The 'T' in 'Transformer' doesn't just stand for Text! The same fundamental architecture has been adapted with incredible success for computer vision and audio processing tasks. The `pipeline` API makes it just as easy to use these models.",
			},
		},
	}
}

func createVisionCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "vision-md"},
			Source: []string{
				"### 5.1. Computer Vision: Image Classification\n\n",
				"The **Vision Transformer (ViT)** model treats an image as a sequence of patches, similar to how a text transformer treats a sentence as a sequence of words. Let's use a ViT model pretrained on ImageNet to classify an image.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "vision-code"},
			Source: []string{
				"from transformers import pipeline\n",
				"import requests\n",
				"from PIL import Image\n",
				"import io\n\n",
				"image_classifier = pipeline(\"image-classification\", model=\"google/vit-base-patch16-224\")\n\n",
				"# Let's grab an image of a cat\n",
				"url = \"https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg\"\n",
				"response = requests.get(url)\n",
				"image = Image.open(io.BytesIO(response.content))\n\n",
				"display(image)\n\n",
				"preds = image_classifier(image)\n",
				"for pred in preds:\n",
				"    print(f\"Label: {pred['label']}, Score: {pred['score']:.4f}\")",
			},
		},
	}
}

func createAudioCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "audio-md"},
			Source: []string{
				"### 5.2. Audio: Automatic Speech Recognition (ASR)\n\n",
				"Models like OpenAI's **Whisper** have revolutionized ASR. They can transcribe speech from audio files with remarkable accuracy. Here, we'll use the `datasets` library to load a sample audio file from the LibriSpeech dataset and transcribe it.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "audio-code"},
			Source: []string{
				"from transformers import pipeline\n",
				"from datasets import load_dataset\n\n",
				"# Use a smaller, distilled version of Whisper for faster inference\n",
				"asr_pipeline = pipeline(\"automatic-speech-recognition\", model=\"distil-whisper/distil-small.en\")\n\n",
				"# Load a sample from an audio dataset\n",
				"dataset = load_dataset(\"hf-internal-testing/librispeech_asr_dummy\", \"clean\", split=\"validation\")\n",
				"audio_sample = dataset[0][\"audio\"]\n\n",
				"print(\"Transcribing audio...\")\n",
				"transcription = asr_pipeline(audio_sample)\n",
				"print(transcription[\"text\"])",
			},
		},
	}
}

func createHubCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "hub-md"},
			Source: []string{
				"## 6. The Hugging Face Hub: The GitHub of Machine Learning\n\n",
				"The Hub is the central place where the community shares models, datasets, and demos (Spaces). After fine-tuning a model, you can easily share it with the world (or your team) by pushing it to the Hub.\n\n",
				"To do this, you'll need:\n",
				"1. A Hugging Face account ([huggingface.co](https://huggingface.co)).\n",
				"2. An access token. Go to `Your Profile > Settings > Access Tokens > New token` (with `write` role).\n\n",
				"We can then log in from our notebook and push the model we just trained.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "hub-login-code"},
			Source: []string{
				"from huggingface_hub import notebook_login\n\n",
				"# This will prompt you to enter your access token\n",
				"notebook_login()",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "hub-push-code"},
			Source: []string{
				"# Now, you can push the trainer object directly to the Hub!\n",
				"# It will create a new repository under your username.\n",
				"# Make sure to give it a unique name.\n",
				"repo_name = \"my-awesome-imdb-sentiment-model\"\n",
				"trainer.push_to_hub(repo_name)",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "hub-conclusion-md"},
			Source: []string{
				"Once pushed, anyone can load your fine-tuned model using `pipeline` or `AutoModel.from_pretrained('your-username/your-repo-name')`. This is the power of open and collaborative machine learning!",
			},
		},
	}
}

func createConclusionCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "conclusion-md"},
			Source: []string{
				"## 7. Conclusion & Next Steps\n\n",
				"Congratulations! You've completed a comprehensive tour of the Hugging Face ecosystem. We've covered:\n\n",
				"- ✅ Using the high-level `pipeline` for various tasks across multiple modalities (text, vision, audio).\n",
				"- ✅ Understanding the core components: `Tokenizer` and `Model`.\n",
				"- ✅ Leveraging the `datasets` library for efficient data handling.\n",
				"- ✅ Fine-tuning a pretrained model for a custom task using the `Trainer` API.\n",
				"- ✅ Sharing your work with the community by pushing a model to the Hub.\n\n",
				"This is just the beginning. The field is vast and rapidly evolving. Here are some excellent resources to continue your journey:\n\n",
				"- **The Hugging Face Course**: An in-depth, free course covering everything from basics to advanced topics. [Link](https://huggingface.co/course)\n",
				"- **PEFT Library**: For Parameter-Efficient Fine-Tuning techniques like LoRA, which allow you to fine-tune massive models on a single GPU. [Link](https://github.com/huggingface/peft)\n",
				"- **TRL Library**: For training language models with Reinforcement Learning (like in RLHF). [Link](https://github.com/huggingface/trl)\n",
				"- **The Hugging Face Blog**: For the latest research and library updates. [Link](https://huggingface.co/blog)\n\n",
				"Happy transforming! 🚀",
			},
		},
	}
}

func createGradioCells() []Cell {
	return []Cell{
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "gradio-intro-md"},
			Source: []string{
				"### 4.3. Creating an Interactive Demo with Gradio\n\n",
				"Now that you've fine-tuned a model, how can you share it with others? The **Gradio** library makes it incredibly simple to create a web-based user interface for any Python function, especially for ML models. It's the technology that powers many Demos (Spaces) on the Hugging Face Hub.\n\n",
				"The core idea is simple:\n",
				"1.  Define a function that takes some inputs and returns some outputs.\n",
				"2.  Define the input and output components (e.g., a text box, an image uploader).\n",
				"3.  Launch the `Interface`.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "gradio-install-code"},
			Source: []string{
				"# First, we need to install the gradio library\n",
				"!pip install -q gradio",
			},
		},
		{
			CellType: "markdown",
			Metadata: map[string]any{"id": "gradio-example-md"},
			Source: []string{
				"Let's create a demo for the summarization pipeline we used earlier. We'll wrap the pipeline call inside a function and then build a Gradio interface around it.",
			},
		},
		{
			CellType: "code",
			Metadata: map[string]any{"id": "gradio-example-code"},
			Source: []string{
				"import gradio as gr\n",
				"from transformers import pipeline\n\n",
				"# 1. Load our model (in this case, a pipeline)\n",
				"summarizer = pipeline(\"summarization\")\n\n",
				"# 2. Define the function that will use the model\n",
				"def summarize_text(article):\n",
				"    \"\"\"This function takes an article and returns its summary.\"\"\"\n",
				"    summary_list = summarizer(article, max_length=100, min_length=30, do_sample=False)\n",
				"    return summary_list[0]['summary_text']\n\n",
				"# 3. Create and launch the Gradio Interface\n",
				"demo = gr.Interface(\n",
				"    fn=summarize_text, # The function to wrap\n",
				"    inputs=gr.Textbox(lines=10, placeholder=\"Enter your article here...\", label=\"Article\"), # Input component\n",
				"    outputs=gr.Textbox(label=\"Summary\"), # Output component\n",
				"    title=\"Text Summarizer 📝\",\n",
				"    description=\"A simple app to summarize long articles using a Hugging Face transformer model.\",\n",
				"    allow_flagging=\"never\"\n",
				")\n\n",
				"# Launch the app! It will appear directly in your Colab notebook output.\n",
				"demo.launch()",
			},
		},
	}
}

