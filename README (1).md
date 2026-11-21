# Visual scene understanding system using LLMs

Computer vision pipeline that detects, segments and describes the objects in an image and answers questions about the scene.

### Main idea:
Turn an image into a structured, verifiable representation of the scene (objects, attributes and spatial relationships) that can be queried deterministically or in natural language through a lightweight LLM.


### Technologies used:
- Python
- PyTorch
- YOLO-World (Ultralytics)
- OWLv2 (Hugging Face Transformers)
- SAM2
- Qwen2.5-1.5B-Instruct
- Weighted Boxes Fusion
- OpenCV
- Pandas / NumPy
- Matplotlib
- Kaggle


### Architectural aspects:
- Modular pipeline: detection, segmentation, representation and reasoning
- Ensemble of two open-vocabulary detectors (YOLO-World + OWLv2) and segmentation with SAM2
- Scene representation in JSON with objects, attributes and spatial relationships
- Natural language questions answered by a lightweight LLM, without hallucinations
