# DoodleItUpdated

Was a pleasure to work on this with Dr.Fred Martin and Vaishali Mahipal. Worked mostly on p5.js canvas functionality and integrating tensorflow json file with frontend. 

Logic:
- Dataset: Google QuickDraw (NumPy format, 28×28 grayscale)
- Classes: 6 classes, 50,000 images each (total: 300,000)
- Train/test split: 60% training, 40% testing

Labels assigned manually for all 6 classes. Sample images are printed to visualize the dataset.

All 6 classes are merged → shuffled → split

Preprocessing:
- Normalize pixel values to [0, 1]
- Reshape to 28×28

Separate into:
- X → image pixel data
- Y → class labels

<img width="991" height="450" alt="doodleitss" src="https://github.com/user-attachments/assets/130582ae-d786-401e-8abe-fefd276f5c6f" />
