from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from efficientnet_pytorch import EfficientNet

app = Flask(__name__)

# Load a pre-trained ResNet model
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
model.load_state_dict(torch.load('./BestModel.pt', map_location='cpu'))


mean = [0.5375, 0.4315, 0.3818]
std = [0.2928, 0.2656, 0.2614]
# Define a transformation
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    transforms.Grayscale()
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part in the request', 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    height, width = image.size
    start_width = int(width / 2 - height / 2)
    end_width = int(width / 2 + height / 2)
    image = image[:, start_width:end_width]
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()

    return jsonify({'predicted_class': predicted_class})


if __name__ == '__main__':
    app.run(debug=True)