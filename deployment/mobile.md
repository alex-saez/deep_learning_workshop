# Apple

Apple uses its library coreml

**dependencies**

```
pip install coremltools==0.8 Keras==2.1.3 imutils==0.4.6 opencv-python==3.4.1.15

pip install tensorflow-gpu==1.8.0 

pip install tensorflow==1.8.0 

```



```
from keras.models import load_model
import coremltools


python coremlconverter.py --model pokedex.model --labelbin lb.pickle

coreml_model = coremltools.converters.keras.convert(model,
	input_names="image",
	image_input_names="image",
	image_scale=1/255.0,
	class_labels=class_labels,
	is_bgr=True)

output = args["model"].rsplit(".", 1)[0] + ".mlmodel"
print("[INFO] saving model as {}".format(output))
coreml_model.save(output)


```



We will be using apple's sample

[Classifying Images with Vision and Core ML | Apple Developer Documentation](https://developer.apple.com/documentation/vision/classifying_images_with_vision_and_core_ml)




In the folder "Vision+ML Example/model", drag you model to that folder .

In the ImageClassificationViewController.swift , look for the line 
```
let model = try VNCoreMLModel(for: mobilenet().model)

```

replace mobilenet with the name of your custom model.




# Android

**Download Assests**
```
git clone https://github.com/googlecodelabs/tensorflow-for-poets-2

cd tensorflow-for-poets-2
```


**Train Model**

```
IMAGE_SIZE=224
PROBLEM=flower
IMAGE_LOCATION=tf_files/flower_photos

PROBLEM=dogbreed
ARCHITECTURE="mobilenet_1.0_${IMAGE_SIZE}"

IMAGE_LOCATION=~/.kaggle/competitions/dog-breed-identification/train_folder


mkdir -p tf_files/${PROBLEM} tf_files/${PROBLEM}/bottlenecks tf_files/${PROBLEM}/models tf_files/${PROBLEM}/training_summaries   

python -m scripts.retrain \
  --bottleneck_dir=tf_files/${PROBLEM}/bottlenecks \
  --model_dir=tf_files/${PROBLEM}/models/"${ARCHITECTURE}" \
  --summaries_dir=tf_files/${PROBLEM}/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/${PROBLEM}/graph.pb \
  --output_labels=tf_files/${PROBLEM}/labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir="${IMAGE_LOCATION}"

```


```
python -m scripts.label_image \
    --graph=tf_files/${PROBLEM}/graph.pb  \
    --image=${PREDICT_IMAGE_FILE} \
    --labels=tf_files/${PROBLEM}/labels.txt 
```


