import glob, json
from label_studio_converter.brush import image2annotation

####################################################

label_file = "dataset/labelling/labels"
label_save = "dataset/labelling/labels.json"

####################################################

files = glob.glob(f"{label_file}/*.png")
json_dump = []

for i, file in enumerate(files):
    image_path = "/data/local-files/?d=" + file.replace("dataset/", "").replace("labels", "images").replace("/", "%5C").replace("\\", "%5C")
    annotation = image2annotation(
        file,
        label_name='hand',
        from_name='tag',
        to_name='image',
    )

    task = {
            'id':i+1,
            'data': {'image': image_path},
            'predictions': [annotation],
        }
    
    json_dump.append(task)
    
json.dump(json_dump, open(label_save, 'w'))
print(f"{i+1} labels formatted & saved to {label_save}")