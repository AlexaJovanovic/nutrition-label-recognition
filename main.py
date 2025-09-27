import easyocr
import cv2

path = r'./generated_labels/'
img_name = "nutrition_label_0.png"

reader = easyocr.Reader(['en'], gpu=False)

result1 = reader.readtext(path + "table_transformed.png")
print(result1)

print("Main finished succesfully")