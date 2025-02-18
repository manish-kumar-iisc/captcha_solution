
import os
import cv2
import matplotlib.pyplot as plt
import pytesseract
import re
import pandas as pd
from tqdm import tqdm
import pdb
import numpy as np

def convert_to_int(text):
    try:
        return int(text)
    except ValueError:
        return None
    
def load_data(path):
    images=[]
    print("number of image in directory: ", len(os.listdir(path)))
    for filename in os.listdir(path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path=os.path.join(path,filename)
            image=cv2.imread(image_path)
            image=pre_processing(image)
            if image is not None:
                images.append([image_path,filename,image])
    return images

def read_labels():
    data=pd.read_csv("/home/manish/test_1/captch/dataset/dataset/captcha_data.csv", header=0,)
    data['image_path']=data['image_path'].str.split('/').str[-1]
    data_list=data.to_dict(orient='records')
    data_list={item['image_path']:item['solution'] for item in data_list}
    return data_list

def load_vit_model(model_type="vit_base"):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    if model_type=="vit_base":
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

    else:
        processor = TrOCRProcessor.from_pretrained("./fine_tuned_trocr")
        model = VisionEncoderDecoderModel.from_pretrained("./fine_tuned_trocr")

    # processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
    # model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed')



    return processor, model

def inference_vit_model(processor, model, image):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def model_v1_pytessrect(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Otsu's thresholding
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use Tesseract to extract text from the thresholded image
    extracted_text = pytesseract.image_to_string(thresholded_image)

    return extracted_text

def pre_processing(image):
    ''' doing threshold & gaussing blurring twice '''
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Otsu's thresholding
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # thresholded_image = np.expand_dims(thresholded_image, axis=-1)
    thresholded_image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2RGB)
    print("image shape",np.shape(thresholded_image))
    return thresholded_image

def model_v2_pytessrect(image):
    ''' doing threshold & gaussing blurring twice '''
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Otsu's thresholding
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(thresholded_image, (5, 5), 0)

    # Apply Otsu's thresholding
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use Tesseract to extract text from the thresholded image
    extracted_text = pytesseract.image_to_string(thresholded_image)

    return extracted_text

def precision_with_levenshtein(data):
    from Levenshtein import distance as levenshtein_distance
    import math

    def edit_distance_precision(ground_truth,detected):
        if not detected:  # If detected CAPTCHA is empty, precision is 0
            return 0.0  

        # Compute Levenshtein edit distance
        edit_dist = levenshtein_distance(ground_truth, detected)
        print(edit_dist, ground_truth, detected)
        # Normalize using the maximum length between ground truth and detected CAPTCHA
        max_length = max(len(ground_truth), len(detected))

        precision = 1 - (edit_dist / max_length)  # Precision based on edit distance
        return max(0, precision)  # Ensure precision is not negative

    
    detected_captchas=list(data["cleaned_text"])
    ground_truths=list(data["ground_labels"])
    avg_precision=0
    for gt, det in zip(ground_truths, detected_captchas):
        # pdb.set_trace()
        gt=str(int(gt))
        if math.isnan(det):
            det=""
        else:
            det=str(int(det))
        precision = edit_distance_precision(gt, det)
        avg_precision+=precision
        # print(f"GT: {gt}, Detected: {det}, Precision: {precision:.2f}" if precision is not None else "No detected characters.")
    print("avg_edit_distance for incorrect samples:", avg_precision/len(ground_truths))


def extract_data(path, model_type="vit_base"):
    
    images=load_data(path)
    num_images=len(images)
    labels=read_labels()
    count=0
    unmatched_data=[]

    if "vit" in model_type:
        processor,model=load_vit_model()
    
    for _,filename,image in tqdm(images[:]):
        
        if "vit" in model_type:
            extracted_text=inference_vit_model(processor,model,image)

        if "pytessrect_v1" in model_type:
            extracted_text=model_v1_pytessrect(image)

        if "pytessrect_v2" in model_type:
            extracted_text=model_v2_pytessrect(image)

        # Remove any non-digit characters and spaces
        cleaned_text = re.sub(r'\D', '', extracted_text)

        # Print the filename and the cleaned text
        # print(f"Filename: {filename}, Extracted CAPTCHA text: {cleaned_text}")
        int_text=convert_to_int(cleaned_text)
        if int_text and int_text==int(labels[filename]):
            count+=1
        else:
            extracted_text=extracted_text.strip()
            if not extracted_text:
                extracted_text=None
            unmatched_data.append([filename, extracted_text, cleaned_text, labels[filename]])
    columns=["filename", "extracted_text", "cleaned_text", "ground_labels"]
    print("total_values: ", num_images, "correct: ", count, "precsion: ", round(count/num_images,3 ))
    unmatched_data=pd.DataFrame(unmatched_data, columns=columns)
    print(unmatched_data)
    unmatched_data.to_csv("./false_data_micro_base_ocr_test.csv")
    avg_edit_distance=precision_with_levenshtein(unmatched_data)

if __name__=="__main__":
    #training data
    # path="/home/manish/test_1/captch/dataset/dataset/train-images/train-images"

    # validation data
    path="/home/manish/test_1/captch/dataset/dataset/validation-images/validation-images/"
    model_type="vit_finetuned" # pytessrect_v1, pytessrect_v2, #vit_base
    extract_data(path, model_type="vit_finetuned")