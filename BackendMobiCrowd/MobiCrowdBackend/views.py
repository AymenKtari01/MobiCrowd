import json
import base64
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt # type: ignore
from MobiCrowdBackend.autoencoder import *
from MobiCrowdBackend.UNIQUE.ScoreModelUtils import load_score_model, evaluate
from PIL import Image  
import io
import tensorflow as tf 
import keras 

# verdict = True 

def load_and_preprocess_embedding(request):
    payload = json.loads(request.body)
    encoded = payload.get('encoded', [])
    if not isinstance(encoded, list):
        raise ValueError("Encoded data should be a list")
    encoded_array = np.array(encoded)
    return encoded_array

# def classify_embedding(encoded_array):
#     classes = ['Flood', 'Fire', 'Other']
#     classification_model = load_tflite_model()

#     # Prepare input data
#     input_data = encoded_array[:, :, :16, :].astype(np.float32)
    
#     # Set input and output details
#     input_details = classification_model.get_input_details()
#     output_details = classification_model.get_output_details()
    
#     # Check if input shape matches
#     classification_model.set_tensor(input_details[0]['index'], input_data)
#     classification_model.invoke()
    
#     # Get output tensor
#     predicted_class = classification_model.get_tensor(output_details[0]['index'])
    
#     predicted_class_index = np.argmax(predicted_class, axis=1)[0]
#     predicted_class = classes[int(predicted_class_index)]
    
#     return predicted_class

def classify_embedding(encoded_array):  

    print('SHAPEEE' , np.shape(encoded_array) )
    classes = ['Fire', 'Flood', 'Other']
    classification_model = classif_model()
    prediction = classification_model.predict(encoded_array)
    print('prediction', prediction )
    predicted_class_index = np.argmax(prediction)
    print('index',predicted_class_index)
    predicted_class = classes[int(predicted_class_index)]
    return predicted_class 

def decode_embedding(encoded_array):
    decoder = decoderr()
    decoded = decoder.predict(encoded_array)
    decoded_array = np.array(decoded.tolist())
    print(f"Decoded array shape: {decoded_array.shape}")
    return decoded_array

def extract_metadata(decoded_array):
    reconstructed_metadata = decoded_array[0][:, 64:, :][0]
    final_metadata = []
    value = 0
    for i in range(0, 24, 3):
        for j in range(3):
            value += np.mean(reconstructed_metadata[i+j])
        final_metadata.append(value / 3)
        value = 0
    return final_metadata

def prepare_image(decoded_array):
    img_array = decoded_array[0][0:64, 0:64 , :]
    img_array = (img_array * 255).astype(np.uint8)
    img = Image.fromarray(img_array)

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def evaluate_score(decoded_array):
    score_model = load_score_model()
    score, _ = evaluate(score_model, decoded_array[0][:64, :64, :])
    return score

@csrf_exempt
def receive_embedding(request):
    if request.method == 'POST':
        try:
            # Process the embedding
            encoded_array = load_and_preprocess_embedding(request)
            print('**********************************************************')
            print(tf.__version__)
            print(keras.__version__)
            print()
            predicted_class  = classify_embedding(encoded_array)
            print('////////////////////////////////////////////////////////////////////')
  
            # decoded_array = decode_embedding(encoded_array)
            # # score = evaluate_score(decoded_array)
            # final_metadata = extract_metadata(decoded_array)
  
            # img_str = prepare_image(decoded_array)

            return JsonResponse({
                'status': 'success',
                # 'image_base64': img_str,
                # 'metadata': final_metadata,
                'predicted_class': predicted_class 
                # 'verdict': verdict ,
                # 'Reconstructed image Score ': score
            }, status=200)
        
        except Exception as e:
            print(f"Error: {str(e)}")
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)



# import json
# import base64
# import numpy as np
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from MobiCrowdBackend.autoencoder import *
# from MobiCrowdBackend.UNIQUE.ScoreModelUtils import load_score_model, evaluate
# from PIL import Image  
# import io

# # Define a score threshold
# SCORE_THRESHOLD = 0.75  

# def load_and_preprocess_embedding(request):
#     payload = json.loads(request.body)
#     encoded = payload.get('encoded', [])
#     if not isinstance(encoded, list):
#         raise ValueError("Encoded data should be a list")
#     encoded_array = np.array(encoded)
#     return encoded_array

# # def classify_embedding(encoded_array):
# #     classes = ['Flood', 'Fire', 'Other']
# #     classification_model = classif_model()
# #     predicted_class = classification_model.predict(encoded_array[:, :, :16, :])
# #     predicted_class_index = np.argmax(predicted_class, axis=1)[0]
# #     predicted_class = classes[int(predicted_class_index)]
# #     return predicted_class 

# def classify_embedding(encoded_array):
#     classes = ['Flood', 'Fire', 'Other']
#     classification_model = load_tflite_model()

#     # Prepare input data
#     input_data = encoded_array[:, :, :16, :].astype(np.float32)
    
#     # Set input and output details
#     input_details = classification_model.get_input_details()
#     output_details = classification_model.get_output_details()
    
#     # Check if input shape matches
#     classification_model.set_tensor(input_details[0]['index'], input_data)
#     classification_model.invoke()
    
#     # Get output tensor
#     predicted_class = classification_model.get_tensor(output_details[0]['index'])
    
#     predicted_class_index = np.argmax(predicted_class, axis=1)[0]
#     predicted_class = classes[int(predicted_class_index)]
    
#     return predicted_class

# def evaluate_image_quality(image_array):
#     score_model = load_score_model()
#     score, _ = evaluate(score_model, image_array)
#     return score

# @csrf_exempt
# def receive_image(request):
#     if request.method == 'POST':
#         try:
#             payload = json.loads(request.body)
#             image_base64 = payload.get('image_base64', '')
#             image_data = base64.b64decode(image_base64)
#             image = Image.open(io.BytesIO(image_data)).convert('RGB')
#             image_array = np.array(image)
#             print(f"Received image shape: {image_array.shape}")

#             # Evaluate the quality score
#             score = evaluate_image_quality(image_array)
#             print(f"Image score: {score}")

#             if score < SCORE_THRESHOLD:
#                 return JsonResponse({
#                     'status': 'rejected',
#                     'message': 'Image quality is too low. Please submit a higher-quality image.',
#                     'score': score
#                 }, status=200)
#             else:
#                 return JsonResponse({
#                     'status': 'accepted',
#                     'message': 'Image accepted.',
#                     'score': score,
#                     'image_base64': image_base64
#                 }, status=200)
        
#         except Exception as e:
#             print(f"Error: {str(e)}")
#             return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
#     else:
#         return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

# @csrf_exempt
# def receive_embedding(request):
#     if request.method == 'POST':
#         try:
#             encoded_array = load_and_preprocess_embedding(request)
#             predicted_class = classify_embedding(encoded_array)

#             if predicted_class in ['Flood', 'Fire']:
#                 return JsonResponse({
#                     'status': 'success',
#                     'predicted_class': predicted_class,
#                     'message': 'Send original image for quality assessment.'
#                 }, status=200)
#             else:
#                 return JsonResponse({
#                     'status': 'rejected',
#                     'predicted_class': predicted_class,
#                     'message': 'Embedding does not belong to a desired class.'
#                 }, status=200)
        
#         except Exception as e:
#             print(f"Error: {str(e)}")
#             return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
#     else:
#         return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)
