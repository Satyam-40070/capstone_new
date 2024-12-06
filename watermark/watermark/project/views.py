from django.http import HttpResponse
from django.shortcuts import render
from PIL import Image
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .script import *
import cv2
import base64
from io import BytesIO

def index(request):
    print("I request##3")
    return render(request, 'project/index.html')

# @csrf_exempt
# def process_image(request):
#     if request.method == 'POST':
#         # Get the uploaded image from the request
#         image_file = request.FILES.get('image')
        
#         # Check if an image was uploaded
#         if image_file is None:
#             return JsonResponse({'error': 'No image provided'}, status=400)
        
#         # Open the image using PIL
#         try:
#             image = np.array(Image.open(image_file))
#         except Exception as e:
#             return JsonResponse({'error': 'Invalid image file'}, status=400)
        
#         # skipcq: PYL-W0612
#         # fs = FileSystemStorage(location=os.path.join(settings.BASE_DIR, 'static/project'))

#         images_path = resize(image)
#         print(images_path)
#         return render(request, 'project/index.html', {'images_path': images_path})
 
    
#     return JsonResponse({'error': 'Invalid request method'}, status=405)



# imageprocessor/views.py
from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from .script2 import run_model
import base64


class ImageResizeView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        print("called function")
        print("Request Files:", request.FILES)
        print("Keys in request.FILES:", list(request.FILES.keys()))
        print(*args, **kwargs)

        files = request.FILES.getlist('image1') #'file_field' --> 'imagen' for you
   
        for f in files:
            print(f"f name is {f}")
            print(f.name)
        # Get the image from the request
        if 'image1' in request.FILES and 'image2' in request.FILES:
            for file_key, uploaded_file in request.FILES.items():
                print(f"Key: {file_key}")
                print(f"File Name: {uploaded_file.name}")
                print(f"File Size: {uploaded_file.size} bytes")
                print(f"Content Type: {uploaded_file.content_type}")
            image_file1= request.FILES.get('image1')
            image_file2= request.FILES.get('image2')

            upload_directory = os.path.join(settings.BASE_DIR, 'uploads')  # or your desired path

            # Create the directory if it doesn't exist
            os.makedirs(upload_directory, exist_ok=True)
            file_path1, file_path2 = "", ""
            # Save the first uploaded file
            if image_file1:
                file_path1 = os.path.join(upload_directory, image_file1.name)
                with open(file_path1, 'wb+') as destination:
                    for chunk in image_file1.chunks():
                        destination.write(chunk)

            # Save the second uploaded file
            if image_file2:
                file_path2 = os.path.join(upload_directory, image_file2.name)
                with open(file_path2, 'wb+') as destination:
                    for chunk in image_file2.chunks():
                        destination.write(chunk)

            print(file_path1, file_path2)


            try:
                image1 = np.array(Image.open(image_file1))
                Image.open(image_file1)
                image2 = np.array(Image.open(image_file2))
              
            except Exception as e:
                return JsonResponse({'error': 'Invalid image file'}, status=400)
            
            print("got upto loading files=________yes")
            resized_image = run_model(file_path1, file_path2)

            image_path = r"C:\Users\jainh\OneDrive\Desktop\Capstone_sh\Capstone_sh\watermark\watermark\decoded_cover_image.jpg"  # Adjust the path as necessary

                # Check if the file exists
            if os.path.exists(image_path):
                    # Read the image file
                    with open(image_path, 'rb') as image_file:
                        # Encode the image to base64
                        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                        img_data_url = f"data:image/jpeg;base64,{img_base64}"

                    return JsonResponse({'image': img_data_url})
            else:
                    return JsonResponse({'error': 'Image not found'}, status=404)
            # if resized_image is not None and resized_image.size > 0:
            #     print(type(resized_image))  # Should be <class 'numpy.ndarray'>
            #     print(resized_image.shape)   # Should be (1, 256, 256, 3)
            #     print(resized_image.dtype)    # Should be float32

            #     # Remove the first dimension if it’s a batch of one image
            #     if resized_image.ndim == 4:  # Check if the array has 4 dimensions
            #         resized_image = np.squeeze(resized_image, axis=0)  # Convert to shape (256, 256, 3)

            #     # Convert from float32 to uint8 (0-255 range)
            #     resized_image = (resized_image * 255).astype(np.uint8)

            #     # Encode to JPEG
            #     _, img_encoded = cv2.imencode('.jpg', resized_image)

            #     # Convert to base64
            #     img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
            #     img_data_url = f"data:image/jpeg;base64,{img_base64}"

            #     return JsonResponse({'image': img_data_url})

            # else:
            #     return JsonResponse({'error': "Resized image is empty or None"})
