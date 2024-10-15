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
        if 'image1' in request.FILES or 'image2' in request.FILES:
            for file_key, uploaded_file in request.FILES.items():
                print(f"Key: {file_key}")
                print(f"File Name: {uploaded_file.name}")
                print(f"File Size: {uploaded_file.size} bytes")
                print(f"Content Type: {uploaded_file.content_type}")
            image_file1= request.FILES.get('image1')
            image_file2= request.FILES.get('image2')
            
            # if not image_file or image_file.content_type != 'image/jpeg' or image_file.content_type != 'image/jpg':
            #     print(image_file.content_type)
            #     return HttpResponse("Invalid image format. Please upload a JPEG image.", status=400)

            try:
                print("try block")
                print(type(image_file1))
                print(image_file1)


        #return HttpResponse(img_encoded.tobytes(), content_type='image/jpeg')


                image1 = np.array(Image.open(image_file1))
                Image.open(image_file1)

                image2 = np.array(Image.open(image_file2))
                cv2.imwrite('secretimage.jpg', image1)
                cv2.imwrite('imagecover1.jpg', image2)
                print("file decoed")
            except Exception as e:
                return JsonResponse({'error': 'Invalid image file'}, status=400)
            
            print("called resize")
            resized_image = run_model() # resize2(image1, image2)
            _, img_encoded = cv2.imencode('.jpg', resized_image)
            img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
            img_data_url = f"data:image/jpeg;base64,{img_base64}"
            print("aftere resize")
            return JsonResponse({'image': img_data_url})
            # return HttpResponse(img_encoded.tobytes(), content_type='image/jpeg')
        else:
            return JsonResponse({'error': "image not upload 2 "})