from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import ImageUploadForm
from .utils import predict_caption

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_path, image_url = handle_uploaded_file(image)
            caption = predict_caption(image_path)
            return render(request, 'captions/result.html', {
            'image_url': image_url,
            'caption': str(caption).capitalize()})
    else:
        form = ImageUploadForm()
    return render(request, 'captions/upload.html', {'form': form})

def handle_uploaded_file(f):
    fs = FileSystemStorage() 
    filename = fs.save(f.name, f)
    return fs.path(filename), fs.url(filename)
