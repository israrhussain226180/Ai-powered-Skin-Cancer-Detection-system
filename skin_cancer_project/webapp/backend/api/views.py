from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .model_loader import get_predictor
import io

@csrf_exempt
def predict_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        
        predictor = get_predictor()
        if predictor is None:
            return JsonResponse({'error': 'Model not loaded'}, status=500)
        
        try:
            # Prediction
            results = predictor.predict(io.BytesIO(image_file.read()))
            return JsonResponse(results)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Invalid request'}, status=400)
