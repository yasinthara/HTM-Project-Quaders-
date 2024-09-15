from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

with open("stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()

vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=True, vocabulary=pickle.load(open("tfidfvectoizer.pkl", "rb")))
model = pickle.load(open("LinearSVCTuned.pkl", 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        user_input = request.form['text']
        transformed_input = vectorizer.fit_transform([user_input])
        prediction = model.predict(transformed_input)[0]
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)





# from fastapi import FastAPI,Request, Form
# from fastapi.responses import HTMLResponse
# from pydantic import BaseModel
# import pickle
# from fastapi.templating import Jinja2Templates
# from sklearn.feature_extraction.text import TfidfVectorizer

# app = FastAPI()
# with open("stopwords.txt", "r") as file:
#     stopwords = file.read().splitlines()

# vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=True, vocabulary=pickle.load(open("tfidfvectoizer.pkl", "rb")))
# model = pickle.load(open("LinearSVCTuned.pkl", 'rb'))

# templates = Jinja2Templates(directory="templates")

# # data from form
# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

# @app.post("/", response_class=HTMLResponse)
# async def predict(request: Request, text: str = Form(...)):
#     transformed_input = vectorizer.fit_transform([text])
#     prediction = model.predict(transformed_input)[0]
    
#     return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})
# # till here

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.0", port=8000, reload=True)





# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse  # Import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from pydantic import BaseModel
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer

# app = FastAPI()

# # Load stopwords and models
# with open("stopwords.txt", "r") as file:
#     stopwords = file.read().splitlines()

# vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=True, vocabulary=pickle.load(open("tfidfvectoizer.pkl", "rb")))
# model = pickle.load(open("LinearSVCTuned.pkl", 'rb'))

# # Set up Jinja2 template directory
# templates = Jinja2Templates(directory="templates")

# # GET request for index page
# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

# # POST request to handle form submission
# @app.post("/", response_class=HTMLResponse)
# async def predict(request: Request, text: str = Form(...)):
#     transformed_input = vectorizer.transform([text])  # Use transform instead of fit_transform
#     prediction = model.predict(transformed_input)[0]
    
#     return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)





# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse  # Import HTMLResponse
# from fastapi.templating import Jinja2Templates
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer

# app = FastAPI()

# # Load stopwords
# with open("stopwords.txt", "r") as file:
#     stopwords = file.read().splitlines()

# # Load vectorizer and model
# vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=True, vocabulary=pickle.load(open("tfidfvectoizer.pkl", "rb")))
# model = pickle.load(open("LinearSVCTuned.pkl", 'rb'))

# # Set up Jinja2 template directory
# templates = Jinja2Templates(directory="templates")

# # GET request for index page
# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

# # POST request to handle form submission
# @app.post("/", response_class=HTMLResponse)
# async def predict(request: Request, text: str = Form(...)):
#     transformed_input = vectorizer.transform([text])  # Use transform instead of fit_transform
#     prediction = model.predict(transformed_input)[0]
    
#     return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)




# from fastapi import FastAPI, Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from pydantic import BaseModel
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer

# app = FastAPI()

# # Set up templates directory
# templates = Jinja2Templates(directory="templates")

# # Load stopwords and models
# with open("stopwords.txt", "r") as file:
#     stopwords = file.read().splitlines()

# vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=True, vocabulary=pickle.load(open("tfidfvectoizer.pkl", "rb")))
# model = pickle.load(open("LinearSVCTuned.pkl", 'rb'))

# class TextInput(BaseModel):
#     text: str

# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

# @app.post("/", response_class=HTMLResponse)
# async def predict(request: Request, input: TextInput):
#     user_input = input.text
#     transformed_input = vectorizer.transform([user_input])
#     prediction = model.predict(transformed_input)[0]
#     return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer

# app = FastAPI()

# # Set up templates directory
# templates = Jinja2Templates(directory="templates")

# # Load stopwords and models
# with open("stopwords.txt", "r") as file:
#     stopwords = file.read().splitlines()

# vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=True, vocabulary=pickle.load(open("tfidfvectoizer.pkl", "rb")))
# model = pickle.load(open("LinearSVCTuned.pkl", 'rb'))

# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

# # Accept form data using FastAPI Form
# @app.post("/", response_class=HTMLResponse)
# async def predict(request: Request, text: str = Form(...)):
#     user_input = text
#     transformed_input = vectorizer.transform([user_input])
#     prediction = model.predict(transformed_input)[0]
#     return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer

# app = FastAPI()

# # Set up templates directory
# templates = Jinja2Templates(directory="templates")

# # Load stopwords and models
# with open("stopwords.txt", "r") as file:
#     stopwords = file.read().splitlines()

# vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=True, vocabulary=pickle.load(open("tfidfvectoizer.pkl", "rb")))
# model = pickle.load(open("LinearSVCTuned.pkl", 'rb'))

# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

# @app.post("/", response_class=HTMLResponse)
# async def predict(request: Request, text: str = Form(...)):
#     # Process input text using the vectorizer and model
#     transformed_input = vectorizer.transform([text])
#     prediction = model.predict(transformed_input)[0]
    
#     # Render the template with the prediction result
#     return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
