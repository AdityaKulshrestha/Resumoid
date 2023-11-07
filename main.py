import os
import uvicorn
import matplotlib
from schema import ResumeScores, Resume_API
from fastapi import FastAPI, Request, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from generate_analysis import extract_info, llm_scoring, create_chart_overall, \
    read_pdf_unstructured, suggest_improvements
from dotenv import load_dotenv

matplotlib.use('TKAgg')

app = FastAPI()

app.mount("/styles", StaticFiles(directory="styles"), name='styles')

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def upload_resume(request: Request):
    return templates.TemplateResponse("landing_page.html", {"request": request})


@app.post('/analysis', response_class=HTMLResponse)
async def generate_analysis(
        request: Request,
        resume: UploadFile = File(...),
        job_description: str = Form(...),
):
    """
    Front end based API for resume analysis
    :param request:
    :param resume:
    :param job_description:
    :return: HTML page with generated analysis
    """
    # resume_text = read_pdf(resume.file)
    os.makedirs('tmp', exist_ok=True)
    resume_text = read_pdf_unstructured(resume)
    print(resume_text)
    resume_info = extract_info(resume_text)
    print(resume_info['Experience'])
    llm_scores = llm_scoring(resume_text, job_description)
    print(llm_scores)
    experience_score_chart = create_chart_overall(llm_scores.experience_score)
    education_score_chart = create_chart_overall(llm_scores.education_score)
    suggestion = suggest_improvements(resume_info['Experience'])
    print(suggestion)

    # Add additional data in the repository.
    return templates.TemplateResponse("analysis.html",
                                      {"request": request, 'resume_info': resume_info, 'resume_scores': llm_scores,
                                       'base64_image_1': education_score_chart,
                                       'base64_image_2': experience_score_chart,
                                       'original_list': suggestion.original_task,
                                       'suggestion_list': suggestion.reframed})


@app.get('/analyse-resume', response_model=Resume_API)
async def analyse_resume(personal_details: str, professional_details: str, education_details: str,
                         preferences_details: str, job_description: str):
    """
    Analyzes and generates report on the resume.
    :param personal_details: Personal Details of the applicant
    :param professional_details: Professional Details of the applicant
    :param education_details: Educational Details of the applicant
    :param preferences_details: Addidtional preferences
    :param job_description: Description of the role
    :return: Generated Analysis in JSON format
    """

    resume_text = personal_details + '\n' + professional_details + '\n' + education_details + '\n' + preferences_details

    llm_scores = llm_scoring(resume_text, job_description)
    suggestions = suggest_improvements(professional_details)
    final_eval = Resume_API(resume_eval = llm_scores, suggestions=suggestions)
    return final_eval


if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
