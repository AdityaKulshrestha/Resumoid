import os
import PyPDF2
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from dotenv import load_dotenv
from schema import *
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate

load_dotenv()

# Defining LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-16k")


def read_pdf(file):
    """
    Reads a resume in PDF file and extract text from it.
    :param file: File object
    :return: String
    """
    reader = PyPDF2.PdfReader(file)
    num_pages = len(reader.pages)
    text = ""
    for i in range(num_pages):
        page = reader.pages[i]
        text += page.extract_text()
    return text


def read_pdf_unstructured(file):
    tmp_location = os.path.join('./tmp', file.filename)
    with open(tmp_location, 'wb') as out:
        out.write(file.file.read())
    loader = UnstructuredPDFLoader(tmp_location, mode='elements')
    docs = loader.load()
    text = ""
    for doc in docs:
        text += doc.page_content
    return text


def create_chart_overall(value: int):
    """
    Return matplotlib.pyplot figure.
    :param value: Integer value.
    :return:
    """
    fig, ax = plt.subplots(figsize=(3, 3))

    value = value * 10
    sizes = [100 - value, value]

    # Define colors (blue for the score, silver for the remaining)
    colors = ['silver', 'blue']

    # Define explode parameters to separate the score section slightly
    explode = (0, 0.1)

    # Create a pie chart with shadows for a 3D effect
    ax.pie(sizes, explode=explode, colors=colors, startangle=90, shadow=True)

    # Draw a white circle in the center
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)
    ax.text(0, 0, f'{value}/100', horizontalalignment='center', verticalalignment='center')

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')

    # Save the figure to a BytesIO object
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)

    # Convert binary image data to a base64 string
    base64_image = base64.b64encode(img_data.read()).decode()

    return base64_image

    # return fig


def create_chart(value: int):
    """
    Return matplotlib.pyplot figure.
    :param value: Intger value.
    :return:
    """
    fig, ax = plt.subplots()

    sizes = [10 - value, value]

    # Create a pie chart
    ax.pie(sizes, colors=['red', 'green'], startangle=90)

    # Draw a white circle in the center
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)
    ax.text(0, 0, f'{value}/10', horizontalalignment='center', verticalalignment='center')

    return fig


def extract_info(resume: str):
    """
    Extracts sections from the resume
    :param resume:
    :return:
    """
    parser = OutputFixingParser.from_llm(parser=PydanticOutputParser(pydantic_object=Resume), llm=llm)
    format_instructions = parser.get_format_instructions()
    resume_text = llm.predict(
        f"Given a resume {resume} \n Extract all the relevant sections including Education, Experience, Personal "
        f"Details, projects and skills. For skills identify all the skills mentioned in the resume.  \n "
        f"{format_instructions}")

    format_instructions = resume_output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template="Given a resume {resume} \n Extract all the relevant sections.  \n {format_instructions}",
        input_variables=["resume"],
        partial_variables={"format_instructions": format_instructions}
    )

    _input = prompt.format_prompt(resume=resume)
    # print(_input.text)
    # # model = OpenAI('gpt-3.5-turbo-16k')
    output = llm.predict(_input.to_string())
    # print("INITIAL OUTPUT (STRUCTURED OUTPUT PARSER)")
    # print(output)
    extracted_resume = resume_output_parser.parse(output)
    # print("STRUCTURED OUTPUT PARSER OUTPUT")
    # print(extracted_resume)
    return extracted_resume


def description_evaluation(resume, job_description):
    prompt_template = f'''You are an Resume Expert. Your job is to give feedback on the resume based on the provided job description.
    Be specific about the points.

    Resume: {resume}

    Job Description: {job_description}

    Please provide the feedback in the following format.

    ## Strengths:
    <list strengths here>

    ## Weaknesses:
    <list weaknesses here>

    ## Recommendations to improve CV:
    <list recommendations here>



    ONLY QUOTE THE INFORMATION PROVIDED IN THE RESUME. DO NOT MAKE UP INFORMATION WHICH IS NOT EXPLICITLY PROVIDED IN RESUME.
    RETURN THE RESPONSE IN MARKDOWN FORMAT IN BULLET POINTS.
    '''
    output = llm.predict(prompt_template)
    return output


def rate_skills(resume, job_description):
    """
    Experimentation for improving skills scoring.
    :param resume:
    :param job_description:
    :return:
    """
    prompt_template = f'''You are an Resume Expert. You have been provided with a resume of a candidate and the role for which the candidate is applying for.

    Resume: {resume}

    Job Description: {job_description}

    You have to accomplish two jobs mentioned below: 

    TASK1 - Based on the {job_description} give all the skills which are ideal for the role and expected a well qualified person to have.

    TASK2 - Extract all the skills from the resume above irrespective of anything or job role. 

    TASK3 - Compare the skills set of the candidate with that of an ideal candidate of the job role. Score the resume on a scale of 1 to 10 (1 being lowers 
    and 10 being highest).

    Return the score only in integer format.
    '''
    output = llm.predict(prompt_template)
    return output


def llm_scoring(resume_text, job_description, llm=llm):
    # Define the prompt
    prompt = f"""
    You are an HR of a company who is hiring for {job_description} role in your company. For the role you have to 
    evaluate a resume strictly based on the hiring role. Also, provide a score between 1 to 10 (where 1 is the 
    lowest and 10 is the highest), and provide feedback for each category and for overall resume using the HINTS for each category: 

    RESUME
    {resume_text}

    CATEGORIES:
    1. Relevant Experience
    HINTS - Experience relevant to the hiring role, Total experience of the industry and position held, kind of work done in previous positions.
    2. Education
    HINTS - Does the hiring role require higher studies or advance degrees, Top college, professional degrees.


    Here are some rules for the scores
    - Relevant Experience should be high only when the current job is same as {job_description}.
    - Education Experience should be high only when the candidate is from top colleges.


    Take a deep breath and provide the scores and feedback to the candidate in the following format:

    Relevant Experience: {{score_experience}}, Feedback: {{feedback_experience}}
    Education: {{score_education}}, Feedback: {{feedback_education}}
    Overall Score: {{score_overall}}, Feedback: {{feedback_overall}}
    """
    # Ask the LLM to score the resume and provide feedback
    response = llm.predict(prompt)

    parser = OutputFixingParser.from_llm(parser=PydanticOutputParser(pydantic_object=ResumeScores), llm=llm)
    format_instructions = parser.get_format_instructions()

    resume_scores = parser.parse(response)
    return resume_scores


def suggest_improvements(experience, llm=llm):
    # Define the prompt
    prompt = f""" Given below is work experience from a resume for the job role

    {experience}
    Please evaluate and provide 
    improvements to the work tasks mentioned in the resume work experience section using the below hints: HINTS: 
    Quantification of work, use of strong action works, overall impact made. 




    Select any 4 to 10 work mentioned in the experience and reframe it for better accepting chances of the resume.

    """
    # Ask the LLM to score the resume and provide feedback
    response = llm.predict(prompt)

    parser = OutputFixingParser.from_llm(parser=PydanticOutputParser(pydantic_object=Suggestion), llm=llm)
    format_instructions = parser.get_format_instructions()

    suggestions = parser.parse(response)

    return suggestions
