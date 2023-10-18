import os
import PyPDF2
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from models2 import *
from models3 import resume_output_parser
from langchain.llms import OpenAI
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
    tmp_location = os.path.join('resumes', file.name)
    with open(tmp_location, 'wb') as out:
        out.write(file.getbuffer())
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

    return fig


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

    print("INITIAL EXTRACTION")
    print(resume_text)
    # resume_info = parser.parse(resume_text)
    # print("PYDANTIC OUTPUT")
    # print(resume_info)
    format_instructions = resume_output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template="Given a resume {resume} \n Extract all the relevant sections.  \n {format_instructions}",
        input_variables=["resume"],
        partial_variables={"format_instructions": format_instructions}
    )

    _input = prompt.format_prompt(resume=resume)
    print(_input.text)
    # model = OpenAI('gpt-3.5-turbo-16k')
    output = llm.predict(_input.to_string())
    print("INITIAL OUTPUT (STRUCTURED OUTPUT PARSER)")
    print(output)
    extracted_resume = resume_output_parser.parse(output)
    print("STRUCTURED OUTPUT PARSER OUTPUT")
    print(extracted_resume)
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


def llm_scoring(llm, resume_text, job_description):
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
    3. Skills
    HINTS - Compare the skills required for {job_description} with the skills mentioned in the resume.
    4. Projects
    HINTS - Alignment and relevancy of projects with {job_description}. Low score for bad relation and high for high relevancy.



    Here are some rules for the scores
    - Relevant Experience should be high only when the current job is same as {job_description}.
    - Education Experience should be high only when the candidate is from top colleges.


    Take a deep breath and provide the scores and feedback to the candidate in the following format:

    Relevant Experience: {{score_experience}}, Feedback: {{feedback_experience}}
    Education: {{score_education}}, Feedback: {{feedback_education}}
    Skills: {{score_skills}}, Feedback: {{feedback_skills}}
    Projects: {{score_projects}}, Feedback: {{feedback_projects}}
    Overall Score: {{score_overall}}, Feedback: {{feedback_overall}}
    """
    # Ask the LLM to score the resume and provide feedback
    response = llm.predict(prompt)

    parser = OutputFixingParser.from_llm(parser=PydanticOutputParser(pydantic_object=ResumeScores), llm=llm)
    format_instructions = parser.get_format_instructions()

    resume_scores = parser.parse(response)

    return resume_scores


def suggest_improvements(llm, experience):
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


def color_cell(value):
    if value == 'Original Tasks':
        return {
            'backgroundColor': 'white',
            'color': 'red'
        }
    else:
        return {
            'backgroundColor': 'white',
            'color': 'green'
        }


color_cell_js = """
   function(params) {
       if (params.value == 'Original Tasks') {
           return {
               'backgroundColor': 'green',
               'color': 'white'
           }
       } else {
           return {
               'backgroundColor': 'white',
               'color': 'black'
           }
       }
   }
   """


def main():
    st.set_page_config(layout="wide")
    st.title("Welcome to Resumoid 🤖")
    st.subheader("🌝 Your personal AI ATS!")

    st.error(""" 🦺 Built by [Satvik](https://www.linkedin.com/in/satvik-paramkusham/). \n
    Note: This is an alpha version. You may encounter bugs 🐞""")

    # st.markdown("Built by [Build Fast with AI](www.buildfastwithai.com)")

    st.markdown("📄 Upload your resume and job role to get feedback in 2 minutes!")

    resume_pdf = st.file_uploader("Upload your resume", type=['pdf'], label_visibility='collapsed')
    job_description = st.text_input("Enter the role for which you are applying")

    submit = st.button("Submit")

    if resume_pdf and job_description and submit:
        resume_text = read_pdf_unstructured(resume_pdf)
        print("Loaded using Unstructured PDF Loader!")
        # except:
        #     resume_text = read_pdf(resume_pdf)
        resume_info = extract_info(resume_text)
        gpt4_model = ChatOpenAI(model='gpt-3.5-turbo-16k')
        resume_scores = llm_scoring(llm=gpt4_model, resume_text=resume_text, job_description=job_description)

        st.divider()

        st.markdown("### Candidate Details")

        # st.text(resume_info)
        # st.write(resume_info)
        # print(resume_info)
        # name, phone, email

        # name = resume_info.personal_details.name
        # st.markdown("**Name:** " + resume_info.personal_details.name)
        # st.markdown("**Name:** " + name)
        # st.markdown("**Email:** " + resume_info.personal_details.email)
        # st.markdown("**Contact Number:** " + resume_info.personal_details.contact_num)
        # st.markdown("**University:** " + resume_info.education[0].university)
        # st.markdown("**Current Job Role:** " + resume_info.experience[0].company_name)
        # st.markdown("**Company:** " + resume_info.experience[0].job_role)
        st.markdown(resume_info['Personal Details'])

        st.divider()

        ocol1, ocol2, ocol3 = st.columns(3)

        ocol2.markdown("### Relevance Score \n\n\n\n")
        ocol2.pyplot(create_chart_overall(resume_scores.overall_score))
        ocol2.markdown(resume_scores.overall_feedback)

        st.divider()

        st.markdown("### Evaluation")

        st.text(f"Here is the evaluation of your resume for the {job_description} role.")

        col1, col2, col3, col4 = st.columns(4)
        # Column 1
        col1.markdown("### Experience \n\n\n")
        col1.pyplot(create_chart(resume_scores.experience_score))
        col1.markdown(resume_scores.experience_feedback)

        # Column 2
        col2.markdown("### Education \n\n\n")
        col2.pyplot(create_chart(resume_scores.education_score))
        col2.markdown(resume_scores.education_feedback)

        # Column 3
        col3.markdown("### Skills \n\n\n\n")
        col3.pyplot(create_chart(resume_scores.skills_score))
        col3.markdown(resume_scores.skills_feedback)

        # Column 4
        col4.markdown("### Projects \n\n\n\n")
        col4.pyplot(create_chart(resume_scores.projects_score))
        col4.markdown(resume_scores.projects_feedback)

        st.divider()

        st.markdown("### Detailed Comments")
        st.markdown("### Rating skills")
        skills_scoring = rate_skills(resume_text, job_description)
        st.markdown(skills_scoring)
        # col4.pyplot(create_chart(int(skills_scoring)))
        # feedback_jobdesc = description_evaluation(resume_text, job_description)
        # st.markdown(feedback_jobdesc)

        # st.markdown("### Suggestions")
        # output = suggest_improvements(llm, resume_info['Experience'])
        #
        # original_tasks = output.original_task
        # improvised_tasks = output.reframed
        #
        # col4, col5 = st.columns(2)
        # col4.markdown("#### Your Points")
        # col5.markdown("#### Suggested Improvement")
        #
        # for task, suggestion in zip(original_tasks, improvised_tasks):
        #     x1, x2 = st.columns(2)
        #     x1.markdown(f"- :red[{task}]")
        #     x2.markdown(f"- :green[{suggestion}]")
        #     st.markdown("---------------")

        st.divider()

        st.success(""" Chat feature coming soon! \n

        Reach out to me at satvik@buildfastwithai.com""")


if __name__ == '__main__':
    main()
